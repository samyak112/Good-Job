#!/usr/bin/env python3
"""
Multi-task BERT classifier `good_job`: shared encoder, two heads.
  head 1: job type (ML Job / Job / Other / Not Job)
  head 2: job board (poster == "Job board" -> 1, any other value -> 0, missing -> ignored)

Requires: torch, transformers, scikit-learn, numpy
Splits come from split_and_sample.py (train.json has a `sample_weight` field).

Examples:
  python train_good_job.py --splits-dir splits --out-dir runs/base
  python train_good_job.py --splits-dir splits --imbalance weights --out-dir runs/cw
  python train_good_job.py --splits-dir splits --imbalance sampler --out-dir runs/sampler
  python train_good_job.py --splits-dir splits --use-author --out-dir runs/author
  python train_good_job.py --splits-dir splits --out-dir runs/final --eval-test   # once, at the end
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

TYPE_LABELS = ["ML Job", "Job", "Other", "Not Job"]
BOARD_LABELS = ["Regular account", "Job board"]
BOARD_VALUE = "Job board"  # value of `poster` that means job board; everything else = regular


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class PostDataset(Dataset):
    def __init__(self, rows, use_author):
        self.rows = rows
        self.use_author = use_author

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        type_y = TYPE_LABELS.index(r["label"])
        board_y = int(r.get("poster") == BOARD_VALUE)   # null or any other value -> 0
        text = r.get("post_text") or ""
        return {"text": text, "author": r.get("author") or "", "type_y": type_y, "board_y": board_y}


def make_collate(tok, max_len, use_author):
    def collate(batch):
        texts = [b["text"] for b in batch]
        if use_author:
            enc = tok([b["author"] for b in batch], texts, truncation="only_second",
                      max_length=max_len, padding=True, return_tensors="pt")
        else:
            enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["type_y"] = torch.tensor([b["type_y"] for b in batch])
        enc["board_y"] = torch.tensor([b["board_y"] for b in batch])
        return enc
    return collate


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class GoodJob(nn.Module):
    def __init__(self, name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.type_head = nn.Linear(h, len(TYPE_LABELS))
        self.board_head = nn.Linear(h, len(BOARD_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        cls = self.encoder(**kw).last_hidden_state[:, 0]  # [CLS]
        z = self.drop(cls)
        return self.type_head(z), self.board_head(z)


def inverse_freq_weights(y, n_classes, power=1.0):
    y = np.array([v for v in y if v >= 0])
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    w = counts ** (-power)
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp, tt, bp, bt = [], [], [], []
    for b in loader:
        inp = {k: b[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
        lt, lb = model(**inp)
        tp += lt.argmax(-1).cpu().tolist()
        tt += b["type_y"].tolist()
        bp += lb.argmax(-1).cpu().tolist()
        bt += b["board_y"].tolist()
    out = {
        "type_macro_f1": f1_score(tt, tp, average="macro", labels=range(len(TYPE_LABELS)), zero_division=0),
        "type_report": classification_report(tt, tp, labels=range(len(TYPE_LABELS)),
                                             target_names=TYPE_LABELS, output_dict=True, zero_division=0),
        "type_confusion": confusion_matrix(tt, tp, labels=range(len(TYPE_LABELS))).tolist(),
    }
    keep = [i for i, v in enumerate(bt) if v >= 0]
    if keep:
        bt_, bp_ = [bt[i] for i in keep], [bp[i] for i in keep]
        out["board_macro_f1"] = f1_score(bt_, bp_, average="macro", labels=[0, 1], zero_division=0)
        out["board_report"] = classification_report(bt_, bp_, labels=[0, 1], target_names=BOARD_LABELS,
                                                    output_dict=True, zero_division=0)
        out["board_confusion"] = confusion_matrix(bt_, bp_, labels=[0, 1]).tolist()
    return out


def print_summary(name, m):
    print(f"[{name}] type macro-F1 {m['type_macro_f1']:.4f}", end="")
    if "board_macro_f1" in m:
        print(f" | board macro-F1 {m['board_macro_f1']:.4f}", end="")
    print()
    for l in TYPE_LABELS:
        r = m["type_report"][l]
        print(f"    {l:<8} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}  (n={int(r['support'])})")
    print("    type confusion (rows=true, cols=pred):", m["type_confusion"])
    if "board_confusion" in m:
        print("    board confusion:", m["board_confusion"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--out-dir", default="runs/run")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--lambda-type", type=float, default=1.0)
    ap.add_argument("--lambda-board", type=float, default=1.0)
    ap.add_argument("--imbalance", choices=["none", "weights", "sampler"], default="none",
                    help="job-type imbalance handling: pick ONE")
    ap.add_argument("--class-power", type=float, default=1.0, help="exponent for --imbalance weights")
    ap.add_argument("--board-weights", action="store_true", help="inverse-frequency weights for job-board head")
    ap.add_argument("--use-author", action="store_true", help="feed `author [SEP] post_text`")
    ap.add_argument("--select-on", choices=["type", "board", "both"], default="type")
    ap.add_argument("--eval-test", action="store_true", help="evaluate on test set (do this ONCE, at the end)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    load = lambda n: json.load(open(Path(args.splits_dir) / f"{n}.json", encoding="utf-8"))
    train_rows, val_rows = load("train"), load("val")

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = make_collate(tok, args.max_len, args.use_author)
    train_ds, val_ds = PostDataset(train_rows, args.use_author), PostDataset(val_rows, args.use_author)

    # train-time imbalance handling (val/test are never touched)
    type_y = [train_ds[i]["type_y"] for i in range(len(train_ds))]
    board_y = [train_ds[i]["board_y"] for i in range(len(train_ds))]
    sampler = None
    type_w = None
    if args.imbalance == "sampler":
        w = np.array([r["sample_weight"] for r in train_rows], dtype=float)
        sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)
    elif args.imbalance == "weights":
        type_w = inverse_freq_weights(type_y, len(TYPE_LABELS), args.class_power).to(device)
    board_w = inverse_freq_weights(board_y, 2).to(device) if args.board_weights else None

    train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, shuffle=sampler is None,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    model = GoodJob(args.model).to(device)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    type_loss = nn.CrossEntropyLoss(weight=type_w)
    board_loss = nn.CrossEntropyLoss(weight=board_w, ignore_index=-100)

    best, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for step, b in enumerate(train_loader, 1):
            inp = {k: b[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
            ty, by = b["type_y"].to(device), b["board_y"].to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                lt, lb = model(**inp)
                l_type = type_loss(lt.float(), ty)
                # CE over an all-ignored batch is NaN, so guard it
                l_board = board_loss(lb.float(), by) if (by >= 0).any() else lt.sum() * 0
                loss = args.lambda_type * l_type + args.lambda_board * l_board
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        m = evaluate(model, val_loader, device)
        score = {"type": m["type_macro_f1"], "board": m.get("board_macro_f1", 0.0),
                 "both": (m["type_macro_f1"] + m.get("board_macro_f1", 0.0)) / 2}[args.select_on]
        print(f"epoch {epoch}: train loss {run/len(train_loader):.4f}")
        print_summary("val", m)
        history.append({"epoch": epoch, "train_loss": run / len(train_loader), "val": m})
        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best.pt")
            json.dump(m, open(out / "val_best.json", "w"), indent=2)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no val improvement for {args.patience} epochs)")
                break

    tok.save_pretrained(out / "tokenizer")
    json.dump({"type_labels": TYPE_LABELS, "board_labels": BOARD_LABELS, "model": args.model,
               "max_len": args.max_len, "use_author": args.use_author, "args": vars(args)},
              open(out / "config.json", "w"), indent=2)
    json.dump(history, open(out / "history.json", "w"), indent=2)
    print(f"best val score ({args.select_on}): {best:.4f}  -> {out/'best.pt'}")

    if args.eval_test:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        test_ds = PostDataset(load("test"), args.use_author)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)
        m = evaluate(model, test_loader, device)
        print_summary("TEST", m)
        json.dump(m, open(out / "test.json", "w"), indent=2)


if __name__ == "__main__":
    main()