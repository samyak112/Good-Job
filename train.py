"""
Multi-task BERT classifier `good_job`: shared encoder, two heads.
  head 1: job type (ML Job / Job / Other / Not Job)
  head 2: job board (poster == "Job board" -> 1, anything else, including missing -> 0)

All settings are in config.py. Run:  python train_good_job.py
Splits come from split_and_sample.py (train.json has a `sample_weight` field).

Requires: torch, transformers, scikit-learn, numpy
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config as C
from config import BOARD_LABELS, BOARD_VALUE, TYPE_LABELS
from model import GoodJob

MODEL_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_examples(path):
    rows = read_json(path)
    examples = [{
        "text": r.get("post_text") or "",
        "author": r.get("author") or "",
        "type_y": TYPE_LABELS.index(r["label"]),
        "board_y": int(r.get("poster") == BOARD_VALUE),
    } for r in rows]
    return rows, examples


def make_collate(tok):
    def collate(batch):
        texts = [b["text"] for b in batch]
        if C.USE_AUTHOR:
            enc = tok([b["author"] for b in batch], texts, truncation="only_second",
                      max_length=C.MAX_LEN, padding=True, return_tensors="pt")
        else:
            enc = tok(texts, truncation=True, max_length=C.MAX_LEN, padding=True, return_tensors="pt")
        enc["type_y"] = torch.tensor([b["type_y"] for b in batch])
        enc["board_y"] = torch.tensor([b["board_y"] for b in batch])
        return enc
    return collate


def model_inputs(batch, device):
    return {k: batch[k].to(device) for k in MODEL_KEYS if k in batch}

def inverse_freq_weights(y, n_classes, power=1.0):
    counts = np.bincount(np.array(y), minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    w = counts ** (-power)
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def head_metrics(prefix, true, pred, names):
    ids = range(len(names))
    return {
        f"{prefix}_macro_f1": f1_score(true, pred, average="macro", labels=ids, zero_division=0),
        f"{prefix}_report": classification_report(true, pred, labels=ids, target_names=names,
                                                  output_dict=True, zero_division=0),
        f"{prefix}_confusion": confusion_matrix(true, pred, labels=ids).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp, tt, bp, bt = [], [], [], []
    for b in loader:
        lt, lb = model(**model_inputs(b, device))
        tp += lt.argmax(-1).cpu().tolist()
        tt += b["type_y"].tolist()
        bp += lb.argmax(-1).cpu().tolist()
        bt += b["board_y"].tolist()
    return {**head_metrics("type", tt, tp, TYPE_LABELS),
            **head_metrics("board", bt, bp, BOARD_LABELS)}


def print_summary(name, m):
    print(f"[{name}] type macro-F1 {m['type_macro_f1']:.4f} | board macro-F1 {m['board_macro_f1']:.4f}")
    for l in TYPE_LABELS:
        r = m["type_report"][l]
        print(f"    {l:<8} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}  (n={int(r['support'])})")
    print("    type confusion (rows=true, cols=pred):", m["type_confusion"])
    print("    board confusion:", m["board_confusion"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    assert C.IMBALANCE in ("none", "weights", "sampler"), f"bad IMBALANCE: {C.IMBALANCE}"
    assert C.SELECT_ON in ("type", "board", "both"), f"bad SELECT_ON: {C.SELECT_ON}"

    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(C.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    splits = Path(C.SPLITS_DIR)

    train_rows, train_ds = load_examples(splits / "train.json")
    _, val_ds = load_examples(splits / "val.json")

    tok = AutoTokenizer.from_pretrained(C.MODEL)
    collate = make_collate(tok)

    # train-time imbalance handling (val/test are never touched)
    sampler, type_w = None, None
    if C.IMBALANCE == "sampler":
        w = np.array([r["sample_weight"] for r in train_rows], dtype=float)
        sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)
    elif C.IMBALANCE == "weights":
        type_w = inverse_freq_weights([e["type_y"] for e in train_ds], len(TYPE_LABELS),
                                      C.CLASS_POWER).to(device)
    board_w = (inverse_freq_weights([e["board_y"] for e in train_ds], len(BOARD_LABELS)).to(device)
               if C.BOARD_WEIGHTS else None)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, sampler=sampler,
                              shuffle=sampler is None, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    model = GoodJob(C.MODEL).to(device)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": C.LR}, {"params": head_params, "lr": C.HEAD_LR}],
        weight_decay=C.WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * C.EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(C.WARMUP * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    type_loss = nn.CrossEntropyLoss(weight=type_w)
    board_loss = nn.CrossEntropyLoss(weight=board_w)

    best, bad, history = -1.0, 0, []
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        run = 0.0
        for b in train_loader:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                lt, lb = model(**model_inputs(b, device))
                loss = (C.LAMBDA_TYPE * type_loss(lt.float(), b["type_y"].to(device))
                        + C.LAMBDA_BOARD * board_loss(lb.float(), b["board_y"].to(device)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        m = evaluate(model, val_loader, device)
        score = {"type": m["type_macro_f1"], "board": m["board_macro_f1"],
                 "both": (m["type_macro_f1"] + m["board_macro_f1"]) / 2}[C.SELECT_ON]
        print(f"epoch {epoch}: train loss {run / len(train_loader):.4f}")
        print_summary("val", m)
        history.append({"epoch": epoch, "train_loss": run / len(train_loader), "val": m})

        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best.pt")
            (out / "val_best.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print(f"early stop (no val improvement for {C.PATIENCE} epochs)")
                break

    tok.save_pretrained(out / "tokenizer")
    settings = {k.lower(): v for k, v in vars(C).items() if k.isupper()}
    (out / "config.json").write_text(json.dumps({
        "type_labels": TYPE_LABELS, "board_labels": BOARD_LABELS, "model": C.MODEL,
        "max_len": C.MAX_LEN, "use_author": C.USE_AUTHOR, "args": settings}, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best val score ({C.SELECT_ON}): {best:.4f}  -> {out / 'best.pt'}")

    if C.EVAL_TEST:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device))
        _, test_ds = load_examples(splits / "test.json")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)
        m = evaluate(model, test_loader, device)
        print_summary("TEST", m)
        (out / "test.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()