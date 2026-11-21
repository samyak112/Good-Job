"""
Standalone evaluation for the trained `good_job` model. No arguments: edit the
constants below, then run:  python eval_good_job.py

Reads everything it needs about the run (labels, model name, max_len, use_author)
from OUT_DIR/config.json written by train_good_job.py, loads OUT_DIR/best.pt and
OUT_DIR/tokenizer, evaluates on each split in SPLITS_TO_EVAL, prints a summary
and writes OUT_DIR/eval_<split>.json.

Requires: torch, transformers, scikit-learn, numpy, model.py (GoodJobForSequenceClassification)
"""
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig

from model import GoodJobConfig, GoodJobForSequenceClassification

# ---- settings (edit here) ---------------------------------------------------
OUT_DIR = "runs/run"            # same as OUT_DIR in config.py
SPLITS_DIR = "splits"      # same as SPLITS_DIR in config.py
SPLITS_TO_EVAL = ("val", "test")
BATCH_SIZE = 64
BOARD_VALUE = "Job board"  # same as BOARD_VALUE in config.py
# -----------------------------------------------------------------------------


def load_examples(path, type_labels):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [{
        "text": r.get("post_text") or "",
        "author": r.get("author") or "",
        "type_y": type_labels.index(r["label"]),
        "board_y": int(r.get("poster") == BOARD_VALUE),
    } for r in rows]


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


def build_model(cfg, device):
    """Architecture only (no pretrained download); weights come from best.pt."""
    d = BertConfig.from_pretrained(cfg["model"]).to_dict()
    for k in ("model_type", "id2label", "label2id", "num_labels",
              "architectures", "_name_or_path", "transformers_version"):
        d.pop(k, None)
    model = GoodJobForSequenceClassification(
        GoodJobConfig(board_labels=cfg["board_labels"], max_seq_length=cfg["max_len"], **d))
    state = torch.load(Path(OUT_DIR) / "best.pt", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def head_metrics(prefix, true, pred, names):
    ids = range(len(names))
    return {
        f"{prefix}_macro_f1": f1_score(true, pred, average="macro", labels=ids, zero_division=0),
        f"{prefix}_report": classification_report(true, pred, labels=ids, target_names=names,
                                                  output_dict=True, zero_division=0),
        f"{prefix}_confusion": confusion_matrix(true, pred, labels=ids).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device, type_labels, board_labels):
    tp, tt, bp, bt = [], [], [], []
    keys = ("input_ids", "attention_mask", "token_type_ids")
    for b in loader:
        out = model(**{k: b[k].to(device) for k in keys if k in b})
        tp += out.logits.argmax(-1).cpu().tolist()
        tt += b["type_y"].tolist()
        keep = b["type_y"] != type_labels.index("Not Job")
        bp += out.board_logits.argmax(-1).cpu()[keep].tolist()
        bt += b["board_y"][keep].tolist()
    return {**head_metrics("type", tt, tp, type_labels),
            **head_metrics("board", bt, bp, board_labels)}


def print_summary(name, m, type_labels):
    print(f"[{name}] type macro-F1 {m['type_macro_f1']:.4f} | board macro-F1 {m['board_macro_f1']:.4f}")
    for l in type_labels:
        r = m["type_report"][l]
        print(f"    {l:<8} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}  (n={int(r['support'])})")
    print("    type confusion (rows=true, cols=pred):", m["type_confusion"])
    print("    board confusion:", m["board_confusion"])


def main():
    out = Path(OUT_DIR)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    type_labels, board_labels = cfg["type_labels"], cfg["board_labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(out / "tokenizer")
    collate = make_collate(tok, cfg["max_len"], cfg["use_author"])
    model = build_model(cfg, device)

    for split in SPLITS_TO_EVAL:
        ds = load_examples(Path(SPLITS_DIR) / f"{split}.json", type_labels)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        m = evaluate(model, loader, device, type_labels, board_labels)
        print_summary(split.upper(), m, type_labels)
        (out / f"eval_{split}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()