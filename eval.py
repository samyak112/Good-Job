"""
Evaluate the trained good_job model on val and test (accuracy per label)
plus CPU inference speed. No arguments: edit constants, run:  python eval_good_job.py
"""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig

from model import GoodJobConfig, GoodJobForSequenceClassification

# ---- settings (edit here) ---------------------------------------------------
OUT_DIR = "runs/run"                  # folder with best.pt, config.json, tokenizer/
SPLITS_DIR = "splits"
SPLITS_TO_EVAL = ("val", "test")
BATCH_SIZE = 32
DEVICE = "cpu"                        # "cpu" or "cuda"
THREADS = 0                           # 0 = torch default
QUANTIZE = False                      # int8 dynamic quantization (CPU only)
WARMUP_BATCHES = 2
# -----------------------------------------------------------------------------


def load_examples(path, type_labels):
    rows = json.load(open(path, encoding="utf-8"))
    ex = [{
        "text": r.get("post_text") or "",
        "author": r.get("author") or "",
        "type_y": type_labels.index(r["label"]),
    } for r in rows]
    ex.sort(key=lambda e: len(e["text"]))   # less padding; metrics are order-independent
    return ex


def make_collate(tok, max_len, use_author):
    def collate(batch):
        texts = [b["text"] for b in batch]
        if use_author:
            enc = tok([b["author"] for b in batch], texts, truncation="only_second",
                      max_length=max_len, padding=True, return_tensors="pt")
        else:
            enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["type_y"] = torch.tensor([b["type_y"] for b in batch])
        return enc
    return collate


def build_model(cfg, device, quantize):
    d = BertConfig.from_pretrained(cfg["model"]).to_dict()
    for k in ("model_type", "id2label", "label2id", "num_labels",
              "architectures", "_name_or_path", "transformers_version"):
        d.pop(k, None)
    model = GoodJobForSequenceClassification(
        GoodJobConfig(max_seq_length=cfg["max_len"], **d))
    sd = torch.load(Path(OUT_DIR) / "best.pt", map_location="cpu")
    # drop the removed board head's weights (old checkpoints)
    sd = {k: v for k, v in sd.items() if not k.startswith("board_head.")}
    model.load_state_dict(sd)
    model.eval()
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model.to(device)



@torch.inference_mode()
def run(model, loader, device):
    keys = ("input_ids", "attention_mask", "token_type_ids")
    tt, tp = [], []

    # warmup (not timed, not scored)
    for i, b in enumerate(loader):
        if i >= WARMUP_BATCHES:
            break
        model(**{k: b[k].to(device) for k in keys if k in b})

    fwd = 0.0
    for b in loader:
        inp = {k: b[k].to(device) for k in keys if k in b}

        t = time.perf_counter()
        out = model(**inp)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd += time.perf_counter() - t

        tp += out.logits.argmax(-1).cpu().tolist()
        tt += b["type_y"].tolist()

    return tt, tp, fwd


def collect_errors(ds, true, pred, names):
    errors = []

    for example, expected, predicted in zip(ds, true, pred):
        if expected == predicted:
            continue

        errors.append({
            "input": example["text"],
            "author": example["author"],
            "expected": names[expected],
            "predicted": names[predicted],
        })

    return errors



def acc_by_class(true, pred, names):
    """Returns {name: (correct, n)} per true class."""
    res = {}
    for i, name in enumerate(names):
        idx = [j for j, t in enumerate(true) if t == i]
        res[name] = (sum(pred[j] == i for j in idx), len(idx))
    return res


def fmt(c, n):
    return f"{c / n:6.2%}  ({c}/{n})" if n else "   n/a  (0/0)"


def error_distribution(title, true, pred, names):
    """For each true label: how often it's wrong, and what it gets predicted as instead."""
    print(f"\n{title}")
    res = {}
    for i, name in enumerate(names):
        n = sum(t == i for t in true)
        wrong = [p for t, p in zip(true, pred) if t == i and p != i]
        k = len(wrong)
        dist = {names[j]: wrong.count(j) for j in range(len(names)) if j != i}
        res[name] = {"n": n, "n_wrong": k,
                     "predicted_as": {l: {"count": c, "share_of_errors": (c / k if k else 0.0)}
                                      for l, c in dist.items()}}
        print(f"  {name}: {k} wrong out of {n} ({k / n:.1%})" if n else f"  {name}: no samples")
        for l, c in sorted(dist.items(), key=lambda x: -x[1]):
            share = f"{c / k:6.1%}" if k else "   n/a"
            print(f"      -> predicted {l:<16} {c:>4}  ({share} of errors)")
    return res



def report(split, model, tok, cfg, out):
    type_labels = cfg["type_labels"]

    ds = load_examples(Path(SPLITS_DIR) / f"{split}.json", type_labels)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate(tok, cfg["max_len"], cfg["use_author"]),
    )

    tt, tp, fwd = run(model, loader, DEVICE)
    n = len(tt)
    S = split.upper()

    # ---- type head ----
    type_correct = sum(a == b for a, b in zip(tt, tp))
    type_acc = type_correct / n
    per_label = acc_by_class(tt, tp, type_labels)

    cm = [[0] * len(type_labels) for _ in type_labels]
    for a, b in zip(tt, tp):
        cm[a][b] += 1

    print(f"\n=== [{S}] TYPE HEAD ({n} posts) ===")
    print(f"overall accuracy : {fmt(type_correct, n)}")

    for l, (c, m) in per_label.items():
        print(f"  {l:<10} {fmt(c, m)}")

    print("confusion (rows=true, cols=pred):", type_labels)
    for l, row in zip(type_labels, cm):
        print(f"  {l:<10} {row}")

    type_err = error_distribution(
        f"[{S}] TYPE HEAD: what wrong predictions turn into",
        tt,
        tp,
        type_labels,
    )

    # ---- collect incorrect predictions ----
    errors = []

    for example, expected, predicted in zip(ds, tt, tp):
        if expected == predicted:
            continue

        errors.append({
            "input": example["text"],
            "author": example["author"],
            "expected": type_labels[expected],
            "predicted": type_labels[predicted],
        })

    error_path = out / f"errors_{split}.json"
    error_path.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nIncorrect predictions: {len(errors)}/{n}")
    print(f"wrote {error_path}")

    # ---- speed ----
    print(
        f"\n=== [{S}] SPEED ({DEVICE}, "
        f"threads={torch.get_num_threads()}, "
        f"bs={BATCH_SIZE}, quantize={QUANTIZE}) ==="
    )
    print(
        f"model forward only: {fwd * 1000:.1f} ms total | "
        f"{fwd * 1000 / n:.2f} ms/post | "
        f"{n / fwd:.1f} posts/s"
    )

    # ---- save evaluation results ----
    result = {
        "split": split,
        "n": n,
        "type_accuracy": type_acc,
        "type_per_label": {
            l: {"correct": c, "n": m}
            for l, (c, m) in per_label.items()
        },
        "type_confusion": cm,
        "type_error_distribution": type_err,
        "incorrect_predictions": len(errors),
        "error_file": str(error_path),
        "ms_per_post": fwd * 1000 / n,
    }

    result_path = out / f"eval_{split}.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"wrote {result_path}")

    return result

def main():
    if THREADS:
        torch.set_num_threads(THREADS)
    out = Path(OUT_DIR)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(out / "tokenizer")
    model = build_model(cfg, DEVICE, QUANTIZE)

    results = {s: report(s, model, tok, cfg, out) for s in SPLITS_TO_EVAL}

    print("\n=== SUMMARY ===")
    print(f"{'split':<6}{'n':>6}{'type acc':>10}{'ms/post':>9}")
    for s, r in results.items():
        print(f"{s:<6}{r['n']:>6}{r['type_accuracy']:>10.2%}{r['ms_per_post']:>9.2f}")


if __name__ == "__main__":
    main()