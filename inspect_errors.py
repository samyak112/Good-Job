#!/usr/bin/env python3
"""
Dump the posts a trained good_job model gets wrong, so you can read them.

Run from the folder that contains train.py (it imports the model and dataset code from it):
    python inspect_errors.py --run-dir runs/base2 --splits-dir splits --split val

Outputs (in the run dir):
    errors_type_<split>.csv    wrong job-type predictions, most confident mistakes first
    errors_board_<split>.csv   wrong job-board predictions, most confident mistakes first
    predictions_<split>.json   every prediction with probabilities

Most useful to read: HIGH-confidence mistakes. Those are often label errors, or posts that
fit two classes. Low-confidence mistakes are usually just hard or borderline.
"""
import argparse
import csv
import importlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--module", default="train", help="python module holding GoodJob/PostDataset (train.py)")
    ap.add_argument("--show", type=int, default=15, help="how many mistakes to print per error type")
    args = ap.parse_args()

    t = importlib.import_module(args.module)
    run = Path(args.run_dir)
    cfg = json.load(open(run / "config.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(run / "tokenizer")
    model = t.GoodJob(cfg["model"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    rows = json.load(open(Path(args.splits_dir) / f"{args.split}.json", encoding="utf-8"))
    ds = t.PostDataset(rows, cfg["use_author"])
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=t.make_collate(tok, cfg["max_len"], cfg["use_author"]))

    type_probs, board_probs = [], []
    with torch.no_grad():
        for b in loader:
            inp = {k: b[k].to(device) for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
            lt, lb = model(**inp)
            type_probs += lt.softmax(-1).cpu().tolist()
            board_probs += lb.softmax(-1).cpu().tolist()

    recs = []
    for r, tp, bp in zip(rows, type_probs, board_probs):
        item = ds[len(recs)]
        tpred, bpred = max(range(len(tp)), key=tp.__getitem__), max(range(len(bp)), key=bp.__getitem__)
        recs.append({
            "id": r.get("id", ""), "source": r.get("source", ""), "author": r.get("author", ""),
            "url": r.get("url", ""), "text": (r.get("post_text") or "").replace("\n", " "),
            "type_true": t.TYPE_LABELS[item["type_y"]], "type_pred": t.TYPE_LABELS[tpred],
            "type_conf": round(tp[tpred], 3),
            "board_true": t.BOARD_LABELS[item["board_y"]] if item["board_y"] >= 0 else "",
            "board_pred": t.BOARD_LABELS[bpred], "board_conf": round(bp[bpred], 3),
            "type_probs": [round(x, 3) for x in tp], "board_probs": [round(x, 3) for x in bp],
        })
    json.dump(recs, open(run / f"predictions_{args.split}.json", "w"), indent=2, ensure_ascii=False)

    def dump(errs, name, cols):
        path = run / f"{name}_{args.split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(errs)
        return path

    cols_t = ["type_true", "type_pred", "type_conf", "source", "author", "text", "url", "id"]
    cols_b = ["board_true", "board_pred", "board_conf", "source", "author", "text", "url", "id"]
    type_err = sorted([r for r in recs if r["type_true"] != r["type_pred"]], key=lambda r: -r["type_conf"])
    board_err = sorted([r for r in recs if r["board_true"] != "" and r["board_true"] != r["board_pred"]],
                       key=lambda r: -r["board_conf"])
    p1, p2 = dump(type_err, "errors_type", cols_t), dump(board_err, "errors_board", cols_b)

    print(f"{args.split}: {len(type_err)}/{len(recs)} type errors, {len(board_err)} board errors")
    print("\nType error pairs (true -> pred):")
    for (a, b), n in Counter((r["type_true"], r["type_pred"]) for r in type_err).most_common():
        print(f"  {a:<8} -> {b:<8} {n}")

    print(f"\nMost confident type mistakes (read these first):")
    for r in type_err[: args.show]:
        print(f"  [{r['type_true']} -> {r['type_pred']} {r['type_conf']:.2f}] {r['author']}: {r['text'][:220]}")
    print(f"\nMost confident board mistakes:")
    for r in board_err[: args.show]:
        print(f"  [{r['board_true']} -> {r['board_pred']} {r['board_conf']:.2f}] {r['author']}: {r['text'][:220]}")
    print(f"\nSaved: {p1}\n       {p2}")


if __name__ == "__main__":
    main()