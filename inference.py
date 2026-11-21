import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import GoodJob

# ----------------------------------------------------------------------------
# CONFIG - change these
# ----------------------------------------------------------------------------
RUN_DIR = "runs/base"        # folder with best.pt, config.json, tokenizer/
INPUT_JSON = "data_pipeline/dataset/test_data.json"      # list of objects with "post_text" (optional: "id", "author")
OUTPUT_CSV = "results.csv"
BATCH_SIZE = 32
QUANTIZE = False               # int8 quantization of Linear layers
THREADS = 0                    # torch CPU threads (0 = default)


# ----------------------------------------------------------------------------
class Predictor:
    def __init__(self, run_dir, quantize=False):
        run = Path(run_dir)
        self.cfg = json.load(open(run / "config.json", encoding="utf-8"))
        self.type_labels = self.cfg["type_labels"]
        self.board_labels = self.cfg["board_labels"]
        self.max_len = self.cfg["max_len"]
        self.use_author = self.cfg["use_author"]

        self.tok = AutoTokenizer.from_pretrained(run / "tokenizer")
        self.model = GoodJob(self.cfg["model"])
        self.model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        self.model.eval()
        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)

    def encode(self, texts, authors=None):
        if self.use_author:
            authors = authors or [""] * len(texts)
            return self.tok(authors, texts, truncation="only_second", max_length=self.max_len,
                            padding=True, return_tensors="pt")
        return self.tok(texts, truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt")

    @torch.inference_mode()
    def forward_logits(self, enc):
        inp = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
        return self.model(**inp)

    def predict(self, texts, authors=None, bs=32):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            t = [texts[i] for i in idx]
            a = [authors[i] for i in idx] if authors else None
            lt, lb = self.forward_logits(self.encode(t, a))
            pt, pb = lt.softmax(-1), lb.softmax(-1)
            for j, i in enumerate(idx):
                ti, bi = int(pt[j].argmax()), int(pb[j].argmax())
                out[i] = {
                    "type": self.type_labels[ti],
                    "type_conf": round(float(pt[j][ti]), 4),
                    "type_probs": {l: round(float(p), 4) for l, p in zip(self.type_labels, pt[j])},
                    "board": self.board_labels[bi],
                    "board_conf": round(float(pb[j][bi]), 4),
                }
        return out


def main():
    if THREADS:
        torch.set_num_threads(THREADS)

    t0 = time.perf_counter()
    pred = Predictor(RUN_DIR, quantize=QUANTIZE)
    print(f"model loaded in {time.perf_counter() - t0:.2f}s")

    rows = json.load(open(INPUT_JSON, encoding="utf-8"))
    texts = [r.get("post_text") or "" for r in rows]
    authors = [r.get("author") or "" for r in rows]

    t0 = time.perf_counter()
    preds = pred.predict(texts, authors, bs=BATCH_SIZE)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{len(texts)} posts in {ms:.1f} ms ({ms / max(len(texts), 1):.2f} ms/post)")

    labels = pred.type_labels
    header = ["id", "author", "post_text", "type", "type_conf",
              *[f"prob_{l}" for l in labels], "board", "board_conf"]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r, t, p in zip(rows, texts, preds):
            w.writerow([r.get("id", ""), r.get("author", "") or "", t,
                        p["type"], p["type_conf"],
                        *[p["type_probs"][l] for l in labels],
                        p["board"], p["board_conf"]])
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()