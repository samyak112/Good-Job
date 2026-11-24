import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig

from model import GoodJobConfig, GoodJobForSequenceClassification, MAX_LEN, MODEL

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
RUN_DIR = "runs/run"            # folder with best.pt (and optionally tokenizer/)
INPUT_JSON = "splits/test.json"  # list of objects with "post_text" (optional: "id", "author")
OUTPUT_CSV = "results.csv"
BATCH_SIZE = 32
QUANTIZE = False                 # int8 dynamic quantization of Linear layers
THREADS = 0                      # 0 = torch default
WARMUP_BATCHES = 2
SINGLE_POST_SAMPLES = 100        # bs=1 latency measured on this many posts
# ----------------------------------------------------------------------------


def load_model(run_dir, quantize=False):
    run = Path(run_dir)
    ckpt = run / "best.pt"
    if ckpt.exists():
        # same construction as GoodJobForSequenceClassification.from_base(),
        # but without downloading/loading pretrained encoder weights
        d = BertConfig.from_pretrained(MODEL).to_dict()
        for k in ("model_type", "id2label", "label2id", "num_labels",
                  "architectures", "_name_or_path", "transformers_version"):
            d.pop(k, None)
        model = GoodJobForSequenceClassification(GoodJobConfig(**d))
        sd = torch.load(ckpt, map_location="cpu")
        # drop the removed board head's weights (old checkpoints)
        sd = {k: v for k, v in sd.items() if not k.startswith("board_head.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        print(f"checkpoint loaded | missing: {missing} | unexpected: {unexpected}")
    else:
        # folder saved with save_pretrained()
        model = GoodJobForSequenceClassification.from_pretrained(run)

    model.eval()
    if quantize:
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    tok_path = run / "tokenizer"
    tok = AutoTokenizer.from_pretrained(tok_path if tok_path.exists() else MODEL)
    return model, tok


@torch.inference_mode()
def run_batch(model, enc):
    inp = {k: enc[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
    return model(**inp).logits.softmax(-1)


def main():
    if THREADS:
        torch.set_num_threads(THREADS)
    print(f"CPU threads: {torch.get_num_threads()} | quantize: {QUANTIZE}")

    t0 = time.perf_counter()
    model, tok = load_model(RUN_DIR, QUANTIZE)
    print(f"model loaded in {time.perf_counter() - t0:.2f}s")

    cfg = model.config
    type_labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
    max_len = getattr(cfg, "max_seq_length", MAX_LEN)

    rows = json.load(open(INPUT_JSON, encoding="utf-8"))
    texts = [r.get("post_text") or "" for r in rows]
    n = len(texts)

    # length-sorted batches to minimise padding
    order = sorted(range(n), key=lambda i: len(texts[i]))
    batches = [order[s:s + BATCH_SIZE] for s in range(0, n, BATCH_SIZE)]

    def tokenize(idx):
        return tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                   padding=True, return_tensors="pt")

    # ---- warmup (not timed) ----
    for idx in batches[:WARMUP_BATCHES]:
        run_batch(model, tokenize(idx))

    # ---- timed pass, tokenization and model time tracked separately ----
    preds = [None] * n
    tok_time = fwd_time = 0.0
    batch_ms = []
    for idx in batches:
        t = time.perf_counter()
        enc = tokenize(idx)
        t_tok = time.perf_counter() - t

        t = time.perf_counter()
        pt = run_batch(model, enc)
        t_fwd = time.perf_counter() - t

        tok_time += t_tok
        fwd_time += t_fwd
        batch_ms.append((t_tok + t_fwd) * 1000)

        for j, i in enumerate(idx):
            ti = int(pt[j].argmax())
            preds[i] = {
                "type": type_labels[ti],
                "type_conf": round(float(pt[j][ti]), 4),
                "type_probs": {l: round(float(p), 4) for l, p in zip(type_labels, pt[j])},
            }

    total = tok_time + fwd_time
    print("\n=== CPU benchmark (batched) ===")
    print(f"posts            : {n}  (batch size {BATCH_SIZE})")
    print(f"total            : {total * 1000:.1f} ms  (tokenize {tok_time * 1000:.1f} + model {fwd_time * 1000:.1f})")
    print(f"per post         : {total * 1000 / max(n, 1):.2f} ms  (model only {fwd_time * 1000 / max(n, 1):.2f} ms)")
    print(f"throughput       : {n / total:.1f} posts/s")
    bs = sorted(batch_ms)
    print(f"batch latency    : p50 {statistics.median(bs):.1f} ms | "
          f"p95 {bs[min(len(bs) - 1, int(len(bs) * 0.95))]:.1f} ms")

    # ---- single-post latency (bs=1, the real-time case) ----
    sample = order[len(order) // 2:][:SINGLE_POST_SAMPLES] or order[:SINGLE_POST_SAMPLES]
    lat = []
    for i in sample:
        t = time.perf_counter()
        run_batch(model, tokenize([i]))
        lat.append((time.perf_counter() - t) * 1000)
    if lat:
        lat.sort()
        print(f"single post (bs=1, {len(lat)} posts): mean {statistics.mean(lat):.1f} ms | "
              f"p50 {statistics.median(lat):.1f} ms | p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.1f} ms")

    # ---- write results ----
    header = ["id", "author", "post_text", "type", "type_conf",
              *[f"prob_{l}" for l in type_labels]]
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r, t, p in zip(rows, texts, preds):
            w.writerow([r.get("id", ""), r.get("author", "") or "", t,
                        p["type"], p["type_conf"],
                        *[p["type_probs"][l] for l in type_labels]])
    print(f"\nwrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()