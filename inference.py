import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import GoodJob


class Predictor:
    def __init__(self, run_dir, quantize=False):
        run = Path(run_dir)
        self.cfg = json.load(open(run / "config.json", encoding="utf-8"))
        self.type_labels = self.cfg["type_labels"]
        self.board_labels = self.cfg["board_labels"]
        self.max_len = self.cfg["max_len"]
        self.use_author = self.cfg["use_author"]

        self.tok = AutoTokenizer.from_pretrained(run / "tokenizer")
        self.model = GoodJob(self.cfg["model"], len(self.type_labels), len(self.board_labels))
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1000, r


def load_posts(path):
    rows = json.load(open(path, encoding="utf-8"))
    return [r.get("post_text") or "" for r in rows], [r.get("author") or "" for r in rows], rows


SAMPLES = [
    "Perplexity is hiring a Member of Technical Staff (Software Engineer, Multimodal) in San Francisco. $220K-$405K + equity.",
    "Hiring Alert ! micro1 Is Hiring a Remote AI Jailbreak & Prompt-Injection Security Expert PAY: $50-$90/Hour This job is fully remote.",
    "Come work with us and be our colleague!",
    "2/ All of this is a massive opportunity to build at the frontier of Applied AI. Momentum is exceptional: Finch is growing 1,000%+ YoY",
    "Askable is hiring Software Engineers - AI Coding Tasks $100/hour Remote Part-time contract opportunity AI coding & software engineering tasks",
]


def print_pred(text, p):
    print(f"  {p['type']:<8} ({p['type_conf']:.2f}) | {p['board']:<16} | {text[:70]!r}")


# ----------------------------------------------------------------------------
# Mode 1: predict
# ----------------------------------------------------------------------------
def run_predict(pred, args):
    if args.text:
        ms, r = timed(lambda: pred.predict([args.text], [args.author]))
        print(json.dumps(r[0], indent=2))
        print(f"(cold single request, incl. first-call overhead: {ms:.1f} ms)")
        return

    if args.input:
        texts, authors, rows = load_posts(args.input)
        ms, preds = timed(lambda: pred.predict(texts, authors, bs=args.bs))
        print(f"{len(texts)} posts in {ms:.1f} ms ({ms / len(texts):.2f} ms/post, bs={args.bs})")
        if args.out:
            res = [{"id": r.get("id"), **p} for r, p in zip(rows, preds)]
            json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
            print(f"wrote {args.out}")
        else:
            for t, p in zip(texts[:10], preds[:10]):
                print_pred(t, p)
            if len(texts) > 10:
                print(f"  ... {len(texts) - 10} more (use --out to save all)")
        return

    # nothing given -> built-in samples
    for t, p in zip(SAMPLES, pred.predict(SAMPLES)):
        print_pred(t, p)


# ----------------------------------------------------------------------------
# Mode 2: bench
# ----------------------------------------------------------------------------
def run_bench(pred, args):
    if args.input:
        texts, authors, _ = load_posts(args.input)
    else:
        texts, authors = SAMPLES, [""] * len(SAMPLES)

    n_single, n_batch, batch_sizes, warmup = args.n_single, args.n_batch, args.batch_sizes, args.warmup

    def cyc(n, off=0):
        return [texts[(off + i) % len(texts)] for i in range(n)], \
               [authors[(off + i) % len(authors)] for i in range(n)]

    print(f"\ntorch threads: {torch.get_num_threads()} | device: cpu | max_len: {pred.max_len} "
          f"| corpus: {len(texts)} posts | quantize: {args.quantize}")

    # warmup (first calls are slow: allocator, oneDNN kernel selection)
    for i in range(warmup):
        t, a = cyc(1, i)
        pred.predict(t, a)
    t, a = cyc(max(batch_sizes))
    pred.predict(t, a, bs=max(batch_sizes))

    # ---- single request: tokenize + forward, one post at a time -------------
    tok_ms, fwd_ms, tot_ms = [], [], []
    for i in range(n_single):
        t, a = cyc(1, i)
        t_tok, enc = timed(lambda: pred.encode(t, a))
        t_fwd, _ = timed(lambda: pred.forward_logits(enc))
        tok_ms.append(t_tok)
        fwd_ms.append(t_fwd)
        tot_ms.append(t_tok + t_fwd)
    print(f"\nSINGLE REQUEST (n={n_single})")
    print(f"  total   mean {statistics.mean(tot_ms):7.2f} ms | p50 {pct(tot_ms, 50):7.2f} | "
          f"p95 {pct(tot_ms, 95):7.2f} | max {max(tot_ms):7.2f}")
    print(f"  tokenize mean {statistics.mean(tok_ms):6.2f} ms | forward mean {statistics.mean(fwd_ms):7.2f} ms")
    print(f"  throughput ~{1000 / statistics.mean(tot_ms):.1f} req/s")

    # ---- batches -------------------------------------------------------------
    print(f"\nBATCH (reps={n_batch} per size)")
    print(f"  {'bs':>4} | {'batch ms (mean)':>15} | {'p95 ms':>8} | {'ms/item':>8} | "
          f"{'items/s':>8} | {'tok ms':>7} | {'fwd ms':>8}")
    for bs in batch_sizes:
        tots, toks, fwds = [], [], []
        for r in range(n_batch):
            t, a = cyc(bs, r * bs)
            t_tok, enc = timed(lambda: pred.encode(t, a))
            t_fwd, _ = timed(lambda: pred.forward_logits(enc))
            toks.append(t_tok)
            fwds.append(t_fwd)
            tots.append(t_tok + t_fwd)
        m = statistics.mean(tots)
        print(f"  {bs:>4} | {m:>15.2f} | {pct(tots, 95):>8.2f} | {m / bs:>8.2f} | {1000 * bs / m:>8.1f} | "
              f"{statistics.mean(toks):>7.2f} | {statistics.mean(fwds):>8.2f}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    # options shared by both modes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", required=True, help="folder with best.pt, config.json, tokenizer/")
    common.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    common.add_argument("--quantize", action="store_true", help="int8 dynamic quantization of Linear layers")
    common.add_argument("--input", help="json list of {post_text, author?}")

    ap = argparse.ArgumentParser(
        description="good_job BERT: get predictions or benchmark CPU speed.")
    sub = ap.add_subparsers(dest="mode", required=True, metavar="{predict,bench}")

    p = sub.add_parser("predict", parents=[common], help="classify a post or a file of posts")
    p.add_argument("--text", help="single post text to classify")
    p.add_argument("--author", default="", help="author for --text (used if the model was trained with author)")
    p.add_argument("--out", help="write predictions json here (with --input)")
    p.add_argument("--bs", type=int, default=32, help="batch size for --input")
    p.set_defaults(func=run_predict)

    b = sub.add_parser("bench", parents=[common], help="measure single-request and batch latency")
    b.add_argument("--n-single", type=int, default=100, help="single-request timing runs")
    b.add_argument("--n-batch", type=int, default=10, help="repetitions per batch size")
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    b.add_argument("--warmup", type=int, default=5)
    b.set_defaults(func=run_bench)

    return ap


def main():
    args = build_parser().parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    pred = Predictor(args.run_dir, quantize=args.quantize)
    print(f"model loaded in {time.perf_counter() - t0:.2f}s (quantize={args.quantize})")

    args.func(pred, args)


if __name__ == "__main__":
    main()