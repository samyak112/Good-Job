#!/usr/bin/env python3
"""
Author-grouped, label-stratified 80/10/10 split + WeightedRandomSampler for training.

Usage:
    python split_and_sample.py --input posts.json --out-dir splits

Input: JSON array (or JSONL) of post objects with at least `author` and `label`.
Authors are grouped as (source, author), so the same handle on Twitter and LinkedIn is kept separate.

Outputs (in --out-dir):
    train.json, val.json, test.json   (train.json has an extra `sample_weight` field)
    split_report.txt, split_authors.json

In your training code:
    from split_and_sample import make_train_sampler
    sampler = make_train_sampler(train_df)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler)   # do NOT pass shuffle=True
    val/test loaders: shuffle=False, no sampler (natural distribution)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["ML Job", "Job", "Other", "Not Job"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            return pd.DataFrame([json.loads(line) for line in f if line.strip()])
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    raise ValueError(f"Unsupported file type: {p.suffix}")


# ----------------------------------------------------------------------------
# Step 1: author-based, label-stratified split
# ----------------------------------------------------------------------------
def label_dist(s: pd.Series, labels) -> np.ndarray:
    return s.value_counts(normalize=True).reindex(labels).fillna(0).values


def one_split(df, author_col, label_col, seed, n_splits=10):
    """10 folds of ~10% each: fold 0 = val, fold 1 = test, the rest = train."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df[label_col], groups=df[author_col])):
        fold[idx] = k
    return df[fold >= 2], df[fold == 0], df[fold == 1]


def split_score(df, parts, label_col, labels, target):
    """Lower is better: label-proportion error + size error, over val and test."""
    total = len(df)
    score = 0.0
    for part in parts[1:]:
        score += np.abs(label_dist(part[label_col], labels) - target).max()
        score += abs(len(part) / total - 0.10)
    return score


def author_split(df, author_col, label_col, n_seeds=300, base_seed=0):
    labels = [l for l in LABELS if l in set(df[label_col])] or sorted(df[label_col].unique())
    target = label_dist(df[label_col], labels)
    best = None
    for s in range(base_seed, base_seed + n_seeds):
        parts = one_split(df, author_col, label_col, s)
        sc = split_score(df, parts, label_col, labels, target)
        if best is None or sc < best[0]:
            best = (sc, s, parts)
    _, seed, (train, val, test) = best

    # Hard guarantee: no author in more than one subset
    a_tr, a_va, a_te = (set(d[author_col]) for d in (train, val, test))
    assert not (a_tr & a_va or a_tr & a_te or a_va & a_te), "Author leakage!"
    assert len(train) + len(val) + len(test) == len(df)
    return train.copy(), val.copy(), test.copy(), seed, labels


def build_report(df, train, val, test, seed, labels, label_col, author_col):
    lines = [f"Best seed: {seed}", f"Total posts: {len(df)}, authors: {df[author_col].nunique()}", ""]
    header = f"{'subset':<8}{'posts':>7}{'share':>8}{'authors':>9}  " + "  ".join(f"{l:>8}" for l in labels)
    lines.append(header)
    for name, d in (("all", df), ("train", train), ("val", val), ("test", test)):
        p = label_dist(d[label_col], labels)
        lines.append(
            f"{name:<8}{len(d):>7}{len(d)/len(df):>8.2%}{d[author_col].nunique():>9}  "
            + "  ".join(f"{x:>8.2%}" for x in p)
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Step 2: class balancing for training only (no rows deleted)
# ----------------------------------------------------------------------------
def compute_sample_weights(train, label_col, author_col=None, class_power=1.0, author_power=0.0):
    """
    class_power:  1.0 = fully inverse-frequency (classes ~equal per epoch),
                  0.5 = softer (inverse sqrt), 0 = no class balancing.
    author_power: 0 = ignore authors; 0.5 = also downweight prolific authors by 1/sqrt(n_posts).
    """
    counts = train[label_col].value_counts()
    w = train[label_col].map(counts) ** (-class_power)
    if author_col is not None and author_power > 0:
        n_author = train[author_col].map(train[author_col].value_counts())
        w = w * n_author ** (-author_power)
    return (w / w.mean()).values  # normalized to mean 1


def make_train_sampler(train, label_col="label", author_col=None, class_power=1.0, author_power=0.0):
    import torch
    from torch.utils.data import WeightedRandomSampler

    w = compute_sample_weights(train, label_col, author_col, class_power, author_power)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(train),  # keep epoch length the same
        replacement=True,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--author-col", default="author")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--seeds", type=int, default=300, help="number of random seeds to search")
    ap.add_argument("--class-power", type=float, default=1.0)
    ap.add_argument("--author-power", type=float, default=0.0)
    args = ap.parse_args()

    df = load_df(args.input).reset_index(drop=True)
    for c in (args.author_col, args.label_col):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not found. Columns: {list(df.columns)}")
    n0 = len(df)
    df = df[df[args.label_col].astype(str).str.strip().ne("") & df[args.label_col].notna()]
    df = df[df[args.author_col].notna()].reset_index(drop=True)
    if len(df) < n0:
        print(f"Dropped {n0 - len(df)} rows with missing label/author")

    # Group key: (source, author) so identical handles on different platforms stay separate
    group_col = "_group"
    df[group_col] = (df["source"].astype(str) + "|" if "source" in df.columns else "") + df[args.author_col].astype(str)

    train, val, test, seed, labels = author_split(df, group_col, args.label_col, args.seeds)
    train["sample_weight"] = compute_sample_weights(
        train, args.label_col, group_col, args.class_power, args.author_power
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.drop(columns=[group_col]).to_json(out / f"{name}.json", orient="records", indent=2, force_ascii=False)
    json.dump(
        {n: sorted(d[group_col].unique()) for n, d in (("train", train), ("val", val), ("test", test))},
        open(out / "split_authors.json", "w"), indent=2, ensure_ascii=False,
    )

    report = build_report(df, train, val, test, seed, labels, args.label_col, group_col)
    (out / "split_report.txt").write_text(report)
    print(report)
    print(f"\nSaved to {out}/. Leakage check passed.")


if __name__ == "__main__":
    main()