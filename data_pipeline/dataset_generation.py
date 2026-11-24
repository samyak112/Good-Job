#!/usr/bin/env python3
"""
Author-grouped, label-stratified 80/10/10 split (deterministic) + WeightedRandomSampler.

Edit the CONFIG block below, then run:  python split_and_sample.py

Outputs (in OUT_DIR):
    train.json, val.json, test.json   (train.json has an extra `sample_weight` field)
    split_report.txt, split_authors.json

In your training code:
    from split_and_sample import make_train_sampler
    sampler = make_train_sampler(train_df)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler)   # do NOT pass shuffle=True
    val/test loaders: shuffle=False, no sampler (natural distribution)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
INPUT = "data_pipeline/dataset/training_data_labelled.json"          # .json / .jsonl / .csv / .parquet
OUT_DIR = "splits"
AUTHOR_COL = "author"
LABEL_COL = "label"
SOURCE_COL = "source"         # authors are grouped as (source, author); ignored if column is absent
RATIOS = (0.8, 0.1, 0.1)      # train / val / test
CLASS_POWER = 1.0             # 1.0 = full inverse-frequency, 0.5 = softer, 0 = off
AUTHOR_POWER = 0.0            # 0 = ignore authors, 0.5 = downweight prolific authors by 1/sqrt(n)
MAX_PASSES = 50               # max refinement passes in the split

LABELS = ["ML JOB", "JOB", "OTHER", "NOT JOB"]

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
# Step 1: deterministic author-grouped, label-stratified split
# ----------------------------------------------------------------------------
def label_dist(s: pd.Series, labels) -> np.ndarray:
    return s.value_counts(normalize=True).reindex(labels).fillna(0).values


def author_split(df, author_col, label_col, ratios=RATIOS, max_passes=MAX_PASSES):
    """
    Greedy assignment of whole authors to train/val/test, then single-author
    move refinement. Cost = squared fraction of each class (and of total posts)
    that is misallocated vs. the target ratios, so rare classes count as much
    as common ones. No randomness: same input -> same split.
    """
    present = set(df[label_col])
    labels = [l for l in LABELS if l in present] + sorted(present - set(LABELS))
    L = len(labels)

    codes, authors = pd.factorize(df[author_col], sort=True)  # independent of row order
    lab = pd.Categorical(df[label_col], categories=labels).codes
    A = len(authors)

    # per-author class counts; last column = total posts
    C = np.zeros((A, L + 1))
    np.add.at(C, (codes, lab), 1)
    C[:, L] = C[:, :L].sum(1)
    total = C.sum(0)
    target = np.asarray(ratios)[:, None] * total  # (3, L+1)

    def cost(S):
        return (((S - target) / total) ** 2).sum()

    S = np.zeros_like(target)
    assign = np.empty(A, dtype=int)

    # greedy: biggest authors first (stable sort -> ties resolved alphabetically)
    for a in np.argsort(-C[:, L], kind="stable"):
        costs = []
        for s in range(3):
            S[s] += C[a]
            costs.append(cost(S))
            S[s] -= C[a]
        assign[a] = int(np.argmin(costs))
        S[assign[a]] += C[a]

    # refine: move single authors while it lowers the cost
    for _ in range(max_passes):
        moved = False
        for a in range(A):
            cur = assign[a]
            best_s, best_c = cur, cost(S)
            S[cur] -= C[a]
            for s in range(3):
                if s == cur:
                    continue
                S[s] += C[a]
                c = cost(S)
                S[s] -= C[a]
                if c < best_c - 1e-12:
                    best_s, best_c = s, c
            S[best_s] += C[a]
            if best_s != cur:
                assign[a] = best_s
                moved = True
        if not moved:
            break

    split_of_row = assign[codes]
    train, val, test = (df[split_of_row == k].copy() for k in range(3))

    # Hard guarantee: no author in more than one subset
    a_tr, a_va, a_te = (set(d[author_col]) for d in (train, val, test))
    assert not (a_tr & a_va or a_tr & a_te or a_va & a_te), "Author leakage!"
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test, labels


def build_report(df, train, val, test, labels, label_col, author_col):
    lines = [f"Total posts: {len(df)}, authors: {df[author_col].nunique()}", ""]
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


def make_train_sampler(train, label_col=LABEL_COL, author_col=None,
                       class_power=CLASS_POWER, author_power=AUTHOR_POWER):
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
    df = load_df(INPUT).reset_index(drop=True)
    for c in (AUTHOR_COL, LABEL_COL):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not found. Columns: {list(df.columns)}")

    # Drop rows with missing label/author
    n0 = len(df)
    df = df[df[LABEL_COL].notna() & df[LABEL_COL].astype(str).str.strip().ne("")]
    df = df[df[AUTHOR_COL].notna()].reset_index(drop=True)
    if len(df) < n0:
        print(f"Dropped {n0 - len(df)} rows with missing label/author")

    # Case/whitespace-insensitive labels: "ml job", " Ml  Job " -> "ML JOB"
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.split().str.join(" ").str.upper()

    # Group key: (source, author) so identical handles on different platforms stay separate
    group_col = "_group"
    prefix = df[SOURCE_COL].astype(str) + "|" if SOURCE_COL in df.columns else ""
    df[group_col] = prefix + df[AUTHOR_COL].astype(str)

    # Split + training weights
    train, val, test, labels = author_split(df, group_col, LABEL_COL)
    train["sample_weight"] = compute_sample_weights(
        train, LABEL_COL, group_col, CLASS_POWER, AUTHOR_POWER
    )

    # Save
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.drop(columns=[group_col]).to_json(
            out / f"{name}.json", orient="records", indent=2, force_ascii=False
        )

    with open(out / "split_authors.json", "w", encoding="utf-8") as f:
        json.dump(
            {n: sorted(d[group_col].unique()) for n, d in (("train", train), ("val", val), ("test", test))},
            f, indent=2, ensure_ascii=False,
        )

    report = build_report(df, train, val, test, labels, LABEL_COL, group_col)
    (out / "split_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}/. Leakage check passed.")


if __name__ == "__main__":
    main()