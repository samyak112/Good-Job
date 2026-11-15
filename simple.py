# pip install pandas numpy sentence-transformers faiss-cpu

import json
import os
import re
import html
import unicodedata
import hashlib

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------

DATA_PATH = "data_original.json"
PAIRS_OUTPUT = "near_duplicate_pairs.csv"

THRESHOLD = 0.9          # pairs with similarity > this are reported
MIN_CHARS = 20
BATCH_SIZE = 128
CHUNK = 1000             # queries per range_search call

EMBEDDINGS_CACHE = "embeddings.npy"
CACHE_META = "embeddings.meta"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- NORMALIZATION ----------------

URL_PATTERN = re.compile(
    r"""(?i)\b(?:
        (?:https?://|www\.)[^\s]+
        |
        (?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|lnkd\.in)/[^\s]+
        |
        (?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/[^\s]*)?
    )""",
    re.VERBOSE,
)


def normalize_text(text):
    text = str(text or "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = URL_PATTERN.sub(" URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " EMAIL ", text)
    text = re.sub(r"([:|/\\\-\n])", r" \1 ", text)
    text = re.sub(r"[^\w\s#:|/\\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- LOAD + CLEAN ----------------

with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.json_normalize(json.load(f))

df = df[["id", "post_text", "author", "label", "poster", "url"]].copy()
df["text"] = df["post_text"].fillna("").apply(normalize_text)
df = df[df["text"].str.len() >= MIN_CHARS].copy()

# exact duplicates (same normalized text)
before = len(df)
df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
print(f"Exact duplicates skipped: {before - len(df):,}")
print(f"Unique posts: {len(df):,}")


# ---------------- EMBEDDINGS (cached) ----------------

fingerprint = hashlib.sha256(
    "\n".join(df["id"].astype(str) + ":" + df["text"]).encode("utf-8")
).hexdigest()

embeddings = None
if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CACHE_META):
    with open(CACHE_META, encoding="utf-8") as f:
        if json.load(f).get("fingerprint") == fingerprint:
            cached = np.load(EMBEDDINGS_CACHE)
            if len(cached) == len(df):
                embeddings = cached
                print("Loaded cached embeddings.")

if embeddings is None:
    print("Generating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(EMBEDDINGS_CACHE, embeddings)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint}, f)


# ---------------- FIND ALL PAIRS ABOVE THRESHOLD ----------------

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

pairs = []  # (i, j, similarity) with i < j

for start in range(0, len(df), CHUNK):
    lims, D, I = index.range_search(embeddings[start:start + CHUNK], THRESHOLD)
    for q in range(len(lims) - 1):
        i = start + q
        for k in range(lims[q], lims[q + 1]):
            j = int(I[k])
            if i < j:  # skips self-matches and the mirrored (j, i) copy
                pairs.append((i, j, float(D[k])))

print(f"\nPairs with similarity > {THRESHOLD}: {len(pairs):,}")

involved = {p[0] for p in pairs} | {p[1] for p in pairs}
print(f"Posts involved in at least one pair: {len(involved):,} "
      f"({len(involved) / len(df):.1%} of {len(df):,})")


# ---------------- SAVE FOR INSPECTION ----------------

out = pd.DataFrame(
    {
        "id_a": [df.at[i, "id"] for i, _, _ in pairs],
        "id_b": [df.at[j, "id"] for _, j, _ in pairs],
        "similarity": [round(s, 4) for _, _, s in pairs],
        "label_a": [df.at[i, "label"] for i, _, _ in pairs],
        "label_b": [df.at[j, "label"] for _, j, _ in pairs],
        "text_a": [df.at[i, "text"] for i, _, _ in pairs],
        "text_b": [df.at[j, "text"] for _, j, _ in pairs],
    }
).sort_values("similarity", ascending=False)

out.to_csv(PAIRS_OUTPUT, index=False)
print(f"Saved pairs to {PAIRS_OUTPUT}")

if len(out):
    diff = (out["label_a"] != out["label_b"]).sum()
    print(f"Pairs where the two labels differ: {diff:,}")