
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

DATA_PATH = "data_deduped.json"
OUTPUT_PATH = "similarity_graph_new.json"

BATCH_SIZE = 128
THRESHOLD = 0.85
MIN_CHARS = 20

EMBEDDINGS_CACHE = "embeddings.npy"
CACHE_META = "embeddings.meta"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- NORMALIZATION ----------------

def normalize_text(text):
    text = str(text or "")

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Preserve the meaning of ampersands
    text = text.replace("&", " and ")

    # Standardize URLs
    URL_PATTERN = re.compile(
    r"""(?i)\b(?:
        (?:https?://|www\.)[^\s]+
        |
        (?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|lnkd\.in)/[^\s]+
        |
        (?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/[^\s]*)?
    )""",
    re.VERBOSE
)

    text = URL_PATTERN.sub(" URL ", text)

    # Standardize emails
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " EMAIL ",
        text
    )

    # Preserve structural punctuation
    text = re.sub(
        r"([:|/\\\-\n])",
        r" \1 ",
        text
    )

    # Remove other symbols while preserving hashtags
    text = re.sub(
        r"[^\w\s#:|/\\-]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------- LOAD POSTS ----------------

with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.json_normalize(json.load(f))

df = df[
    ["id", "post_text", "author", "label", "poster", "url"]
].copy()

df["text"] = df["post_text"].fillna("").apply(normalize_text)

# Remove short posts
df = df[df["text"].str.len() >= MIN_CHARS].copy().reset_index(drop=True)

print(f"Unique posts remaining: {len(df):,}")
print(f"Unique authors: {df['author'].nunique():,}")

# Convert NaN values to None for valid JSON
df = df.astype(object).where(pd.notna(df), None)

# ---------------- GENERATE OR LOAD EMBEDDINGS ----------------

fingerprint = hashlib.sha256(
    "\n".join(
        df["id"].astype(str) + ":" + df["text"].astype(str)
    ).encode("utf-8")
).hexdigest()

cache_valid = False
embeddings = None

if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CACHE_META):

    with open(CACHE_META, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("fingerprint") == fingerprint:

        cached = np.load(EMBEDDINGS_CACHE)

        if len(cached) == len(df):
            embeddings = cached
            cache_valid = True


if cache_valid:

    print("Loading cached embeddings...")

else:

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

    print("Embeddings cached.")


# ---------------- BUILD GRAPH PROGRESSIVELY ----------------

nodes = []
edges = []

# Global FAISS index. No author partitioning.
index = faiss.IndexFlatIP(embeddings.shape[1])

for i, row in df.iterrows():

    post_id = str(row["id"])

    # Add node with metadata
    nodes.append({
        "id": post_id,
        "text": row["text"],
        "author": row["author"],
        "label": row["label"],
        "poster": row["poster"],
        "url": row["url"],
    })

    # Compare against every previously processed post
    if index.ntotal > 0:

        similarities, neighbors = index.search(
            embeddings[i:i + 1],
            index.ntotal
        )

        for similarity, j in zip(similarities[0], neighbors[0]):

            similarity = float(similarity)

            if similarity < THRESHOLD:
                continue

            candidate_id = str(df.iloc[j]["id"])

            edges.append({
                "source": candidate_id,
                "target": post_id,
                "similarity": round(similarity, 4),
            })

    # Add current embedding only after comparing it
    index.add(embeddings[i:i + 1])

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1:,}/{len(df):,} posts | "
            f"Nodes: {len(nodes):,} | "
            f"Edges: {len(edges):,}"
        )


# ---------------- SAVE GRAPH ----------------

graph = {
    "nodes": nodes,
    "edges": edges,
    "metadata": {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
    }
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        graph,
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )

print("\nGraph saved.")
print(f"Nodes: {len(nodes):,}")
print(f"Edges: {len(edges):,}")
print(f"Output: {OUTPUT_PATH}")
