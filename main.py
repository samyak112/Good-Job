
# pip install pandas numpy sentence-transformers faiss-cpu

import json
import os
import re
import hashlib
import html
import unicodedata

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "data.json"
BATCH_SIZE = 128
THRESHOLD = 0.85
MIN_CHARS = 20

EMBEDDINGS_CACHE = "embeddings.npy"
FAISS_CACHE = "posts.index"
CACHE_META = "embeddings.meta"


def normalize_text(text):
    text = str(text or "")

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(
    r"""(?i)\b(?:(?:https?://|www\.)\S+|(?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/\S*)?)""",
    " ",
    text
)

    # Remove email addresses
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove emojis and unwanted symbols, preserve # and words
    text = re.sub(
        r"[^\w\s#]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


# Load posts
with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.json_normalize(json.load(f))

df = df[["id", "post_text", "author", "label", "poster"]].copy()

# Normalize text before similarity
df["text"] = df["post_text"].fillna("").apply(normalize_text)

# Remove short and empty posts
df = df[df["text"].str.len() >= MIN_CHARS].reset_index(drop=True)

# Cache fingerprint: invalidate if dataset or normalization changes
fingerprint = hashlib.sha256(
    "\n".join(df["id"].astype(str) + ":" + df["text"]).encode("utf-8")
).hexdigest()

cache_valid = False

if all(os.path.exists(p) for p in [
    EMBEDDINGS_CACHE,
    FAISS_CACHE,
    CACHE_META
]):
    with open(CACHE_META, encoding="utf-8") as f:
        cache_valid = json.load(f).get("fingerprint") == fingerprint

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load or generate embeddings
if cache_valid:
    print("Loading cached embeddings and FAISS index...")

    embeddings = np.load(EMBEDDINGS_CACHE)
    index = faiss.read_index(FAISS_CACHE)

    if len(embeddings) != len(df) or index.ntotal != len(df):
        cache_valid = False

if not cache_valid:
    print("Generating embeddings...")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save cache
    np.save(EMBEDDINGS_CACHE, embeddings)
    faiss.write_index(index, FAISS_CACHE)

    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint}, f)

    print("Cache saved.")


# Compare each post against all other posts
results = []
matched_ids = set()

for i in range(len(df)):
    query = embeddings[i:i+1]

    query_id = df.iloc[i]["id"]
    query_text = df.iloc[i]["text"]
    parent_author = df.iloc[i]["author"]

    # Skip posts already included in another cluster
    if query_id in matched_ids:
        continue

    similarities, neighbors = index.search(query, len(df))

    matches = []

    for similarity, j in zip(similarities[0], neighbors[0]):

        if j == i or similarity < THRESHOLD:
            continue

        candidate_id = df.iloc[j]["id"]
        candidate_text = df.iloc[j]["text"]

        # Skip identical normalized text
        if candidate_text == query_text:
            continue

        if candidate_id in matched_ids or candidate_id == query_id:
            continue

        matches.append({
            "id": candidate_id,
            "text": candidate_text,
            "author": df.iloc[j]["author"],
            "label": df.iloc[j]["label"],
            "similarity": round(float(similarity), 4),
            "poster": df.iloc[j]["poster"],
        })

    if matches:
        # Sort by embedding similarity
        matches.sort(
            key=lambda x: x["similarity"],
            reverse=True
        )

        results.append({
            "id": query_id,
            "text": query_text,
            "author": parent_author,
            "length": len(matches),
            "poster": df.iloc[i]["poster"],
            "label": df.iloc[i]["label"],
            "matches": matches,
        })

        matched_ids.update(match["id"] for match in matches)

    print(f"Processed {i + 1}/{len(df)} | Matches: {len(matches)}")


# Save results
with open("similar_posts.json", "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )

print("\nSaved to similar_posts.json")
