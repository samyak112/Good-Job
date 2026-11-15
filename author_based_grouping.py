
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

DATA_PATH = "data.json"
BATCH_SIZE = 128
THRESHOLD = 0.85
MIN_CHARS = 20

EMBEDDINGS_CACHE = "embeddings.npy"
CACHE_META = "embeddings.meta"


def normalize_text(text):
    text = str(text or "")

    text = unicodedata.normalize("NFKC", text)

    # 1. Standardize URLs and Emails into explicit placeholder tokens
    text = re.sub(
        r"""(?i)\b(?:(?:https?://|www\.)\S+|(?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/\S*)?)""",
        " <URL> ",
        text
    )
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " <EMAIL> ", text)

    # 3. PRESERVE structural delimiters (colons, pipes, hyphens, newlines) 
    # Instead of deleting them, normalize whitespace around them so they act as clean tokens.
    text = re.sub(r"([:\|\-\/\n])", r" \1 ", text)

    # 4. Strip out emojis and truly useless symbols, but KEEP structural punctuation
    # Notice we keep #, :, |, -, /, and \n in the allowed character set
    text = re.sub(r"[^\w\s\#:\|\-\/\n]", " ", text, flags=re.UNICODE)

    # 5. Normalize whitespace (DO NOT lowercase yet if you want to use case for NER/slots)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load posts
with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.json_normalize(json.load(f))

df = df[["id", "post_text", "author", "label", "poster", "url"]].copy()

# Normalize text
df["text"] = df["post_text"].fillna("").apply(normalize_text)

# Remove short posts
df = df[df["text"].str.len() >= MIN_CHARS].copy()

# Sort by author
df = df.sort_values(
    by=["author", "id"],
    kind="stable"
).reset_index(drop=True)

# Exclude authors with only one post
author_counts = df["author"].value_counts()
df = df[df["author"].map(author_counts) > 1].reset_index(drop=True)

print(f"Total posts: {len(df)}")
print(f"Unique authors: {df['author'].nunique()}")


# Generate or load embeddings
fingerprint = hashlib.sha256(
    "\n".join(
        df["id"].astype(str) + ":" +
        df["author"].fillna("").astype(str) + ":" +
        df["text"]
    ).encode("utf-8")
).hexdigest()

cache_valid = False

if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CACHE_META):
    with open(CACHE_META, encoding="utf-8") as f:
        cache_valid = json.load(f).get("fingerprint") == fingerprint

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if cache_valid:
    print("Loading cached embeddings...")
    embeddings = np.load(EMBEDDINGS_CACHE)

    if len(embeddings) != len(df):
        cache_valid = False

if not cache_valid:
    print("Generating embeddings...")

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



# First pass: N x N similarity within each author
results = []

author_groups = df.groupby("author", sort=False, dropna=False)

for author, group in author_groups:

    indices = group.index.to_numpy()
    author_embeddings = embeddings[indices]

    index = faiss.IndexFlatIP(author_embeddings.shape[1])
    index.add(author_embeddings)

    similarities, neighbors = index.search(
        author_embeddings,
        len(group)
    )

    matched_ids = set()

    for i in range(len(group)):

        query_row = group.iloc[i]
        query_id = query_row["id"]
        query_text = query_row["text"]

        # Skip posts already assigned to another post's group
        if query_id in matched_ids:
            continue

        matches = []

        for similarity, j in zip(similarities[i], neighbors[i]):

            candidate = group.iloc[j]
            candidate_id = candidate["id"]

            if j == i or similarity < THRESHOLD:
                continue

            if candidate_id in matched_ids:
                continue

            if candidate["text"] == query_text:
                continue

            matches.append({
                "id": candidate_id,
                "text": candidate["text"],
                "author": candidate["author"],
                "label": candidate["label"],
                "similarity": round(float(similarity), 4),
                "poster": candidate["poster"],
                "url":candidate['url']
            })

        if matches:

            matches.sort(
                key=lambda x: x["similarity"],
                reverse=True
            )

            results.append({
                "id": query_id,
                "text": query_text,
                "author": author,
                "length": len(matches),
                "poster": query_row["poster"],
                "label": query_row["label"],
                "url":query_row["url"],
                "matches": matches,
            })

            # Mark all matched posts as assigned
            matched_ids.update(match["id"] for match in matches)

    print(
        f"Author: {author} | "
        f"Posts: {len(group)} | "
        f"Matched groups: {sum(1 for r in results if r['author'] == author)}"
    )



# Save results
with open("similar_posts_author_pass.json", "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )

print(f"\nSaved {len(results)} groups to similar_posts_author_pass.json")
