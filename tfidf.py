# pip install pandas numpy scikit-learn scipy

import json
import re
import html
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------- CONFIG ----------------

DATA_PATH = "data_deduped.json"
OUTPUT_PATH = "similarity_graph_tfidf.json"
SKIPPED_PATH = "tfidf_skipped_pairs.json"   # pairs at/above threshold that were gated out

THRESHOLD = 0.9            # save edges at or above this; pick the real cutoff (e.g. 0.90) in the head script
MIN_CHARS = 20
CHUNK = 1000

REQUIRE_SAME_LABEL = True     # no edge between posts with different labels
REQUIRE_SAME_NUMBERS = True   # no edge between posts whose salaries/numbers differ


# ---------------- NORMALIZATION ----------------

URL_PATTERN = re.compile(
    r"(?i)(?:https?\s*:\s*/\s*/\S*|www\.\S+"
    r"|\b(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|lnkd\.in)/\S+"
    r"|\b(?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/\S*)?)"
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
TRACKING_PATTERN = re.compile(
    r"(?i)\b(?:utm_\w+|referralcode|ref|source|campaign)\b\s*=?\s*\S*"
)


def normalize_text(text):
    text = str(text or "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = UUID_PATTERN.sub(" ", text)
    text = TRACKING_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


LINK_PATTERN = re.compile(r"https?://\S+|\b[\w-]+(?:\.[\w-]+)+/\S*")


def numbers(raw):
    """Numbers in the raw post, ignoring links, hashtags and list numbering."""
    raw = LINK_PATTERN.sub(" ", str(raw or ""))
    raw = re.sub(r"[#@]\w+", " ", raw)
    raw = re.sub(r"(?m)^\s*\d+[.)]\s+", "", raw)
    out = set()
    for m in re.findall(r"\d[\d,\.]*\s*[kK]?", raw):
        m = m.strip().lower().replace(",", "")
        if m.endswith("k"):
            m = m[:-1] + "000"
        m = m.rstrip(".")
        if m:
            out.add(m)
    return frozenset(out)


# ---------------- LOAD POSTS ----------------

with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.json_normalize(json.load(f))

df = df[["id", "post_text", "author", "label", "poster", "url"]].copy()

df["text"] = df["post_text"].fillna("").apply(normalize_text)
df = df[df["text"].str.len() >= MIN_CHARS].copy().reset_index(drop=True)

print(f"Posts compared: {len(df):,}")
print(f"Unique authors: {df['author'].nunique():,}")

df = df.astype(object).where(pd.notna(df), None)

n = len(df)
ids = df["id"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
nums = [numbers(t) for t in df["post_text"]]


# ---------------- TF-IDF ----------------

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8, sublinear_tf=True)
X = normalize(vec.fit_transform(df["text"].tolist())).tocsr()
XT = X.T.tocsc()
print(f"TF-IDF matrix: {X.shape[0]:,} x {X.shape[1]:,}")


# ---------------- BUILD EDGES (chunked, upper triangle) ----------------

edges = []
skipped = []
skipped_label = 0
skipped_numbers = 0

for start in range(0, n, CHUNK):
    block = (X[start:start + CHUNK] @ XT).tocoo()
    rows = block.row + start
    mask = (block.col > rows) & (block.data >= THRESHOLD)

    for i, j, v in zip(rows[mask], block.col[mask], block.data[mask]):
        i, j, v = int(i), int(j), float(v)

        edges.append({
            "source": ids[i],
            "target": ids[j],
            "similarity": round(v, 4),
        })
        continue

    print(f"Scored {min(start + CHUNK, n):,}/{n:,} | edges: {len(edges):,}")


# ---------------- NODES ----------------

nodes = [
    {
        "id": ids[i],
        "text": df.at[i, "text"],
        "author": df.at[i, "author"],
        "label": df.at[i, "label"],
        "poster": df.at[i, "poster"],
        "url": df.at[i, "url"],
    }
    for i in range(n)
]


# ---------------- SAVE ----------------

graph = {
    "nodes": nodes,
    "edges": edges,
    "metadata": {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "threshold": THRESHOLD,
        "method": "tfidf",
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 0.8,
        "require_same_label": REQUIRE_SAME_LABEL,
        "require_same_numbers": REQUIRE_SAME_NUMBERS,
        "skipped_label_conflict": skipped_label,
        "skipped_numbers_differ": skipped_numbers,
    },
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(graph, f, ensure_ascii=False, indent=2, default=str)

with open(SKIPPED_PATH, "w", encoding="utf-8") as f:
    json.dump(skipped, f, ensure_ascii=False, indent=1, default=str)

print("\nGraph saved.")
print(f"Nodes: {len(nodes):,}")
print(f"Edges: {len(edges):,}")
print(f"Skipped, label conflict: {skipped_label:,}")
print(f"Skipped, numbers differ: {skipped_numbers:,}")
print(f"Output: {OUTPUT_PATH} | skipped pairs: {SKIPPED_PATH}")