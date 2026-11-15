# pip install pandas numpy scikit-learn scipy sentence-transformers

import json
import os
import re
import html
import random
import hashlib
import unicodedata
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------- CONFIG ----------------

DATA_PATH = "data_original.json"

EMB_THRESHOLD = 0.90        # embedding cosine cutoff
TFIDF_THRESHOLD = 0.90      # TF-IDF cosine cutoff

MIN_CHARS = 20
CHUNK = 1000
BATCH_SIZE = 128
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS_CACHE = "embeddings.npy"
CACHE_META = "embeddings.meta"

REPORT_OUT = "compare_report.json"

# What to print
SAMPLES = {"both": 5, "emb_only": 15, "tfidf_only": 15}
SAMPLE_MODE = "lowest"      # "lowest" = pairs where the other method disagrees most, "random" = random pairs
TEXT_CHARS = 250            # characters of each text to print (full text is saved in the JSON)
RANDOM_SEED = 0


# ---------------- NORMALIZERS ----------------

# Embedding normalizer (same as your graph script, so the embedding cache stays valid)
EMB_URL_PATTERN = re.compile(
    r"""(?i)\b(?:
        (?:https?://|www\.)[^\s]+
        |
        (?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|lnkd\.in)/[^\s]+
        |
        (?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/[^\s]*)?
    )""",
    re.VERBOSE
)


def normalize_emb(text):
    text = str(text or "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = EMB_URL_PATTERN.sub(" URL ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " EMAIL ", text)
    text = re.sub(r"([:|/\\\-\n])", r" \1 ", text)
    text = re.sub(r"[^\w\s#:|/\\-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# TF-IDF normalizer (same as your lexical audit script)
LEX_URL_PATTERN = re.compile(
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


def normalize_lex(text):
    text = str(text or "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = LEX_URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = UUID_PATTERN.sub(" ", text)
    text = TRACKING_PATTERN.sub(" ", text)
    text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------- LOAD (one shared universe of posts) ----------------

with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)

ids, labels, raw_texts, emb_texts, lex_texts = [], [], [], [], []
for r in raw:
    t = normalize_emb(r.get("post_text"))
    if len(t) < MIN_CHARS:
        continue
    ids.append(str(r.get("id")))
    labels.append(str(r.get("label")))
    raw_texts.append(str(r.get("post_text") or ""))
    emb_texts.append(t)
    lex_texts.append(normalize_lex(r.get("post_text")))

n = len(ids)
print(f"Records: {len(raw):,} | posts compared: {n:,}")
print(f"Thresholds -> embeddings >= {EMB_THRESHOLD} | TF-IDF >= {TFIDF_THRESHOLD}")


# ---------------- EMBEDDINGS (cached) ----------------

fingerprint = hashlib.sha256(
    "\n".join(f"{i}:{t}" for i, t in zip(ids, emb_texts)).encode("utf-8")
).hexdigest()

E = None
if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CACHE_META):
    with open(CACHE_META, encoding="utf-8") as f:
        if json.load(f).get("fingerprint") == fingerprint:
            cached = np.load(EMBEDDINGS_CACHE)
            if len(cached) == n:
                E = cached
                print("Loaded cached embeddings")

if E is None:
    print("Generating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    E = model.encode(
        emb_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(EMBEDDINGS_CACHE, E)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint}, f)


# ---------------- TF-IDF ----------------

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8, sublinear_tf=True)
X = normalize(vec.fit_transform(lex_texts)).tocsr()
XT = X.T.tocsc()
print(f"TF-IDF matrix: {X.shape[0]:,} x {X.shape[1]:,}")


# ---------------- PAIRS ABOVE EACH METHOD'S OWN THRESHOLD ----------------

emb_pairs = set()
for s in range(0, n, CHUNK):
    sims = E[s:s + CHUNK] @ E.T
    rl, c = np.nonzero(sims >= EMB_THRESHOLD)
    r = rl + s
    m = c > r                                   # upper triangle only
    emb_pairs.update(zip(r[m].tolist(), c[m].tolist()))
print(f"Embedding pairs >= {EMB_THRESHOLD}: {len(emb_pairs):,}")

tfidf_pairs = set()
for s in range(0, n, CHUNK):
    block = (X[s:s + CHUNK] @ XT).tocoo()
    rows = block.row + s
    mask = (block.col > rows) & (block.data >= TFIDF_THRESHOLD)
    tfidf_pairs.update(zip(rows[mask].tolist(), block.col[mask].tolist()))
print(f"TF-IDF pairs >= {TFIDF_THRESHOLD}: {len(tfidf_pairs):,}")


# ---------------- BUCKETS (both scores computed for every pair) ----------------

def cross_scores(pair_list):
    if not pair_list:
        return np.array([]), np.array([])
    I = np.array([p[0] for p in pair_list])
    J = np.array([p[1] for p in pair_list])
    emb = np.einsum("ij,ij->i", E[I], E[J])
    lex = np.asarray(X[I].multiply(X[J]).sum(axis=1)).ravel()
    return emb, lex


def build_bucket(name, pairs):
    pairs = sorted(pairs)
    emb, lex = cross_scores(pairs)
    return {
        "name": name,
        "pairs": pairs,
        "emb": emb,
        "lex": lex,
        "posts": len({k for p in pairs for k in p}),
        "conflicts": sum(labels[i] != labels[j] for i, j in pairs),
    }


buckets = {
    "both": build_bucket("found by both", emb_pairs & tfidf_pairs),
    "emb_only": build_bucket("embeddings only", emb_pairs - tfidf_pairs),
    "tfidf_only": build_bucket("TF-IDF only", tfidf_pairs - emb_pairs),
}

print("\n" + "=" * 70)
print(f"PAIR COUNTS  (emb >= {EMB_THRESHOLD}, tfidf >= {TFIDF_THRESHOLD})")
print("=" * 70)
print(f"{'bucket':<18}{'pairs':>10}{'posts':>10}{'label-conflict':>18}")
for b in buckets.values():
    print(f"{b['name']:<18}{len(b['pairs']):>10,}{b['posts']:>10,}{b['conflicts']:>18,}")
print(f"{'union':<18}{len(emb_pairs | tfidf_pairs):>10,}")

for key, other, col in [("emb_only", "TF-IDF", "lex"), ("tfidf_only", "embedding", "emb")]:
    b = buckets[key]
    if len(b["pairs"]):
        s = b[col]
        print(f"\n{b['name']}: {other} score of these pairs -> "
              f"min {s.min():.3f} | median {np.median(s):.3f} | max {s.max():.3f}")
        hist = Counter(int(v * 10) / 10 for v in s)
        for k in sorted(hist):
            print(f"   {k:.1f}-{k + 0.1:.1f}: {hist[k]:,}")


# ---------------- SAMPLES WITH IDS AND COMPARED TEXT ----------------

def clip(s):
    return s if len(s) <= TEXT_CHARS else s[:TEXT_CHARS] + "..."


def print_pair(b, k):
    i, j = b["pairs"][k]
    print(f"\n[{ids[i]}] vs [{ids[j]}]   "
          f"emb={b['emb'][k]:.3f}  tfidf={b['lex'][k]:.3f}  "
          f"labels={labels[i]} / {labels[j]}"
          f"{'   <-- LABEL CONFLICT' if labels[i] != labels[j] else ''}")
    print(f"  EMB text   A: {clip(emb_texts[i])}")
    print(f"  EMB text   B: {clip(emb_texts[j])}")
    print(f"  TFIDF text A: {clip(lex_texts[i])}")
    print(f"  TFIDF text B: {clip(lex_texts[j])}")


random.seed(RANDOM_SEED)
for key, title, other_col in [
    ("both", "FOUND BY BOTH", "emb"),
    ("emb_only", "EMBEDDINGS ONLY (TF-IDF below threshold)", "lex"),
    ("tfidf_only", "TF-IDF ONLY (embeddings below threshold)", "emb"),
]:
    b = buckets[key]
    count = min(SAMPLES[key], len(b["pairs"]))
    if count == 0:
        continue
    if SAMPLE_MODE == "lowest":
        order = np.argsort(b[other_col])[:count]
        how = "lowest agreement first"
    else:
        order = random.sample(range(len(b["pairs"])), count)
        how = "random"
    print(f"\n{'=' * 70}\n{title}: {count} samples ({how})\n{'=' * 70}")
    for k in order:
        print_pair(b, int(k))


# ---------------- SAVE (full texts, nothing clipped) ----------------

def dump(b):
    return [
        {
            "id_a": ids[i], "id_b": ids[j],
            "emb": round(float(b["emb"][k]), 4),
            "tfidf": round(float(b["lex"][k]), 4),
            "label_a": labels[i], "label_b": labels[j],
            "raw_a": raw_texts[i], "raw_b": raw_texts[j],
            "emb_text_a": emb_texts[i], "emb_text_b": emb_texts[j],
            "tfidf_text_a": lex_texts[i], "tfidf_text_b": lex_texts[j],
        }
        for k, (i, j) in enumerate(b["pairs"])
    ]


report = {
    "emb_threshold": EMB_THRESHOLD,
    "tfidf_threshold": TFIDF_THRESHOLD,
    "posts_compared": n,
    "counts": {k: len(v["pairs"]) for k, v in buckets.items()},
    "union": len(emb_pairs | tfidf_pairs),
    "pairs": {k: dump(v) for k, v in buckets.items()},
}
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print(f"\nSaved: {REPORT_OUT}")