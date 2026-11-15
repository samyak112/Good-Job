"""
graph_building.py

Builds the post-similarity graph.

    from graph_building import GraphBuilder, TfidfSimilarity
    graph = GraphBuilder(TfidfSimilarity(), 0.90, require_same_label=True).build(records)

The similarity method is a plug-in: subclass SimilarityMethod (EmbeddingSimilarity and
TfidfSimilarity are the built-ins). GraphBuilder handles loading, the min-length filter,
label gating, the chunked upper-triangle scan and the node/edge/metadata output.

pip install pandas numpy scikit-learn scipy sentence-transformers
"""

import hashlib
import html
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

COLUMNS = ["id", "post_text", "author", "label", "poster", "url"]


# ======================= NORMALIZERS =======================

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")


_TF_URL = re.compile(
    r"(?i)(?:https?\s*:\s*/\s*/\S*|www\.\S+"
    r"|\b(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|buff\.ly|lnkd\.in)/\S+"
    r"|\b(?:[\w-]+\.)+(?:com|org|net|io|co|in|dev|ai|edu|gov|me|app|tech|xyz)\b(?:/\S*)?)"
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
_TRACK = re.compile(r"(?i)\b(?:utm_\w+|referralcode|ref|source|campaign)\b\s*=?\s*\S*")


def strip_links(text):
    """Shared by both normalizers: delete URLs, emails, UUIDs and tracking params."""
    text = _TF_URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _UUID.sub(" ", text)
    return _TRACK.sub(" ", text)


def normalize(text, keep_structure=False):
    """
    One normalizer for both methods: unescape, NFKC, '&' -> 'and', strip links,
    drop symbols, collapse whitespace.
    keep_structure=True also keeps ': | / \\ -' (spaced out) for the embedding model,
    and keeps case. keep_structure=False strips all punctuation except '#' and lowercases
    (what TF-IDF needs).
    """
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKC", text).replace("&", " and ")
    text = strip_links(text)
    if keep_structure:
        text = re.sub(r"([:|/\\\-\n])", r" \1 ", text)
        text = re.sub(r"[^\w\s#:|/\\-]", " ", text, flags=re.UNICODE)
    else:
        text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text if keep_structure else text.lower()


# ======================= LOADING =======================

def _prepare(records, normalizer, min_chars):
    df = pd.json_normalize(records).reindex(columns=COLUMNS).copy()
    df["text"] = df["post_text"].fillna("").apply(normalizer)
    df = df[df["text"].str.len() >= min_chars].reset_index(drop=True)
    return df.astype(object).where(pd.notna(df), None)


def _make_gate(df, require_same_label):
    labels = df["label"].astype(str).tolist()

    def gate(i, j):
        """None if the pair may be linked, else the reason it is blocked."""
        if require_same_label and labels[i] != labels[j]:
            return "label"
        return None

    return gate


def _nodes(df):
    return [
        {
            "id": str(df.at[i, "id"]),
            "text": df.at[i, "text"],
            "author": df.at[i, "author"],
            "label": df.at[i, "label"],
            "poster": df.at[i, "poster"],
            "url": df.at[i, "url"],
        }
        for i in range(len(df))
    ]


# ======================= SIMILARITY METHODS =======================

class SimilarityMethod:
    """
    Plug-in scoring strategy. Subclass this to add a new similarity method.

    normalize(text)  -> the text form this method wants to score
    prepare(df, ...) -> fit / embed once over all normalized posts (df has a "text" column)
    pairs(start, stop, threshold)
                     -> (rows, cols, scores) for rows start:stop against ALL posts,
                        scores >= threshold, as global row/col indices
    """
    name = "base"

    def normalize(self, text):
        raise NotImplementedError

    def prepare(self, df, cache_dir=None, verbose=True):
        raise NotImplementedError

    def pairs(self, start, stop, threshold):
        raise NotImplementedError

    def describe(self):
        return {"method": self.name}


class EmbeddingSimilarity(SimilarityMethod):
    """Sentence-embedding cosine similarity (dense)."""
    name = "embedding"

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.emb = None

    def normalize(self, text):
        return normalize(text, keep_structure=True)

    def prepare(self, df, cache_dir=None, verbose=True):
        n = len(df)
        fingerprint = hashlib.sha256(
            "\n".join(df["id"].astype(str) + ":" + df["text"].astype(str)).encode("utf-8")
        ).hexdigest()

        cache_npy = cache_meta = None
        if cache_dir:
            cache_npy = os.path.join(cache_dir, "embeddings.npy")
            cache_meta = os.path.join(cache_dir, "embeddings.meta")
            if os.path.exists(cache_npy) and os.path.exists(cache_meta):
                with open(cache_meta, encoding="utf-8") as f:
                    if json.load(f).get("fingerprint") == fingerprint:
                        cached = np.load(cache_npy)
                        if len(cached) == n:
                            self.emb = cached
                            if verbose:
                                print("[embedding] loaded cached embeddings")
                            return

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        self.emb = model.encode(
            df["text"].tolist(), batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=verbose,
        ).astype("float32")
        if cache_npy:
            np.save(cache_npy, self.emb)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fingerprint}, f)

    def pairs(self, start, stop, threshold):
        sims = self.emb[start:stop] @ self.emb.T          # rows are L2-normalized
        r, c = np.nonzero(sims >= threshold)
        return r + start, c, sims[r, c]

    def describe(self):
        return {"method": self.name, "model": self.model_name}


class TfidfSimilarity(SimilarityMethod):
    """1-2gram TF-IDF cosine similarity (sparse)."""
    name = "tfidf"

    def __init__(self, ngram_range=(1, 2), min_df=1, max_df=0.8, sublinear_tf=True):
        self.params = dict(ngram_range=ngram_range, min_df=min_df,
                           max_df=max_df, sublinear_tf=sublinear_tf)
        self.X = self.XT = None

    def normalize(self, text):
        return normalize(text, keep_structure=False)

    def prepare(self, df, cache_dir=None, verbose=True):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(**self.params)
        self.X = normalize(vec.fit_transform(df["text"].tolist())).tocsr()
        self.XT = self.X.T.tocsc()
        if verbose:
            print(f"[tfidf] matrix: {self.X.shape[0]:,} x {self.X.shape[1]:,}")

    def pairs(self, start, stop, threshold):
        block = (self.X[start:stop] @ self.XT).tocoo()
        mask = block.data >= threshold
        return block.row[mask] + start, block.col[mask], block.data[mask]

    def describe(self):
        return {"method": self.name, **{k: list(v) if isinstance(v, tuple) else v
                                        for k, v in self.params.items()}}


# ======================= GRAPH BUILDER =======================

class GraphBuilder:
    """
    Builds the similarity graph for any SimilarityMethod. Handles loading, the
    min-length filter, label/number gating, the chunked upper-triangle scan and
    the node/edge/metadata output. The method only supplies the scores.

        g = GraphBuilder(TfidfSimilarity(), threshold=0.90,
                         require_same_label=True).build(records)
    """

    def __init__(self, method, threshold, *, min_chars=20, chunk=1000,
                 require_same_label=False):
        self.method = method
        self.threshold = threshold
        self.min_chars = min_chars
        self.chunk = chunk
        self.require_same_label = require_same_label

    def build(self, records, cache_dir=None, verbose=True):
        m = self.method
        df = _prepare(records, m.normalize, self.min_chars)
        ids = df["id"].astype(str).tolist()
        n = len(df)
        if verbose:
            print(f"[{m.name}] posts compared: {n:,}")

        m.prepare(df, cache_dir=cache_dir, verbose=verbose)
        gate = _make_gate(df, self.require_same_label)

        edges, skipped = [], []
        skipped_label = 0

        for start in range(0, n, self.chunk):
            rows, cols, vals = m.pairs(start, min(start + self.chunk, n), self.threshold)
            keep = cols > rows                                 # upper triangle only
            for i, j, v in zip(rows[keep], cols[keep], vals[keep]):
                i, j, v = int(i), int(j), round(float(v), 4)
                reason = gate(i, j)
                if reason is None:
                    edges.append({"source": ids[i], "target": ids[j], "similarity": v})
                else:
                    skipped.append({"source": ids[i], "target": ids[j],
                                    "similarity": v, "blocked_by": reason})
                    skipped_label += 1

        return {
            "nodes": _nodes(df), "edges": edges, "skipped": skipped,
            "metadata": {"total_nodes": n, "total_edges": len(edges),
                         "threshold": self.threshold, **m.describe(),
                         "require_same_label": self.require_same_label,
                         "skipped_label_conflict": skipped_label},
        }