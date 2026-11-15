"""
clean_pipeline.py

    from clean_pipeline import clean_dataset
    clean_dataset("data_original.json", "training_data.json")

Pass 1: MiniLM embedding graph -> greedy cluster dedup
Pass 2: TF-IDF graph           -> greedy cluster dedup
Graph building lives in graph_builder.py, cluster analysis / dedup decisions in
graph_analysis.py. The input file is never modified.
"""

import json
import os

from graph_analysis import dedup_graph
from graph_builder import EmbeddingSimilarity, GraphBuilder, TfidfSimilarity

# ---------------- THRESHOLDS ----------------
# GRAPH = minimum similarity for an edge to exist in the graph
# DEDUP = dedup only acts on edges strictly above this (keep GRAPH <= DEDUP)
EMBED_GRAPH_THRESHOLD = 0.85
EMBED_DEDUP_THRESHOLD = 0.90
TFIDF_GRAPH_THRESHOLD = 0.85
TFIDF_DEDUP_THRESHOLD = 0.90


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def clean_dataset(input_path, output_path, work_dir="data_pipeline/cleaning_work",
                  passes=None, keep_ids=(), save_graphs=False):
    """
    Run each dedup pass on the records and write the cleaned result to output_path.
    Logs go to work_dir: cluster_decisions_<pass>.json, removed_ids.txt, and the
    graphs themselves if save_graphs=True (e.g. for the Neo4j export).

    passes: list of (name, GraphBuilder, dedup_threshold), run in order.
            Default: embedding then tfidf, using the thresholds at the top of this file,
            never linking posts with different labels.
    keep_ids: post ids that must never be deleted.
    """
    os.makedirs(work_dir, exist_ok=True)
    keep_ids = {str(k) for k in keep_ids}

    if passes is None:
        passes = [
            ("embedding",
             GraphBuilder(EmbeddingSimilarity(), EMBED_GRAPH_THRESHOLD, require_same_label=True),
             EMBED_DEDUP_THRESHOLD),
            ("tfidf",
             GraphBuilder(TfidfSimilarity(), TFIDF_GRAPH_THRESHOLD, require_same_label=True),
             TFIDF_DEDUP_THRESHOLD),
        ]

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    raw_count = len(records)
    removed_log = []

    for name, builder, dedup_threshold in passes:
        graph = builder.build(records, cache_dir=work_dir)
        remove_ids, report = dedup_graph(graph, dedup_threshold=dedup_threshold, keep_ids=keep_ids)

        if save_graphs:
            _dump({k: v for k, v in graph.items() if k != "skipped"},
                  os.path.join(work_dir, f"graph_{name}.json"))
        _dump(report, os.path.join(work_dir, f"cluster_decisions_{name}.json"))
        removed_log += [f"{name}\t{i}" for i in sorted(remove_ids)]

        before = len(records)
        records = [r for r in records if str(r.get("id")) not in remove_ids]
        m = report["metadata"]
        print(f"[{name}] clusters={m['clusters']:,} removed={len(remove_ids):,} "
              f"| {before:,} -> {len(records):,}")

    _dump(records, output_path)
    with open(os.path.join(work_dir, "removed_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(removed_log))

    print(f"Done: {raw_count:,} -> {len(records):,} | saved {output_path}")
    return records


if __name__ == "__main__":
    clean_dataset("data_pipeline/dataset/data_original.json", "data_pipeline/dataset/training_data.json",save_graphs=True)