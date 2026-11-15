import json
from collections import defaultdict, deque

GRAPH_PATH = "similarity_graph_tfidf.json"
OUTPUT_PATH = "cluster_decisions.json"
DEDUP_THRESHOLD = 0.90   # only edges above this define duplicates

RAW_PATH = "data_deduped.json"        # never modified
CLEANED_PATH = "data_deduped_final.json"     # raw data minus deleted posts
REMOVED_LOG = "removed_ids.txt"

SKIP_LABEL_CONFLICT_CLUSTERS = False   # delete in every cluster, even where the posts have different labels
KEEP_IDS = set()                       # ids to protect, e.g. {"123", "456"}


# ---------------- LOAD GRAPH ----------------

with open(GRAPH_PATH, encoding="utf-8") as f:
    graph = json.load(f)

nodes = {str(n["id"]): n for n in graph["nodes"]}

adj = defaultdict(set)
for e in graph["edges"]:
    if e["similarity"] > DEDUP_THRESHOLD:
        a, b = str(e["source"]), str(e["target"])
        adj[a].add(b)
        adj[b].add(a)


# ---------------- FIND CLUSTERS ----------------

def find_clusters(adj):
    seen = set()
    clusters = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp = []
        queue = deque([start])
        seen.add(start)
        while queue:
            n = queue.popleft()
            comp.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
        clusters.append(sorted(comp))
    # biggest first, ties by first id, so cluster numbering is stable
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# ---------------- DECIDE INSIDE ONE CLUSTER ----------------

def decide(cluster):
    """Keep the post with the most remaining links, delete its neighbors, repeat."""
    alive = set(cluster)
    kept = []
    deleted = {}      # deleted id -> id of the kept post that replaces it
    decisions = []    # step-by-step record

    while True:
        live_degree = {n: sum(m in alive for m in adj[n]) for n in alive}
        live_degree = {n: d for n, d in live_degree.items() if d > 0}
        if not live_degree:
            break

        # most links, then longer text, then lowest id
        keeper = min(
            live_degree,
            key=lambda n: (-live_degree[n], -len(nodes[n]["text"]), n),
        )
        removed_now = sorted(m for m in adj[keeper] if m in alive)

        for m in removed_now:
            deleted[m] = keeper
            alive.discard(m)
        alive.discard(keeper)
        kept.append(keeper)

        decisions.append({
            "keeper": keeper,
            "links_when_chosen": live_degree[keeper],
            "deleted": removed_now,
        })

    # posts left in the cluster with no remaining links to any kept post
    kept.extend(sorted(alive))
    return kept, deleted, decisions


# ---------------- RUN ----------------

results = []
total_deleted = 0
remove_ids = set()          # ids actually removed from the raw data
skipped_conflict = 0
skipped_protected = 0

for cluster_id, cluster in enumerate(find_clusters(adj)):
    kept, deleted, decisions = decide(cluster)
    labels = sorted({str(nodes[n].get("label")) for n in cluster})
    conflict = len(labels) > 1
    total_deleted += len(deleted)

    applied = not (SKIP_LABEL_CONFLICT_CLUSTERS and conflict)
    if applied:
        for d in deleted:
            if d in KEEP_IDS:
                skipped_protected += 1
            else:
                remove_ids.add(d)
    else:
        skipped_conflict += len(deleted)

    results.append({
        "cluster_id": cluster_id,
        "size": len(cluster),
        "labels": labels,
        "label_conflict": conflict,
        "applied": applied,
        "kept": kept,
        "deleted": sorted(deleted),
        "replaced_by": deleted,
        "decisions": decisions,
    })

output = {
    "metadata": {
        "dedup_threshold": DEDUP_THRESHOLD,
        "total_nodes": len(nodes),
        "clusters": len(results),
        "posts_in_clusters": sum(r["size"] for r in results),
        "total_deleted": total_deleted,
        "deleted_from_raw": len(remove_ids),
        "skipped_label_conflict": skipped_conflict,
        "skipped_protected": skipped_protected,
        "clusters_with_label_conflict": sum(r["label_conflict"] for r in results),
    },
    "clusters": results,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)


# ---------------- DELETE FROM THE RAW DATA ----------------

with open(RAW_PATH, encoding="utf-8") as f:
    raw = json.load(f)

cleaned = [r for r in raw if str(r.get("id")) not in remove_ids]

with open(CLEANED_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

with open(REMOVED_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(remove_ids)))

print(f"Clusters: {len(results):,}")
print(f"Posts in clusters: {output['metadata']['posts_in_clusters']:,}")
print(f"Decided to delete: {total_deleted:,}")
print(f"  skipped, label-conflict clusters: {skipped_conflict:,}")
print(f"  skipped, protected by KEEP_IDS: {skipped_protected:,}")
print(f"Removed from raw data: {len(remove_ids):,}")
print(f"Raw records: {len(raw):,} -> {len(cleaned):,}")
print(f"Saved: {OUTPUT_PATH}, {CLEANED_PATH}, {REMOVED_LOG}")