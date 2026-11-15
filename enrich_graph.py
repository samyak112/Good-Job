import json
from collections import Counter, defaultdict, deque

GRAPH_PATH = "similarity_graph.json"
OUTPUT_PATH = "similarity_graph_marked.json"   # the original file is not modified
DEDUP_THRESHOLD = 0.90                         # edges strictly above this define duplicates


# ---------------- LOAD ----------------

with open(GRAPH_PATH, encoding="utf-8") as f:
    graph = json.load(f)

nodes = {str(n["id"]): n for n in graph["nodes"]}

adj = defaultdict(set)
for e in graph["edges"]:
    if e["similarity"] > DEDUP_THRESHOLD:
        a, b = str(e["source"]), str(e["target"])
        adj[a].add(b)
        adj[b].add(a)


# ---------------- CLUSTERS ----------------

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
    clusters.sort(key=lambda c: (-len(c), c[0]))   # biggest first, stable numbering
    return clusters


# ---------------- DECISION INSIDE ONE CLUSTER ----------------

def decide(cluster):
    """Keep the post with the most remaining links, remove its neighbors, repeat."""
    alive = set(cluster)
    keepers = []    # (keeper id, step)
    removed = {}    # removed id -> (keeper id, step)
    step = 0

    while True:
        live_degree = {n: sum(m in alive for m in adj[n]) for n in alive}
        live_degree = {n: d for n, d in live_degree.items() if d > 0}
        if not live_degree:
            break

        keeper = min(
            live_degree,
            key=lambda n: (-live_degree[n], -len(nodes[n]["text"]), n),
        )
        step += 1
        for m in sorted(m for m in adj[keeper] if m in alive):
            removed[m] = (keeper, step)
            alive.discard(m)
        alive.discard(keeper)
        keepers.append((keeper, step))

    leftovers = sorted(alive)   # in the cluster, but no link left to a kept post
    return keepers, removed, leftovers


# ---------------- MARK NODES ----------------

node_info = {}

for cluster_id, cluster in enumerate(find_clusters(adj)):
    keepers, removed, leftovers = decide(cluster)

    labels = {str(nodes[n].get("label")) for n in cluster}
    base = {
        "cluster_id": cluster_id,
        "cluster_size": len(cluster),
        "label_conflict": len(labels) > 1,
    }

    replaces = Counter(k for k, _ in removed.values())

    for k, step in keepers:
        node_info[k] = {**base, "role": "keeper", "decision": "keep",
                        "decision_step": step, "replaced_by": None,
                        "replaces_count": replaces[k], "degree": len(adj[k])}
    for m, (k, step) in removed.items():
        node_info[m] = {**base, "role": "removed", "decision": "remove",
                        "decision_step": step, "replaced_by": k,
                        "replaces_count": 0, "degree": len(adj[m])}
    for n in leftovers:
        node_info[n] = {**base, "role": "leftover", "decision": "keep",
                        "decision_step": None, "replaced_by": None,
                        "replaces_count": 0, "degree": len(adj[n])}

NO_DUPLICATE = {
    "cluster_id": None, "cluster_size": 1, "label_conflict": False,
    "role": "unique", "decision": "keep", "decision_step": None,
    "replaced_by": None, "replaces_count": 0, "degree": 0,
}

for n in graph["nodes"]:
    n.update(node_info.get(str(n["id"]), NO_DUPLICATE))


# ---------------- MARK EDGES ----------------
# replaced_by      : above threshold, the link that caused a deletion (removed -> keeper)
# other_duplicate  : above threshold, but not the deciding link (disappears with its node)
# related_survives : at or below threshold, both posts stay
# related_dropped  : at or below threshold, one post is removed so the edge disappears

removed_ids = {n for n, info in node_info.items() if info["decision"] == "remove"}

for e in graph["edges"]:
    a, b = str(e["source"]), str(e["target"])
    sim = e["similarity"]
    a_removed, b_removed = a in removed_ids, b in removed_ids

    e["is_duplicate"] = sim > DEDUP_THRESHOLD
    e["will_be_deleted"] = a_removed or b_removed

    if sim > DEDUP_THRESHOLD:
        if a_removed and node_info[a]["replaced_by"] == b:
            e["role"], e["removed_id"], e["keeper_id"] = "replaced_by", a, b
        elif b_removed and node_info[b]["replaced_by"] == a:
            e["role"], e["removed_id"], e["keeper_id"] = "replaced_by", b, a
        else:
            e["role"] = "other_duplicate"
    else:
        e["role"] = "related_dropped" if e["will_be_deleted"] else "related_survives"


# ---------------- SAVE ----------------

role_counts = Counter(n["role"] for n in graph["nodes"])
edge_counts = Counter(e["role"] for e in graph["edges"])

graph["metadata"]["dedup_threshold"] = DEDUP_THRESHOLD
graph["metadata"]["marked_nodes"] = dict(role_counts)
graph["metadata"]["marked_edges"] = dict(edge_counts)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(graph, f, ensure_ascii=False, indent=2, default=str)

print("Nodes:")
for role in ["keeper", "removed", "leftover", "unique"]:
    print(f"  {role:<9} {role_counts.get(role, 0):,}")
print("Edges:")
for role in ["replaced_by", "other_duplicate", "related_survives", "related_dropped"]:
    print(f"  {role:<17} {edge_counts.get(role, 0):,}")
print(f"\nSaved to {OUTPUT_PATH} (nothing was deleted)")