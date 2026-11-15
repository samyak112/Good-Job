"""
graph_analysis.py

Analyses a similarity graph: finds duplicate clusters and decides which posts to keep.

    from graph_analysis import dedup_graph
    remove_ids, report = dedup_graph(graph, dedup_threshold=0.90)

Takes the graph dict produced by graph_building.GraphBuilder.build (nodes + edges).
"""

from collections import defaultdict, deque


def _find_clusters(adj):
    seen, clusters = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            x = queue.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    queue.append(y)
        clusters.append(sorted(comp))
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


def _decide(cluster, adj, nodes):
    """Keep the post with the most remaining links, delete its neighbours, repeat."""
    alive, kept, deleted, decisions = set(cluster), [], {}, []
    while True:
        deg = {x: sum(m in alive for m in adj[x]) for x in alive}
        deg = {x: d for x, d in deg.items() if d > 0}
        if not deg:
            break
        keeper = min(deg, key=lambda x: (-deg[x], -len(nodes[x]["text"]), x))
        removed_now = sorted(m for m in adj[keeper] if m in alive)
        for m in removed_now:
            deleted[m] = keeper
            alive.discard(m)
        alive.discard(keeper)
        kept.append(keeper)
        decisions.append({"keeper": keeper, "links_when_chosen": deg[keeper],
                          "deleted": removed_now})
    kept.extend(sorted(alive))
    return kept, deleted, decisions


def dedup_graph(graph, *, dedup_threshold=0.90, skip_label_conflict_clusters=False,
                keep_ids=frozenset()):
    """Returns (ids_to_remove, report). Same greedy logic as the original dedup script."""
    nodes = {str(n["id"]): n for n in graph["nodes"]}
    adj = defaultdict(set)
    for e in graph["edges"]:
        if e["similarity"] > dedup_threshold:
            a, b = str(e["source"]), str(e["target"])
            adj[a].add(b)
            adj[b].add(a)

    remove_ids, results = set(), []
    total_deleted = skipped_conflict = skipped_protected = 0

    for cid, cluster in enumerate(_find_clusters(adj)):
        kept, deleted, decisions = _decide(cluster, adj, nodes)
        labels = sorted({str(nodes[x].get("label")) for x in cluster})
        conflict = len(labels) > 1
        total_deleted += len(deleted)
        applied = not (skip_label_conflict_clusters and conflict)
        if applied:
            for d in deleted:
                if d in keep_ids:
                    skipped_protected += 1
                else:
                    remove_ids.add(d)
        else:
            skipped_conflict += len(deleted)
        results.append({"cluster_id": cid, "size": len(cluster), "labels": labels,
                        "label_conflict": conflict, "applied": applied, "kept": kept,
                        "deleted": sorted(deleted), "replaced_by": deleted,
                        "decisions": decisions})

    report = {
        "metadata": {
            "dedup_threshold": dedup_threshold,
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
    return remove_ids, report