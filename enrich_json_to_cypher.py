import json
from collections import defaultdict

INPUT_PATH = "similarity_graph_marked.json"   # output of enrich_graph.py
OUTPUT_PATH = "neo4j_import.cypher"

NODE_BATCH_SIZE = 500
EDGE_BATCH_SIZE = 1000

NODE_FIELDS = [
    "id", "text", "author", "label", "poster", "url",
    "cluster_id", "cluster_size", "label_conflict",
    "role", "decision", "decision_step", "replaced_by", "replaces_count", "degree",
]

# edge role -> (relationship type, directed?)
EDGE_TYPES = {
    "replaced_by":      ("REPLACED_BY", True),       # removed post -> the post that replaces it
    "other_duplicate":  ("OTHER_DUPLICATE", False),
    "related_survives": ("SIMILAR_TO", False),
    "related_dropped":  ("SIMILAR_DROPPED", False),
}


def cypher_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def node_to_cypher(node):
    return "{" + ", ".join(
        f"{key}: {cypher_value(node.get(key))}" for key in NODE_FIELDS
    ) + "}"


def edge_to_cypher(edge):
    if edge["role"] == "replaced_by":
        source, target = edge["removed_id"], edge["keeper_id"]
    else:
        source, target = edge["source"], edge["target"]
    return (
        "{"
        f"source: {cypher_value(str(source))}, "
        f"target: {cypher_value(str(target))}, "
        f"score: {cypher_value(edge['similarity'])}"
        "}"
    )


def write_import(graph):
    nodes = graph["nodes"]

    edges_by_role = defaultdict(list)
    for e in graph["edges"]:
        edges_by_role[e["role"]].append(e)

    node_batches = list(chunks(nodes, NODE_BATCH_SIZE))
    statements = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        f.write("""
MATCH (p:Post)
DETACH DELETE p;

CREATE CONSTRAINT post_id IF NOT EXISTS
FOR (p:Post)
REQUIRE p.id IS UNIQUE;

CREATE INDEX post_cluster IF NOT EXISTS
FOR (p:Post)
ON (p.cluster_id);

""")
        statements += 3

        for batch_number, batch in enumerate(node_batches, 1):
            values = ",\n".join(node_to_cypher(n) for n in batch)
            f.write(f"""
// Node batch {batch_number}/{len(node_batches)}
UNWIND [
{values}
] AS node

MERGE (p:Post {{id: node.id}})
SET p += node;

""")
            statements += 1

        for role, (rel_type, directed) in EDGE_TYPES.items():
            role_edges = edges_by_role.get(role, [])
            arrow = "->" if directed else "-"
            role_batches = list(chunks(role_edges, EDGE_BATCH_SIZE))

            for batch_number, batch in enumerate(role_batches, 1):
                values = ",\n".join(edge_to_cypher(e) for e in batch)
                f.write(f"""
// {rel_type} batch {batch_number}/{len(role_batches)}
UNWIND [
{values}
] AS edge

MATCH (a:Post {{id: edge.source}})
MATCH (b:Post {{id: edge.target}})
MERGE (a)-[r:{rel_type}]{arrow}(b)
SET r.score = edge.score;

""")
                statements += 1

        f.write("""
// Node labels so the browser can colour nodes by role
MATCH (p:Post) WHERE p.role = 'keeper'   SET p:Keeper;
MATCH (p:Post) WHERE p.role = 'removed'  SET p:Removed;
MATCH (p:Post) WHERE p.role = 'leftover' SET p:Leftover;
""")
        statements += 3

    print(f"Generated: {OUTPUT_PATH}")
    print(f"Nodes: {len(nodes):,}")
    for role, (rel_type, _) in EDGE_TYPES.items():
        print(f"  {rel_type}: {len(edges_by_role.get(role, [])):,}")
    print(f"Total Cypher statements: {statements:,}")


with open(INPUT_PATH, encoding="utf-8") as f:
    graph = json.load(f)

write_import(graph)