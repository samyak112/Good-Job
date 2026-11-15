
import json

INPUT_PATH = "similarity_graph_new.json"
OUTPUT_PATH = "neo4j_import.cypher"

NODE_BATCH_SIZE = 500
EDGE_BATCH_SIZE = 1000


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
    properties = {
        "id": node["id"],
        "text": node["text"],
        "author": node.get("author"),
        "label": node.get("label"),
        "poster": node.get("poster"),
        "url": node.get("url"),
    }

    return "{" + ", ".join(
        f"{key}: {cypher_value(value)}"
        for key, value in properties.items()
    ) + "}"


def edge_to_cypher(edge):
    return (
        "{"
        f"source: {cypher_value(edge['source'])}, "
        f"target: {cypher_value(edge['target'])}, "
        f"score: {cypher_value(edge['similarity'])}"
        "}"
    )


def write_import(graph):
    nodes = graph["nodes"]
    edges = graph["edges"]

    node_batches = list(chunks(nodes, NODE_BATCH_SIZE))
    edge_batches = list(chunks(edges, EDGE_BATCH_SIZE))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        # Clear previous graph
        f.write("""
MATCH (p:Post)
DETACH DELETE p;

""")

        # Recreate constraint
        f.write("""
CREATE CONSTRAINT post_id IF NOT EXISTS
FOR (p:Post)
REQUIRE p.id IS UNIQUE;

""")

        # Import nodes
        for batch_number, batch in enumerate(node_batches, 1):
            values = ",\n".join(
                node_to_cypher(node) for node in batch
            )

            f.write(f"""
// Node batch {batch_number}/{len(node_batches)}
UNWIND [
{values}
] AS node

MERGE (p:Post {{id: node.id}})
SET p += node;

""")

        # Import relationships
        for batch_number, batch in enumerate(edge_batches, 1):
            values = ",\n".join(
                edge_to_cypher(edge) for edge in batch
            )

            f.write(f"""
// Relationship batch {batch_number}/{len(edge_batches)}
UNWIND [
{values}
] AS edge

MATCH (a:Post {{id: edge.source}})
MATCH (b:Post {{id: edge.target}})
MERGE (a)-[r:SIMILAR_TO]-(b)
SET r.score = edge.score;

""")

    total_statements = 2 + len(node_batches) + len(edge_batches)

    print(f"Generated: {OUTPUT_PATH}")
    print(f"Nodes: {len(nodes):,}")
    print(f"Relationships: {len(edges):,}")
    print(f"Node batches: {len(node_batches):,}")
    print(f"Relationship batches: {len(edge_batches):,}")
    print(f"Total Cypher statements: {total_statements:,}")


with open(INPUT_PATH, encoding="utf-8") as f:
    graph = json.load(f)

write_import(graph)
