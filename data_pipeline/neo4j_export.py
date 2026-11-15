"""
neo4j_export.py

Turns a saved similarity graph into a Cypher import script for Neo4j.
Each method gets its own file, so you can load and look at the embedding graph and the
tfidf graph separately:

    cleaning_work/graph_embedding.json -> neo4j_import_embedding.cypher
    cleaning_work/graph_tfidf.json     -> neo4j_import_tfidf.cypher

The graph files only exist if the pipeline ran with save_graphs=True:

    clean_dataset("data_original.json", "training_data.json", save_graphs=True)

Every generated script starts by deleting all :Post nodes, so load ONE script at a time
(running the second one replaces the first).
"""

import json
import os

WORK_DIR = "data_pipeline/cleaning_work"       # where clean_dataset saved graph_<method>.json
OUTPUT_DIR = "data_pipeline/cleaning_work"                 # where the .cypher files are written
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


def write_import(graph, output_path):
    nodes = graph["nodes"]
    edges = graph["edges"]

    node_batches = list(chunks(nodes, NODE_BATCH_SIZE))
    edge_batches = list(chunks(edges, EDGE_BATCH_SIZE))

    with open(output_path, "w", encoding="utf-8") as f:

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

    print(f"Generated: {output_path}")
    print(f"Nodes: {len(nodes):,}")
    print(f"Relationships: {len(edges):,}")
    print(f"Node batches: {len(node_batches):,}")
    print(f"Relationship batches: {len(edge_batches):,}")
    print(f"Total Cypher statements: {total_statements:,}")


def export_cypher(methods=("embedding", "tfidf")):
    """
    methods: a pass name ("embedding" or "tfidf") or a list of them.
    Writes one neo4j_import_<method>.cypher per method and returns the paths.
    """
    if isinstance(methods, str):
        methods = [methods]

    graph_paths = {m: os.path.join(WORK_DIR, f"graph_{m}.json") for m in methods}
    missing = [p for p in graph_paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run clean_dataset(..., save_graphs=True) first."
        )

    outputs = []
    for method, graph_path in graph_paths.items():
        with open(graph_path, encoding="utf-8") as f:
            graph = json.load(f)

        output_path = os.path.join(OUTPUT_DIR, f"neo4j_import_{method}.cypher")
        print(f"\n--- {method} (edges >= {graph['metadata']['threshold']}) ---")
        write_import(graph, output_path)
        outputs.append(output_path)

    return outputs


if __name__ == "__main__":
    export_cypher(["embedding","tfidf"])     # or "tfidf", or ["embedding", "tfidf"]