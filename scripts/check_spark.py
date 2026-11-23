"""Check that this repo's client can use the Spark cluster and GraphFrames.

Run from the repo root: python -m scripts.check_spark

Uses only a made-up five-node graph, so no data is needed.
"""

from __future__ import annotations

import logging

from graphframes import GraphFrame

from src.spark import get_spark

logger = logging.getLogger(__name__)

TOY_VERTICES = [("a",), ("b",), ("c",), ("d",), ("e",)]
TOY_EDGES = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")]


def check_spark() -> dict[str, object]:
    """Connect, run PageRank and connected components on a toy graph, return the results."""
    spark = get_spark()
    vertices = spark.createDataFrame(TOY_VERTICES, ["id"])
    edges = spark.createDataFrame(TOY_EDGES, ["src", "dst"])
    graph = GraphFrame(vertices, edges)

    ranks = graph.pageRank(resetProbability=0.15, maxIter=5).vertices
    ranked = {row["id"]: round(row["pagerank"], 3) for row in ranks.collect()}
    components = graph.connectedComponents().select("id", "component").collect()
    n_components = len({row["component"] for row in components})

    return {"spark_version": spark.version, "pagerank": ranked, "components": n_components}


def main() -> None:
    """Log the check results; raise if the toy graph does not give the expected answer."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = check_spark()
    logger.info("Spark version: %s", result["spark_version"])
    logger.info("PageRank: %s", result["pagerank"])
    logger.info("Connected components: %s (expected 2)", result["components"])
    if result["components"] != 2:
        raise RuntimeError("GraphFrames returned the wrong number of components for the toy graph.")


if __name__ == "__main__":
    main()
