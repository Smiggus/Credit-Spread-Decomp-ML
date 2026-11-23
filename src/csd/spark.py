"""Spark session for the liquidity (b) and network (d) stages."""

from __future__ import annotations

from pyspark.sql import SparkSession

from csd.config import Settings, load_settings


def get_spark(app_name: str = "csd", settings: Settings | None = None) -> SparkSession:
    """Build or reuse a Spark session.

    With SPARK_REMOTE set (an sc:// URL), connect to a Spark Connect server. Jars such as
    GraphFrames must then be on the server's classpath, so GRAPHFRAMES_PACKAGE is ignored.

    Without it, start Spark locally with SPARK_MASTER (default local[*]) and add
    GraphFrames as a Spark package when GRAPHFRAMES_PACKAGE is set.
    """
    settings = settings or load_settings()
    if settings.spark_remote:
        return SparkSession.builder.remote(settings.spark_remote).appName(app_name).getOrCreate()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
    )
    if settings.graphframes_package:
        builder = builder.config("spark.jars.packages", settings.graphframes_package)
    return builder.getOrCreate()
