"""Spark session for the liquidity (b) and network (d) stages."""

from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from src.config import Settings, load_config, load_settings


def get_spark(settings: Settings | None = None, config: dict[str, Any] | None = None) -> SparkSession:
    """Build or reuse a Spark session, with the app name and timezone from ``config.yaml`` and the
    TIME type turned on.

    With ``SPARK_REMOTE`` set (an ``sc://`` URL), connect to a Spark Connect server. Jars
    such as GraphFrames must then be on the server's classpath, so ``GRAPHFRAMES_PACKAGE``
    is ignored.

    Without it, start Spark locally with ``SPARK_MASTER`` and add GraphFrames as a Spark
    package when ``GRAPHFRAMES_PACKAGE`` is set.
    """
    settings = settings or load_settings()
    spark_cfg = (config or load_config())["spark"]
    if settings.spark_remote:
        builder = SparkSession.builder.remote(settings.spark_remote)
    else:
        builder = SparkSession.builder.master(settings.spark_master)
        if settings.graphframes_package:
            builder = builder.config("spark.jars.packages", settings.graphframes_package)
    builder = builder.appName(spark_cfg["app_name"]).config("spark.sql.session.timeZone", spark_cfg["timezone"])
    # The TRACE delivery stores execution and report times as Parquet TIME, which Spark reads only with this on.
    builder = builder.config("spark.sql.timeType.enabled", "true")
    return builder.getOrCreate()
