"""TRACE trades from the data platform's delivery. See docs/specs/b_liquidity.md.

The delivery is read through Spark: only the Spark Connect server holds the bucket's
credentials, so this machine keeps no keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import ensure_outside_repo, load_config


def check_delivery(
    manifests: dict[str, dict[str, Any]], counts: dict[str, int], months: list[str], required: list[str]
) -> None:
    """Raise ValueError naming what is wrong unless every month is delivered, each month's row
    count matches its manifest, and each manifest lists every required column.

    manifests: month ("YYYY-MM") to its manifest. counts: month to the rows Spark counted.
    """
    missing = [month for month in months if month not in manifests]
    if missing:
        raise ValueError(f"Months missing from the delivery: {missing}")
    differ = _differing_counts(manifests, counts, months)
    if differ:
        raise ValueError(f"Row counts differ from the manifest for: {differ}")
    absent = _absent_columns(manifests, months, required)
    if absent:
        raise ValueError(f"Columns missing from the delivery: {absent}")


def _differing_counts(manifests: dict[str, dict[str, Any]], counts: dict[str, int], months: list[str]) -> list[str]:
    """Months whose counted rows differ from their manifest."""
    return [month for month in months if counts.get(month, 0) != manifests[month]["rows"]]


def _absent_columns(manifests: dict[str, dict[str, Any]], months: list[str], required: list[str]) -> dict:
    """Month to the required columns its manifest does not list, for months that lack any."""
    absent = {month: sorted(set(required) - {col["name"] for col in manifests[month]["columns"]}) for month in months}
    return {month: cols for month, cols in absent.items() if cols}


def read_trades(
    spark: SparkSession, trace_dir: str, table: str, months: list[str], config: dict[str, Any] | None = None
) -> DataFrame:
    """The delivered trades for ``months``, after checking the delivery against its manifests.

    Adds a ``month`` column ("YYYY-MM"). A local ``trace_dir`` must lie outside the repo.
    """
    if "://" not in trace_dir:
        ensure_outside_repo(Path(trace_dir), "TRACE delivery")
    required = (config or load_config())["trace"]["required_columns"]
    root = f"{trace_dir.rstrip('/')}/{table}"

    # Spark skips files whose names start with "_" or ".", so the manifest is manifest.json, and the
    # trades are read with a *.parquet filter so the manifests stay out of the scan.
    found = spark.read.json(f"{root}/year=*/month=*/manifest.json", multiLine=True).collect()
    manifests = {row["month"]: row.asDict(recursive=True) for row in found}
    trades = spark.read.option("pathGlobFilter", "*.parquet").parquet(root)
    month = F.concat_ws("-", F.col("year").cast("string"), F.lpad(F.col("month").cast("string"), 2, "0"))
    trades = trades.withColumn("month", month).drop("year").filter(F.col("month").isin(months))
    counts = {row["month"]: row["count"] for row in trades.groupBy("month").count().collect()}
    check_delivery(manifests, counts, months, required)
    return trades
