"""TRACE trades from the data platform's delivery: check, read, map and clean them.

See docs/specs/b_liquidity.md. The delivery is read through Spark: only the Spark Connect
server holds the bucket's credentials, so this machine keeps no keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import ensure_outside_repo, load_config

# WRDS trade status codes. From February 2012: T trade, X cancel, C cancelled correction,
# R new correction, Y reversal. Before: T trade, C cancellation, W correction.
STATUS = {"T": "trade", "X": "cancel", "C": "cancel", "R": "correction", "W": "correction", "Y": "reversal"}


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
    """Months whose counted rows differ from their manifest; a month with no files counts as zero."""
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


def normalise(raw: DataFrame) -> DataFrame:
    """WRDS columns and status codes to the plain names the cleaning and measures use.

    Message sequence numbers are unique only within a report date, so they are keyed on it.
    An unknown status code becomes "unknown" and is dropped by the cleaning.
    """
    codes = F.create_map(*[F.lit(x) for pair in STATUS.items() for x in pair])
    status = F.coalesce(codes[F.col("trc_st")], F.lit("unknown"))
    orig = F.when(F.col("orig_msg_seq_nb").isNotNull(), F.concat_ws(":", "trd_rpt_dt", "orig_msg_seq_nb"))
    return raw.select(
        "cusip_id", "trd_exctn_dt", F.col("trd_exctn_tm").alias("trd_seq"), "rptd_pr",
        F.col("entrd_vol_qt").alias("par_usd"), F.col("cntra_mp_id").alias("contra"),
        F.col("rpt_side_cd").alias("side"), status.alias("status"),
        F.concat_ws(":", "trd_rpt_dt", "msg_seq_nb").alias("msg_seq"), orig.alias("orig_msg_seq"),
    )


def _drop_cancelled(trades: DataFrame) -> DataFrame:
    """Remove cancel reports and the reports they cancel."""
    cancelled = trades.filter(F.col("status") == "cancel").select("cusip_id", F.col("orig_msg_seq").alias("msg_seq"))
    return trades.filter(F.col("status") != "cancel").join(cancelled, ["cusip_id", "msg_seq"], "left_anti")


def _apply_corrections(trades: DataFrame) -> DataFrame:
    """Remove the reports that a correction replaces; the correction stays as the trade."""
    replaced = trades.filter(F.col("status") == "correction").select(
        "cusip_id", F.col("orig_msg_seq").alias("msg_seq")
    )
    return trades.join(replaced, ["cusip_id", "msg_seq"], "left_anti")


def _drop_reversed(trades: DataFrame) -> DataFrame:
    """Remove reversals and the trades they reverse, matched on the trade's details."""
    keys = ["cusip_id", "trd_exctn_dt", "trd_seq", "rptd_pr", "par_usd", "side", "contra"]
    reversed_ = trades.filter(F.col("status") == "reversal").select(*keys).distinct()
    return trades.filter(F.col("status") != "reversal").join(reversed_, keys, "left_anti")


def _drop_interdealer_buys(trades: DataFrame) -> DataFrame:
    """Both dealers report an interdealer trade; keep the seller's report only."""
    return trades.filter(~((F.col("contra") == "D") & (F.col("side") == "B")))


def _drop_unknown(trades: DataFrame) -> DataFrame:
    """Remove reports whose status code is not recognised."""
    return trades.filter(F.col("status").isin("trade", "correction"))


STEPS: dict[str, Callable[[DataFrame], DataFrame]] = {
    "cancelled": _drop_cancelled,
    "corrected": _apply_corrections,
    "reversed": _drop_reversed,
    "interdealer_buy_side": _drop_interdealer_buys,
    "unknown_status": _drop_unknown,
}


def clean_trades(trades: DataFrame) -> DataFrame:
    """Dick-Nielsen's Enhanced TRACE filter on plain-named trades, step by step."""
    for step in STEPS.values():
        trades = step(trades)
    return trades


def cleaning_counts(trades: DataFrame) -> dict[str, int]:
    """Reports removed by each cleaning step, plus the reports at the start and the end. Totals only."""
    counts = {"start": trades.count()}
    before = counts["start"]
    for name, step in STEPS.items():
        trades = step(trades)
        after = trades.count()
        counts[name] = before - after
        before = after
    counts["end"] = before
    return counts
