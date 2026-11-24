"""TRACE trades from the data platform's delivery: check, read and clean them.

See docs/specs/b_liquidity.md. The delivery is read through Spark: only the Spark Connect
server holds the bucket's credentials, so this machine keeps no keys.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import ensure_outside_repo, load_config

# TRACE moved to a new platform on this date and changed its status codes. The cleaning below
# follows the rules from this date on; earlier reports wait for #13.
NEW_PLATFORM = dt.date(2012, 2, 6)
# The trade details a cancel or reversal must share with the trade it removes.
DETAILS = ["cusip_id", "entrd_vol_qt", "rptd_pr", "rpt_side_cd", "cntra_mp_id", "trd_exctn_dt", "trd_exctn_tm"]


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


def _anti_join(trades: DataFrame, removers: DataFrame, keys: list[str]) -> DataFrame:
    """Trades with no match in ``removers`` on ``keys``. Empty values match each other."""
    left, right = trades.alias("trades"), removers.select(*keys).distinct().alias("removers")
    condition = [F.col(f"trades.{key}").eqNullSafe(F.col(f"removers.{key}")) for key in keys]
    return left.join(right, condition, "left_anti")


def _keep_trade_reports(trades: DataFrame, _reports: DataFrame) -> DataFrame:
    """Keep trade reports (T) and corrections (R, the corrected version of a trade)."""
    return trades.filter(F.col("trc_st").isin("T", "R"))


def _drop_cancelled(trades: DataFrame, reports: DataFrame) -> DataFrame:
    """A cancel (X) or cancelled correction (C) carries the message number and details of the
    trade it removes."""
    return _anti_join(trades, reports.filter(F.col("trc_st").isin("X", "C")), DETAILS + ["msg_seq_nb"])


def _drop_reversed(trades: DataFrame, reports: DataFrame) -> DataFrame:
    """A reversal (Y) points to the trade it removes through its original message number."""
    reversals = reports.filter(F.col("trc_st") == "Y").withColumn("msg_seq_nb", F.col("orig_msg_seq_nb"))
    return _anti_join(trades, reversals, DETAILS + ["msg_seq_nb"])


def _drop_interdealer_buys(trades: DataFrame, _reports: DataFrame) -> DataFrame:
    """Both dealers report an interdealer trade. Drop a dealer's buy report when a dealer sell
    report matches it on bond, day, volume and price."""
    dealer = F.col("cntra_mp_id").eqNullSafe("D")
    dealer_buy = dealer & F.col("rpt_side_cd").eqNullSafe("B")
    sells = trades.filter(dealer & F.col("rpt_side_cd").eqNullSafe("S"))
    buys = _anti_join(trades.filter(dealer_buy), sells, ["cusip_id", "trd_exctn_dt", "entrd_vol_qt", "rptd_pr"])
    return trades.filter(~dealer_buy).unionByName(buys)


def _drop_special(trades: DataFrame, _reports: DataFrame) -> DataFrame:
    """Drop when-issued, special and as-of trades, and trades settling more than 7 days after execution."""
    blank = [F.col(c).isNull() | (F.col(c) == "") for c in ("spcl_trd_fl", "asof_cd")]
    settles = F.col("stlmnt_dt").isNull() | (F.datediff("stlmnt_dt", "trd_exctn_dt") <= 7)
    days = F.col("days_to_sttl_ct").isNull() | (F.col("days_to_sttl_ct") <= 7)
    return trades.filter((F.col("wis_fl") == "N") & blank[0] & blank[1] & settles & days)


STEPS: dict[str, Callable[[DataFrame, DataFrame], DataFrame]] = {
    "not_a_trade_report": _keep_trade_reports,
    "cancelled": _drop_cancelled,
    "reversed": _drop_reversed,
    "interdealer_buy_side": _drop_interdealer_buys,
    "special_or_late": _drop_special,
}


def _refuse_old_platform(reports: DataFrame) -> None:
    if reports.filter(F.col("trd_rpt_dt") < F.lit(NEW_PLATFORM)).limit(1).count():
        raise ValueError("Reports from before 6 February 2012 follow other rules, which come with #13.")


def clean_trades(reports: DataFrame) -> DataFrame:
    """Dick-Nielsen's filter for Enhanced TRACE reported from 6 February 2012 on, on WRDS columns."""
    _refuse_old_platform(reports)
    trades = reports
    for step in STEPS.values():
        trades = step(trades, reports)
    return trades


def cleaning_counts(reports: DataFrame) -> dict[str, int]:
    """Reports removed by each cleaning step, plus the reports at the start and the end. Totals only."""
    _refuse_old_platform(reports)
    counts = {"start": reports.count()}
    trades, before = reports, counts["start"]
    for name, step in STEPS.items():
        trades = step(trades, reports)
        after = trades.count()
        counts[name], before = before - after, after
    counts["end"] = before
    return counts


def plain_trades(cleaned: DataFrame) -> DataFrame:
    """Cleaned trades with the plain names the measures use."""
    return cleaned.select(
        "cusip_id", "trd_exctn_dt", F.col("trd_exctn_tm").alias("trd_seq"), "rptd_pr",
        F.col("entrd_vol_qt").alias("par_usd"), F.col("cntra_mp_id").alias("contra"),
        F.col("rpt_side_cd").alias("side"),
    )
