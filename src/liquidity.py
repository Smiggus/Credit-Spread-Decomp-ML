"""Stage b: monthly liquidity measures per bond from cleaned TRACE trades.

See docs/specs/b_liquidity.md. Input trades use the plain names from ``trace.plain_trades``:
cusip_id, trd_exctn_dt, trd_seq (execution time), msg_seq (breaks ties within a second),
rptd_pr, par_usd.
"""

from __future__ import annotations

import datetime as dt
import operator
from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

KEY = ["cusip_id", "month"]
DAY = ["cusip_id", "month", "trd_exctn_dt"]
ORDER = ["trd_seq", "msg_seq"]
SCALED = ["amihud", "amihud_risk", "irc", "irc_risk"]  # the four measures lambda adds up


def _before_snapshot(trades: DataFrame, snapshots: dict[str, dt.date]) -> DataFrame:
    """Trades with a positive price and size in the snapshot months, before each month's snapshot
    day; adds ``month``."""
    days = trades.sparkSession.createDataFrame(list(snapshots.items()), "month string, snapshot_date date")
    trades = trades.filter((F.col("rptd_pr") > 0) & (F.col("par_usd") > 0))
    trades = trades.withColumn("month", F.date_format("trd_exctn_dt", "yyyy-MM"))
    return trades.join(days, "month").filter(F.col("trd_exctn_dt") < F.col("snapshot_date")).drop("snapshot_date")


def _daily(trades: DataFrame) -> DataFrame:
    """Daily Amihud and IRC per bond from trades already limited to before the snapshot day.

    Amihud: mean over the day's trades of |return from the previous trade| / par in USD millions.
    IRC: a roundtrip is two or three trades of the same size on one day, with no other trade of that
    size that day; IRC is the day's mean over roundtrips of (highest - lowest price) / highest price.
    """
    previous = F.lag("rptd_pr").over(Window.partitionBy(*DAY).orderBy(*ORDER))
    impact = F.abs(F.col("rptd_pr") / previous - 1) / (F.col("par_usd") / 1e6)
    amihud = trades.withColumn("impact", impact).groupBy(*DAY).agg(F.avg("impact").alias("amihud"))

    sizes = trades.groupBy(*DAY, "par_usd").agg(
        F.count("*").alias("n"), F.max("rptd_pr").alias("high"), F.min("rptd_pr").alias("low")
    )
    roundtrips = sizes.filter(F.col("n").between(2, 3))
    irc = roundtrips.groupBy(*DAY).agg(F.avg((F.col("high") - F.col("low")) / F.col("high")).alias("irc"))
    return amihud.join(irc, DAY, "full")


def daily_measures(trades: DataFrame, snapshots: dict[str, dt.date]) -> DataFrame:
    """Daily Amihud and IRC per bond, from trades before each month's snapshot day."""
    return _daily(_before_snapshot(trades, snapshots))


def _roll(trades: DataFrame) -> DataFrame:
    """Roll's spread per bond and month: 2 x sqrt(-cov(dP, previous dP)) over consecutive trades,
    0 when the covariance is positive, empty with fewer than three price changes."""
    order = Window.partitionBy(*KEY).orderBy("trd_exctn_dt", *ORDER)
    changes = trades.withColumn("dp", F.col("rptd_pr") - F.lag("rptd_pr").over(order))
    pairs = changes.withColumn("dp_before", F.lag("dp").over(order))
    cov = pairs.groupBy(*KEY).agg(F.count("dp").alias("n_changes"), F.covar_samp("dp", "dp_before").alias("cov"))
    roll = F.when(F.col("n_changes") < 3, None).when(F.col("cov") < 0, 2 * F.sqrt(-F.col("cov"))).otherwise(0.0)
    return cov.select(*KEY, roll.alias("roll"))


def _zero_trade_share(trades: DataFrame) -> DataFrame:
    """Share of the month's trading days (days with any trade in the input) on which the bond did
    not trade. A bond with no trades at all gets no row here; the join to the panel gives it 1.0."""
    market = trades.groupBy("month").agg(F.countDistinct("trd_exctn_dt").alias("market_days"))
    bond = trades.groupBy(*KEY).agg(F.countDistinct("trd_exctn_dt").alias("bond_days"))
    share = 1 - F.col("bond_days") / F.col("market_days")
    return bond.join(market, "month").select(*KEY, share.alias("zero_trade_share"))


def monthly_measures(trades: DataFrame, snapshots: dict[str, dt.date]) -> DataFrame:
    """Every measure except lambda, per bond and month, from trades before each snapshot day.

    Amihud and IRC: the median of the daily values; their risks: the standard deviation, empty
    with one day.
    """
    before = _before_snapshot(trades, snapshots)
    daily = _daily(before).groupBy(*KEY).agg(
        F.median("amihud").alias("amihud"), F.stddev_samp("amihud").alias("amihud_risk"),
        F.median("irc").alias("irc"), F.stddev_samp("irc").alias("irc_risk"),
    )
    return _zero_trade_share(before).join(daily, KEY, "left").join(_roll(before), KEY, "left")


def add_lambda(measures: DataFrame, train_months: list[str]) -> DataFrame:
    """Add lambda: the sum of the four Amihud and IRC measures, each standardised with the mean and
    standard deviation of the train months only. Empty when any of the four is empty.

    Raises ValueError if a measure has no spread in the train months, since it cannot be scaled.
    """
    train = measures.filter(F.col("month").isin(train_months))
    stats = train.agg(*[f(c).alias(f"{name}_{c}") for c in SCALED
                        for name, f in (("mean", F.avg), ("sd", F.stddev_samp))]).first()
    flat = [c for c in SCALED if not stats[f"sd_{c}"]]
    if flat:
        raise ValueError(f"No spread in the train months for: {flat}")
    scaled = [(F.col(c) - stats[f"mean_{c}"]) / stats[f"sd_{c}"] for c in SCALED]
    return measures.withColumn("lambda", reduce(operator.add, scaled))
