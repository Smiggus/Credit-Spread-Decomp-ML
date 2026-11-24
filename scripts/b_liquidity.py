"""Stage b on the cluster: check and clean the TRACE delivery, build the monthly liquidity measures
for the panel's bonds, and bring them to LIQUIDITY_PATH (a parquet file outside the repo). Also
writes reports/b_cleaning_counts.csv. Totals only.

Run from the repo root with SPARK_REMOTE, TRACE_DIR and LIQUIDITY_PATH set: python -m scripts.b_liquidity
The measures cover the panel's train, gap and valid months; the test months stay closed until the pricer.
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src.config import REPO_ROOT, ensure_outside_repo, load_config, load_settings, require
from src.liquidity import add_lambda, monthly_measures, snapshot_days
from src.panel import load_panel
from src.spark import get_spark
from src.trace import clean_trades, cleaning_counts, plain_trades, read_trades

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports"


def cleaned_trades(
    spark: SparkSession, trace_dir: str, months: list[str], config: dict[str, Any]
) -> tuple[DataFrame, dict[str, dict[str, int]]]:
    """The cleaned trades of every TRACE table for ``months``, and each table's cleaning counts."""
    counts, trades = {}, []
    for table in config["trace"]["tables"]:
        reports = read_trades(spark, trace_dir, table, months, config).select(*config["trace"]["required_columns"])
        counts[table] = cleaning_counts(reports)
        logger.info("%s cleaning counts: %s", table, counts[table])
        trades.append(plain_trades(clean_trades(reports)))
    return reduce(DataFrame.unionByName, trades), counts


def main() -> None:
    """Write the measures to LIQUIDITY_PATH and the cleaning counts to ``reports/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings, config = load_settings(), load_config()
    out = ensure_outside_repo(require(settings.liquidity_path, "LIQUIDITY_PATH"), "liquidity measures")
    panel = load_panel()
    snapshots = snapshot_days(panel)
    spark = get_spark(settings, config)

    trades, counts = cleaned_trades(spark, require(settings.trace_dir, "TRACE_DIR"), sorted(snapshots), config)
    # Market trading days come from every bond's trades; the panel's bonds are kept after that, so
    # lambda's scale comes from the panel's bonds in the train months.
    bonds = spark.createDataFrame(pd.DataFrame({"cusip_id": panel["cusip_id"].unique()}))
    measures = monthly_measures(trades, snapshots).join(bonds, "cusip_id")
    train_months = sorted(snapshot_days(panel[panel["partition"] == "train"]))
    table = add_lambda(measures, train_months).toPandas()

    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    pd.DataFrame(counts).rename_axis("step").reset_index().to_csv(REPORTS / "b_cleaning_counts.csv", index=False)
    logger.info("Wrote %d bond-months of measures and the cleaning counts", len(table))


if __name__ == "__main__":
    main()
