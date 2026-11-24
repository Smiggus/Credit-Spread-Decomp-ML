"""Stage b rerun: stage a with the TRACE measures added to the liquidity group.

Reads the measures from LIQUIDITY_PATH (written by scripts.b_liquidity) and writes
reports/b_liquidity_summary.csv, reports/b_group_shares.csv, reports/figures/b_group_shares.png and
reports/b_drop_group.csv. Totals only.

Run from the repo root: python -m scripts.b_rerun_shap
About as long as stage a (8.5 minutes on a laptop).
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.a_shap_split import REPORTS, plot_monthly_shares, shares_table
from src.config import ensure_outside_repo, load_config, load_settings, require
from src.liquidity import join_measures, measure_summary, with_measures
from src.panel import load_panel
from src.shap_split import drop_group_mae, fit_spread_model, group_contributions

logger = logging.getLogger(__name__)


def main() -> None:
    """Write the measure summary, the shares table, the chart and the drop-group check to ``reports/``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = load_config()
    config = with_measures(base)
    path = ensure_outside_repo(require(load_settings().liquidity_path, "LIQUIDITY_PATH"), "liquidity measures")
    df = join_measures(load_panel(["train", "valid"]), pd.read_parquet(path))
    summary = measure_summary(df, base["liquidity"]["measures"], base["panel"]["stratum"])
    summary.to_csv(REPORTS / "b_liquidity_summary.csv", index=False, float_format="%.6g")

    train, valid = df[df["partition"] == "train"], df[df["partition"] == "valid"]
    table = shares_table(group_contributions(fit_spread_model(train, config=config), df, config), df)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "b_group_shares.csv", index=False, float_format="%.2f")
    plot_monthly_shares(table, REPORTS / "figures" / "b_group_shares.png")
    drop_group_mae(train, valid, config).to_csv(REPORTS / "b_drop_group.csv", index=False, float_format="%.2f")
    logger.info("Wrote the measure summary, shares table, chart and drop-group check to %s", REPORTS)


if __name__ == "__main__":
    main()
