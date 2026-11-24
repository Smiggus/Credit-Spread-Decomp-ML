"""Stage c, #14: Merton spreads for the panel's train and valid months.

Writes reports/c_merton.csv (each model spread against the observed spread, by grade and maturity)
and reports/c_merton_coverage.csv (bond-months by outcome). Totals only.

Run from the repo root: python -m scripts.c_merton
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import REPO_ROOT
from src.merton import RATE_AT_MATURITY, merton_summary, panel_spreads
from src.panel import load_panel

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports"


def main() -> None:
    """Solve the panel's company-months, price each bond, and write the two tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    panel = load_panel(["train", "valid"], extra_columns=[RATE_AT_MATURITY])
    spreads, counts = panel_spreads(panel)
    merton_summary(spreads, {"merton": "merton_spread_bps"}).to_csv(REPORTS / "c_merton.csv", index=False)
    coverage = pd.Series(counts, name="bond_months").rename_axis("outcome").reset_index()
    coverage.to_csv(REPORTS / "c_merton_coverage.csv", index=False)
    logger.info("Merton spreads for %d bond-months: %s", len(spreads), counts)


if __name__ == "__main__":
    main()
