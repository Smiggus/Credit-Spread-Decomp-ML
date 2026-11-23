"""Made-up panels for tests. No real data."""

import numpy as np
import pandas as pd

from src.panel import feature_columns

CATEGORIES = {
    "maturity_bucket": ["1-3y", "3-5y"],
    "grade_bucket": ["IG", "HY"],
    "debt_type": ["S-NT"],
    "sector": ["Energy", "Utilities"],
}


def synthetic_panel(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """A panel with the configured columns: one snapshot per row, monthly from Jan 2024.

    Partitions follow the row order: the first half train, then gap, valid and test.
    """
    rng = np.random.default_rng(seed)
    cut = [n // 2, n // 2 + n // 12, n - n // 6]
    partition = np.select(
        [np.arange(n) < cut[0], np.arange(n) < cut[1], np.arange(n) < cut[2]], ["train", "gap", "valid"], "test"
    )
    data = {
        "snapshot_date": pd.date_range("2024-01-31", periods=n, freq="ME"),
        "cusip_id": [f"C{i % 4}" for i in range(n)],
        "issuer_ticker": [f"T{i % 3}" for i in range(n)],
        "partition": partition,
        "spread_bps": rng.uniform(50, 500, n),
    }
    for col in feature_columns():
        if col in CATEGORIES:
            values = CATEGORIES[col]
            data[col] = [values[i % len(values)] for i in range(n)]
        elif col.startswith(("is_", "has_")) or col == "sector_is_pit":
            data[col] = [bool(i % 2) for i in range(n)]
        else:
            data[col] = rng.normal(size=n)
    return pd.DataFrame(data)


def driven_panel(columns: list[str], n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A synthetic panel whose spread depends only on the given numeric columns, plus small noise."""
    df = synthetic_panel(n, seed)
    rng = np.random.default_rng(seed + 1)
    df["spread_bps"] = 200 + sum(40 * df[col] for col in columns) + rng.normal(scale=2, size=n)
    return df
