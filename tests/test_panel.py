"""Tests for src.panel, on a small synthetic panel (no real data)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.panel import feature_columns, file_sha256, group_of, input_groups, load_panel, panel_summary

CATEGORIES = {
    "maturity_bucket": ["1-3y", "3-5y"],
    "grade_bucket": ["IG", "HY"],
    "debt_type": ["S-NT"],
    "sector": ["Energy", "Utilities"],
}


def synthetic_panel(n: int = 12) -> pd.DataFrame:
    """A made-up panel with the configured columns."""
    rng = np.random.default_rng(0)
    data = {
        "snapshot_date": pd.date_range("2024-01-31", periods=n, freq="ME"),
        "cusip_id": [f"C{i % 4}" for i in range(n)],
        "issuer_ticker": [f"T{i % 3}" for i in range(n)],
        "partition": (["train"] * 6 + ["gap"] + ["valid"] * 3 + ["test"] * 2)[:n],
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


def write_panel(tmp_path, df, sha=None):
    path = tmp_path / "panel.parquet"
    df.to_parquet(path)
    manifest = {"sha256": sha or file_sha256(path), "rows": len(df)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_input_groups_cover_33_inputs_once():
    cols = feature_columns()
    assert len(cols) == 33
    assert len(set(cols)) == 33
    assert set(input_groups()) == {"bond_terms", "liquidity", "company", "stock_and_balance_sheet", "market"}
    assert group_of()["naive_dd"] == "stock_and_balance_sheet"


def test_load_panel_keeps_default_partitions_and_closes_test(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    df = load_panel(path=path)
    assert set(df["partition"]) == {"train", "gap", "valid"}
    assert "test" not in set(df["partition"])
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])


def test_load_panel_rejects_a_changed_file(tmp_path):
    path = write_panel(tmp_path, synthetic_panel(), sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        load_panel(path=path)


def test_load_panel_rejects_unknown_partition(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    with pytest.raises(ValueError, match="Unknown partitions requested"):
        load_panel(partitions=["holdout"], path=path)


def test_load_panel_needs_the_manifest(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_panel(path=path)


def test_load_panel_opens_test_only_when_asked(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    df = load_panel(partitions=["test"], path=path)
    assert set(df["partition"]) == {"test"}


def test_panel_summary_is_totals_only():
    summary = panel_summary(synthetic_panel())
    assert summary["rows"] == 12
    assert summary["bonds"] == 4
    assert summary["companies"] == 3
    assert sum(summary["rows_by_partition"].values()) == 12
    assert set(summary["rows_by_stratum"]) == {"IG", "HY"}


def test_config_target_is_spread():
    assert load_config()["panel"]["target"] == "spread_bps"
