"""Tests for src.panel, on a small synthetic panel (no real data)."""

import json

import pandas as pd
import pytest

import src.config
from src.panel import feature_columns, file_sha256, input_groups, load_panel, panel_summary
from tests.synthetic import synthetic_panel


@pytest.fixture(autouse=True)
def repo_elsewhere(monkeypatch, tmp_path):
    """pytest's tmp_path sits inside the repo, so point the repo guard at a folder beside it."""
    monkeypatch.setattr(src.config, "REPO_ROOT", tmp_path / "repo")


def write_panel(folder, df, sha=None):
    """Write a panel and its manifest; return the panel's path."""
    path = folder / "panel.parquet"
    df.to_parquet(path)
    manifest = {"sha256": sha or file_sha256(path), "rows": len(df)}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_input_groups_cover_33_inputs_once():
    cols = feature_columns()
    assert len(cols) == len(set(cols)) == 33
    assert set(input_groups()) == {"bond_terms", "liquidity", "company", "stock_and_balance_sheet", "market"}


def test_default_load_keeps_test_months_closed(tmp_path):
    df = load_panel(path=write_panel(tmp_path, synthetic_panel()))
    assert set(df["partition"]) == {"train", "gap", "valid"}
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])


def test_test_months_open_only_when_asked(tmp_path):
    df = load_panel(partitions="test", path=write_panel(tmp_path, synthetic_panel()))
    assert set(df["partition"]) == {"test"}


def test_unknown_partition_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown partitions requested"):
        load_panel(partitions=["holdout"], path=write_panel(tmp_path, synthetic_panel()))


def test_changed_file_is_refused(tmp_path):
    path = write_panel(tmp_path, synthetic_panel(), sha="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        load_panel(path=path)


def test_upper_case_checksum_matches(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": file_sha256(path).upper()}), encoding="utf-8")
    assert len(load_panel(path=path)) == 10


def test_missing_manifest_names_the_folder(tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json not found in"):
        load_panel(path=path)


def test_missing_panel_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="Panel file not found"):
        load_panel(path=tmp_path / "nowhere.parquet")


def test_extra_columns_are_read_only_when_asked(tmp_path):
    path = write_panel(tmp_path, synthetic_panel().assign(ust_yield_at_maturity_pct=4.0))
    assert "ust_yield_at_maturity_pct" not in load_panel(path=path)
    df = load_panel(path=path, extra_columns=["ust_yield_at_maturity_pct"])
    assert (df["ust_yield_at_maturity_pct"] == 4.0).all()


def test_missing_column_fails_the_read(tmp_path):
    path = write_panel(tmp_path, synthetic_panel().drop(columns=["naive_dd"]))
    with pytest.raises(Exception, match="naive_dd"):
        load_panel(path=path)


def test_panel_inside_repo_is_refused(monkeypatch, tmp_path):
    path = write_panel(tmp_path, synthetic_panel())
    monkeypatch.setattr(src.config, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="inside the repo"):
        load_panel(path=path)


def test_panel_summary_is_totals_only():
    summary = panel_summary(synthetic_panel())
    assert (summary["rows"], summary["bonds"], summary["companies"]) == (12, 4, 3)
    assert sum(summary["rows_by_partition"].values()) == 12
    assert set(summary["rows_by_stratum"]) == {"IG", "HY"}
