"""Tests for src.config."""

from pathlib import Path

import pytest

from src.config import REPO_ROOT, ensure_outside_repo, load_settings, require

ENV_NAMES = ("PANEL_PATH", "SPARK_MASTER", "SPARK_REMOTE", "GRAPHFRAMES_PACKAGE", "TRACE_DIR", "LIQUIDITY_PATH")


def test_defaults_without_env(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.spark_master == "local[*]"
    assert settings.spark_remote is None
    assert settings.panel_path is None
    assert settings.trace_dir is None
    assert settings.liquidity_path is None


def test_trace_dir_stays_a_url_and_liquidity_path_is_a_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", "s3a://bucket/raw")
    monkeypatch.setenv("LIQUIDITY_PATH", "~/data/measures.parquet")
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.trace_dir == "s3a://bucket/raw"
    assert settings.liquidity_path == Path("~/data/measures.parquet").expanduser()


def test_blank_variable_counts_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARK_REMOTE", "   ")
    assert load_settings(env_file=tmp_path / "missing.env").spark_remote is None


def test_path_inside_repo_is_refused():
    with pytest.raises(ValueError, match="panel file is inside the repo"):
        ensure_outside_repo(REPO_ROOT / "data" / "panel.parquet", "panel file")


def test_relative_path_inside_repo_is_refused(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    with pytest.raises(ValueError):
        ensure_outside_repo(Path("data/panel.parquet"), "panel file")


def test_path_outside_repo_is_kept():
    outside = REPO_ROOT.parent / "outside-the-repo" / "panel.parquet"
    assert ensure_outside_repo(outside, "panel file") == outside


def test_require_names_missing_variable():
    with pytest.raises(RuntimeError, match="PANEL_PATH"):
        require(None, "PANEL_PATH")
    assert require(Path("x"), "PANEL_PATH") == Path("x")
