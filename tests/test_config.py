"""Tests for src.config."""

from pathlib import Path

import pytest

from src.config import REPO_ROOT, load_config, load_settings, require

ENV_NAMES = ("PANEL_PATH", "SPARK_MASTER", "SPARK_REMOTE", "GRAPHFRAMES_PACKAGE")


def test_defaults_without_env(monkeypatch, tmp_path):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.spark_master == "local[*]"
    assert settings.spark_remote is None
    assert settings.panel_path is None


def test_blank_variable_counts_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARK_REMOTE", "   ")
    assert load_settings(env_file=tmp_path / "missing.env").spark_remote is None


def test_panel_path_inside_repo_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("PANEL_PATH", str(REPO_ROOT / "data" / "panel.parquet"))
    with pytest.raises(ValueError, match="PANEL_PATH"):
        load_settings(env_file=tmp_path / "missing.env")


def test_panel_path_outside_repo_is_kept(monkeypatch, tmp_path):
    # pytest's tmp_path sits inside the repo (pytest.ini basetemp), so use a sibling folder.
    outside = REPO_ROOT.parent / "outside-the-repo" / "panel.parquet"
    monkeypatch.setenv("PANEL_PATH", str(outside))
    assert load_settings(env_file=tmp_path / "missing.env").panel_path == outside


def test_require_names_missing_variable():
    with pytest.raises(RuntimeError, match="PANEL_PATH"):
        require(None, "PANEL_PATH")
    assert require(Path("x"), "PANEL_PATH") == Path("x")


def test_committed_config_has_spark_settings():
    config = load_config()
    assert config["spark"]["app_name"]
    assert config["spark"]["timezone"] == "UTC"
