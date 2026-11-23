from pathlib import Path

import pytest

from csd.config import REPO_ROOT, load_settings, require


def test_defaults_without_env(monkeypatch, tmp_path):
    for name in ("DATA_DIR", "PANEL_PATH", "SPARK_MASTER", "SPARK_REMOTE", "BBG_CDS_PATH"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_file=tmp_path / "missing.env")
    assert settings.spark_master == "local[*]"
    assert settings.spark_remote is None
    assert settings.data_dir is None


def test_data_dir_inside_repo_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(REPO_ROOT / "data"))
    with pytest.raises(ValueError):
        load_settings(env_file=tmp_path / "missing.env")


def test_require_names_missing_variable():
    with pytest.raises(RuntimeError, match="DATA_DIR"):
        require(None, "DATA_DIR")
    assert require(Path("x"), "DATA_DIR") == Path("x")
