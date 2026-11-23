"""Settings and paths.

Two sources, kept apart:
- ``config.yaml`` holds project settings that are safe to commit.
- ``.env`` holds machine-specific paths and connection details, never committed.

Licensed data lives outside the repo. Every data path comes from the environment, so no
data location is written in the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Settings:
    """Machine-specific settings read from the environment."""

    data_dir: Path | None
    panel_path: Path | None
    xbrl_facts_path: Path | None
    bbg_cds_path: Path | None
    wrds_username: str | None
    sec_user_agent: str | None
    spark_master: str
    spark_remote: str | None
    graphframes_package: str | None


def _path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the committed project settings from ``config.yaml``."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(env_file: Path | None = None) -> Settings:
    """Read machine-specific settings, loading ``.env`` from the repo root first."""
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)
    settings = Settings(
        data_dir=_path("DATA_DIR"),
        panel_path=_path("PANEL_PATH"),
        xbrl_facts_path=_path("XBRL_FACTS_PATH"),
        bbg_cds_path=_path("BBG_CDS_PATH"),
        wrds_username=_text("WRDS_USERNAME"),
        sec_user_agent=_text("SEC_USER_AGENT"),
        spark_master=_text("SPARK_MASTER") or "local[*]",
        spark_remote=_text("SPARK_REMOTE"),
        graphframes_package=_text("GRAPHFRAMES_PACKAGE"),
    )
    _check_outside_repo(settings)
    return settings


def _check_outside_repo(settings: Settings) -> None:
    """Refuse data paths inside the repo, so licensed data cannot be committed by mistake."""
    for name in ("data_dir", "panel_path", "xbrl_facts_path", "bbg_cds_path"):
        path = getattr(settings, name)
        if path is not None and path.resolve().is_relative_to(REPO_ROOT):
            raise ValueError(f"{name} points inside the repo. Keep licensed data outside it.")


def require(path: Path | None, name: str) -> Path:
    """Return a configured path, or fail with a message naming the missing variable."""
    if path is None:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return path
