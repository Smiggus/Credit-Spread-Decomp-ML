"""Settings and paths.

Two sources, kept apart:
- ``config.yaml`` holds project settings that are safe to commit.
- ``.env`` holds machine-specific paths and connection details, never committed.

Licensed data lives outside the repo. Every data path comes from the environment, so no
data location is written in the code. Add a setting here when a stage first needs it.
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

    panel_path: Path | None
    spark_master: str
    spark_remote: str | None
    graphframes_package: str | None


def _env(name: str) -> str | None:
    """An environment variable, or None when it is unset or blank."""
    return os.getenv(name, "").strip() or None


def load_config() -> dict[str, Any]:
    """Read the committed project settings from ``config.yaml``."""
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_settings(env_file: Path | None = None) -> Settings:
    """Read machine-specific settings, loading ``.env`` from the repo root first."""
    load_dotenv(env_file or REPO_ROOT / ".env", override=False)
    panel = _env("PANEL_PATH")
    return Settings(
        panel_path=Path(panel).expanduser() if panel else None,
        spark_master=_env("SPARK_MASTER") or "local[*]",
        spark_remote=_env("SPARK_REMOTE"),
        graphframes_package=_env("GRAPHFRAMES_PACKAGE"),
    )


def ensure_outside_repo(path: Path, what: str) -> Path:
    """Return ``path`` if it lies outside the repo; raise ValueError if it lies inside.

    Every reader of licensed data calls this on the path it is about to read, so the data
    cannot end up in a commit.
    """
    if path.resolve().is_relative_to(REPO_ROOT):
        raise ValueError(f"The {what} is inside the repo. Keep licensed data outside it.")
    return path


def require(path: Path | None, name: str) -> Path:
    """Return a configured path, or fail with a message naming the missing variable."""
    if path is None:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return path
