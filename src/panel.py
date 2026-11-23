"""Load the ML II bond-month panel and name its input groups.

The panel is licensed data. It is read from ``PANEL_PATH`` (outside the repo) and checked
against the ``manifest.json`` beside it, so every run uses the same file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import load_config, load_settings, require

ALL_PARTITIONS = ("train", "gap", "valid", "test")


def panel_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``panel`` section of ``config.yaml``."""
    return (config or load_config())["panel"]


def input_groups(config: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Input group name to its columns, in config order."""
    return {name: list(cols) for name, cols in panel_config(config)["input_groups"].items()}


def feature_columns(config: dict[str, Any] | None = None) -> list[str]:
    """All model inputs, group by group."""
    return [col for cols in input_groups(config).values() for col in cols]


def group_of(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Column to the name of its input group."""
    return {col: name for name, cols in input_groups(config).items() for col in cols}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(parquet: Path, manifest_name: str) -> dict[str, Any]:
    """Check the parquet file against its manifest and return the manifest.

    Raises FileNotFoundError if the manifest is missing and ValueError if the checksum differs.
    """
    manifest_path = parquet.with_name(manifest_name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_name} not found next to the panel file.")
    info = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = file_sha256(parquet)
    if actual != info["sha256"]:
        raise ValueError(
            f"Panel file does not match its manifest (sha256 {actual[:12]}... vs {info['sha256'][:12]}...)."
        )
    return info


def validate_panel(df: pd.DataFrame, config: dict[str, Any] | None = None) -> None:
    """Fail if a required column is missing or a partition is unknown."""
    cfg = panel_config(config)
    required = list(cfg["id_columns"]) + [cfg["target"]] + feature_columns(config)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing columns: {missing}")
    unknown = set(df["partition"].dropna().unique()) - set(ALL_PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown partitions in the panel: {sorted(unknown)}")


def load_panel(
    partitions: Iterable[str] | None = None,
    path: Path | None = None,
    config: dict[str, Any] | None = None,
    check: bool = True,
) -> pd.DataFrame:
    """Read the panel, keep the requested partitions and return ids, target and inputs.

    partitions: default from ``config.yaml`` (train, gap and valid; test stays closed).
    path: default ``PANEL_PATH`` from ``.env``.
    check: verify the file against its manifest first.
    """
    cfg = panel_config(config)
    parts = list(partitions) if partitions is not None else list(cfg["default_partitions"])
    bad = set(parts) - set(ALL_PARTITIONS)
    if bad:
        raise ValueError(f"Unknown partitions requested: {sorted(bad)}")

    parquet = path or require(load_settings().panel_path, "PANEL_PATH")
    if check:
        check_manifest(parquet, cfg["manifest_name"])

    columns = list(cfg["id_columns"]) + [cfg["target"]] + feature_columns(config)
    df = pd.read_parquet(parquet, columns=columns)
    validate_panel(df, config)
    df = df[df["partition"].isin(parts)].copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.reset_index(drop=True)


def panel_summary(df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Totals only: rows, bonds, companies, date range and rows by partition and stratum."""
    cfg = panel_config(config)
    return {
        "rows": int(len(df)),
        "bonds": int(df["cusip_id"].nunique()),
        "companies": int(df["issuer_ticker"].nunique()),
        "first_snapshot": str(df["snapshot_date"].min().date()) if len(df) else None,
        "last_snapshot": str(df["snapshot_date"].max().date()) if len(df) else None,
        "rows_by_partition": {k: int(v) for k, v in df["partition"].value_counts().sort_index().items()},
        "rows_by_stratum": {k: int(v) for k, v in df[cfg["stratum"]].value_counts().sort_index().items()},
    }
