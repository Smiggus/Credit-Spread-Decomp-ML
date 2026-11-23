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

from src.config import ensure_outside_repo, load_config, load_settings, require

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


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(parquet: Path, manifest_name: str) -> None:
    """Raise unless the parquet file exists and matches the sha256 in the manifest beside it."""
    if not parquet.exists():
        raise FileNotFoundError(f"Panel file not found: {parquet}")
    manifest_path = parquet.with_name(manifest_name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_name} not found in {parquet.parent}")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"].lower()
    actual = file_sha256(parquet)
    if actual != expected:
        raise ValueError(f"Panel file does not match its manifest (sha256 {actual[:12]}... vs {expected[:12]}...).")


def load_panel(
    partitions: str | Iterable[str] | None = None,
    path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read the panel's ids, target and inputs for the requested partitions.

    partitions: one name or several; default from ``config.yaml`` (train, gap and valid).
        The test months are read only when asked for.
    path: default ``PANEL_PATH`` from ``.env``. It must lie outside the repo.
    The file is checked against its manifest first. A missing column fails in the read.
    """
    cfg = panel_config(config)
    if partitions is None:
        partitions = cfg["default_partitions"]
    parts = [partitions] if isinstance(partitions, str) else list(partitions)
    unknown = set(parts) - set(ALL_PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown partitions requested: {sorted(unknown)}")

    parquet = ensure_outside_repo(path or require(load_settings().panel_path, "PANEL_PATH"), "panel file")
    check_manifest(parquet, cfg["manifest_name"])

    columns = list(cfg["id_columns"]) + [cfg["target"]] + feature_columns(config)
    df = pd.read_parquet(parquet, columns=columns, filters=[("partition", "in", parts)])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


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
