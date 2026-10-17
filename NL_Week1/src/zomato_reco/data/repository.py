"""In-memory restaurant repository loaded from the processed parquet.

No Streamlit or Groq imports. The ``st.cache_resource`` wrapper lives in
``ui/state.py`` (Phase 3) so this module stays importable in tests.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from zomato_reco.data.preprocess import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Raised when the processed dataset cannot be loaded or is invalid."""


class Repository:
    """Read-only view over the cleaned restaurant parquet."""

    def __init__(self, df: pd.DataFrame, manifest: dict[str, Any], path: Path) -> None:
        self._df = df
        self._manifest = manifest
        self._path = path
        self._locations = sorted(df["location"].dropna().unique().tolist())
        cuisine_set: set[str] = set()
        for items in df["cuisines"]:
            if isinstance(items, list):
                cuisine_set.update(items)
        self._cuisines = sorted(cuisine_set)

    def all(self) -> pd.DataFrame:
        """Return a copy of the full frame so callers cannot mutate storage."""
        return self._df.copy()

    def locations(self) -> list[str]:
        return list(self._locations)

    def cuisines(self) -> list[str]:
        return list(self._cuisines)

    def by_ids(self, ids: list[str]) -> pd.DataFrame:
        """Return rows for the given ids, each found id once, preserving id order."""
        if not ids:
            return self._df.iloc[0:0].copy()
        seen: set[str] = set()
        ordered: list[str] = []
        for rid in ids:
            if rid not in seen:
                seen.add(rid)
                ordered.append(rid)
        frame = self._df[self._df["id"].isin(ordered)].copy()
        # Preserve request order
        rank = {rid: i for i, rid in enumerate(ordered)}
        frame["_ord"] = frame["id"].map(rank)
        frame = frame.sort_values("_ord", kind="mergesort").drop(columns=["_ord"])
        return frame.reset_index(drop=True)

    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list):
            return [str(v) for v in items if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if items is None or (isinstance(items, float) and math.isnan(items)):
            return []
    return []


def _optional_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if value is pd.NA or bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_rest_type(value: Any) -> str | None:
    if _optional_number(value) is None and not isinstance(value, str):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize list columns and missing numbers; add ``cuisines_key``."""
    out = df.copy()
    out["cuisines"] = out["cuisines"].map(_as_str_list)
    out["dish_liked"] = out["dish_liked"].map(_as_str_list)
    out["cost_for_two"] = out["cost_for_two"].map(_optional_number).astype(object)
    out["rating"] = out["rating"].map(_optional_number).astype(object)
    out["rest_type"] = out["rest_type"].map(_normalize_rest_type)
    out["cuisines_key"] = out["cuisines"].map(
        lambda items: " ".join(items).casefold() if items else ""
    )
    return out


def _check_columns(df: pd.DataFrame) -> None:
    actual = set(df.columns)
    expected = set(CANONICAL_COLUMNS)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise RepositoryError(
            "Processed dataset schema mismatch. " + "; ".join(parts) + ". Run ingest again."
        )


def load_repository(path: Path | str) -> Repository:
    """Load a parquet into memory, close the file, and return a Repository.

    Raises
    ------
    RepositoryError
        Corrupt file, schema mismatch, missing/invalid manifest, or no locations.
    """
    parquet_path = Path(path)
    manifest_path = parquet_path.with_name("manifest.json")

    if not parquet_path.is_file():
        raise RepositoryError(
            f"Processed dataset not found at {parquet_path}. Run scripts/ingest.py."
        )
    if not manifest_path.is_file():
        raise RepositoryError(
            f"Manifest not found at {manifest_path}. Run scripts/ingest.py."
        )

    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RepositoryError(
            "Manifest could not be read. Run scripts/ingest.py again."
        ) from exc

    try:
        # Read fully then release the file handle (important on Windows).
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # noqa: BLE001 - surface as typed error
        logger.exception("parquet_read_failed", extra={"path": str(parquet_path)})
        raise RepositoryError(
            "Data could not be read, run ingest again."
        ) from exc

    _check_columns(df)
    df = _normalize_frame(df)

    locations = df["location"].dropna().unique().tolist()
    if not locations:
        raise RepositoryError(
            "Processed dataset has no locations. Run scripts/ingest.py again."
        )

    return Repository(df=df, manifest=manifest, path=parquet_path)
