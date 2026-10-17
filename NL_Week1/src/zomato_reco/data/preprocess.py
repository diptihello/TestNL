"""Pure preprocessing functions for the Zomato Bangalore dataset.

Every function is unit-testable without network, Streamlit, or Groq.
No module in ``data/`` may import ``streamlit`` or ``groq``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: source CSV column names -> canonical names
# ---------------------------------------------------------------------------
SOURCE_COLUMNS: list[str] = [
    "name",
    "address",
    "online_order",
    "book_table",
    "rate",
    "votes",
    "location",
    "rest_type",
    "dish_liked",
    "cuisines",
    "approx_cost(for two people)",
    "listed_in(type)",
]

# ``listed_in(type)`` is read only to explain duplicates; it is never output.
OPTIONAL_SOURCE_COLUMNS: frozenset[str] = frozenset({"listed_in(type)"})
REQUIRED_SOURCE_COLUMNS: list[str] = [
    c for c in SOURCE_COLUMNS if c not in OPTIONAL_SOURCE_COLUMNS
]

CANONICAL_COLUMNS: list[str] = [
    "id",
    "name",
    "address",
    "location",
    "location_key",
    "cuisines",
    "cost_for_two",
    "rating",
    "votes",
    "rest_type",
    "dish_liked",
    "online_order",
    "book_table",
]

# Expected pandas dtypes of the cleaned frame (and therefore of the parquet).
CANONICAL_DTYPES: dict[str, str] = {
    "id": "object",
    "name": "object",
    "address": "object",
    "location": "object",
    "location_key": "object",
    "cuisines": "object",  # list[str]
    "cost_for_two": "Int64",  # nullable integer
    "rating": "float64",  # NaN when unrated
    "votes": "int64",
    "rest_type": "object",  # str or None
    "dish_liked": "object",  # list[str]
    "online_order": "bool",
    "book_table": "bool",
}

DISH_LIKED_LIMIT = 5

# Rating pattern: e.g. "4.1/5", "4 /5", "4.1 / 5"
_RATING_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*5\s*$")
# A raw rating value that "looks numeric" contains at least one digit.
_HAS_DIGIT_RE = re.compile(r"\d")
# A run of letters, optionally with one apostrophe part ("chef's").
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NA


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------


def parse_rating(value: Any) -> float | None:
    """Parse a rating string like ``"4.1/5"`` to a float, or *None* for non-numeric values.

    Accepts spacing around the slash.  Returns *None* for ``"NEW"``, ``"-"``,
    empty strings, ``None``, and ``NaN``.  Rejects values outside 0-5.
    """
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NEW" or text == "-":
        return None
    match = _RATING_RE.match(text)
    if not match:
        return None
    rating = round(float(match.group(1)), 1)
    if rating < 0 or rating > 5:
        return None
    return rating


def parse_cost(value: Any) -> int | None:
    """Parse a cost string like ``"1,200"`` to an int, or *None* for non-numeric values.

    Strips commas and spaces.  Handles ``"800.0"`` -> 800.  Rejects <= 0.
    """
    if _is_missing(value):
        return None
    text = str(value).strip().replace(",", "").replace(" ", "")
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    result = int(num)
    if result <= 0:
        return None
    return result


def _title_word(word: str) -> str:
    """Title-case one whitespace token, keeping short acronyms and apostrophes intact."""
    if word.isupper() and len(word) <= 4:
        return word
    return _WORD_RE.sub(lambda m: m.group(0)[:1].upper() + m.group(0)[1:].lower(), word)


def split_list(value: Any, limit: int | None = None) -> list[str]:
    """Split a comma-separated string, strip, deduplicate keeping order, title-case.

    All-caps tokens of 4 characters or fewer (like "BBQ") are kept as-is.
    """
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        cased = " ".join(_title_word(word) for word in item.split())
        key = cased.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cased)
    if limit is not None:
        result = result[:limit]
    return result


def parse_yes_no(value: Any) -> bool:
    """Return *True* only for case-insensitive ``"yes"``; everything else is *False*."""
    if _is_missing(value):
        return False
    return str(value).strip().lower() == "yes"


def parse_votes(value: Any) -> int:
    """Parse votes to int.  Null or non-numeric become 0."""
    if _is_missing(value):
        return 0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    try:
        num = float(text)
    except ValueError:
        return 0
    if math.isnan(num) or math.isinf(num):
        return 0
    return max(0, int(num))


def normalize_key(text: Any) -> str:
    """Strip and casefold text for use as a dedupe key."""
    if _is_missing(text):
        return ""
    return str(text).strip().casefold()


def make_id(name: str, address: str) -> str:
    """Deterministic restaurant id from normalized name and address.

    ``"r_" + sha1(f"{normalize_key(name)}|{normalize_key(address)}").hexdigest()[:10]``
    """
    combined = f"{normalize_key(name)}|{normalize_key(address)}"
    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:10]
    return f"r_{digest}"


# ---------------------------------------------------------------------------
# Parse-rate helpers (computed on raw, pre-dedupe values)
# ---------------------------------------------------------------------------


def rating_parse_stats(raw_rate: pd.Series) -> tuple[int, int]:
    """Return ``(numeric_looking, parsed)`` counts for a raw ``rate`` column.

    A value is numeric-looking when it contains a digit.  ``"NEW"``, ``"-"``
    and nulls are legitimately unrated and do not count against the rate.
    """
    text = raw_rate.fillna("").astype(str)
    numeric_looking = text.str.contains(_HAS_DIGIT_RE, regex=True)
    parsed = text[numeric_looking].map(parse_rating).notna()
    return int(numeric_looking.sum()), int(parsed.sum())


def cost_parse_stats(raw_cost: pd.Series) -> tuple[int, int]:
    """Return ``(non_empty, parsed)`` counts for a raw cost column."""
    text = raw_cost.fillna("").astype(str).str.strip()
    non_empty = text != ""
    parsed = text[non_empty].map(parse_cost).notna()
    return int(non_empty.sum()), int(parsed.sum())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 1.0


# ---------------------------------------------------------------------------
# DataFrame-level cleaning
# ---------------------------------------------------------------------------


def _pick_display_spelling(df: pd.DataFrame) -> pd.Series:
    """Map each ``location_key`` to its most frequent spelling, alphabetical tiebreak."""
    counts = (
        df.groupby(["location_key", "location"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["location_key", "n", "location"], ascending=[True, False, True])
    )
    best = counts.drop_duplicates("location_key", keep="first")
    return best.set_index("location_key")["location"]


def clean_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules column-wise, deduplicate, and return the 13 canonical columns.

    Parameters
    ----------
    raw : DataFrame
        Raw dataframe with the source columns (read from CSV with ``dtype=str``).

    Returns
    -------
    DataFrame
        Cleaned and deduplicated with exactly the :data:`CANONICAL_COLUMNS` in
        that order and with :data:`CANONICAL_DTYPES`.  ``result.attrs["stats"]``
        holds row counts and parse statistics used by :func:`build_manifest`.

    Raises
    ------
    KeyError
        If a required source column is missing.
    ValueError
        If two different restaurants produce the same id.
    """
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"Required source columns missing: {missing}")

    rows_in = len(raw)
    rating_numeric, rating_parsed = rating_parse_stats(raw["rate"])
    cost_present, cost_parsed = cost_parse_stats(raw["approx_cost(for two people)"])

    df = raw.copy()

    # -- basic string cleaning -----------------------------------------------
    for col in ("name", "address", "location"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    # -- drop rows with empty name or empty location -------------------------
    empty_name = df["name"] == ""
    empty_location = df["location"] == ""
    keep = ~(empty_name | empty_location)
    dropped_empty_name = int(empty_name.sum())
    dropped_empty_location = int((empty_location & ~empty_name).sum())
    df = df[keep].copy()

    # -- parse columns -------------------------------------------------------
    df["rating"] = df["rate"].map(parse_rating).astype("float64")
    df["cost_for_two"] = pd.array(
        df["approx_cost(for two people)"].map(parse_cost).tolist(), dtype="Int64"
    )
    df["cuisines"] = df["cuisines"].map(split_list)
    df["dish_liked"] = df["dish_liked"].map(lambda v: split_list(v, limit=DISH_LIKED_LIMIT))
    df["votes"] = df["votes"].map(parse_votes).astype("int64")
    df["online_order"] = df["online_order"].map(parse_yes_no).astype(bool)
    df["book_table"] = df["book_table"].map(parse_yes_no).astype(bool)

    rest_type = df["rest_type"].fillna("").astype(str).str.strip()
    df["rest_type"] = rest_type.astype(object).where(rest_type != "", None)

    df["location_key"] = df["location"].map(normalize_key)

    # -- deduplicate on normalized (name, address) ---------------------------
    # Stable sort: the row with the most votes comes first; ties keep file order.
    df["_dedupe_name"] = df["name"].map(normalize_key)
    df["_dedupe_addr"] = df["address"].map(normalize_key)
    rows_before_dedupe = len(df)
    df = df.sort_values(by="votes", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset=["_dedupe_name", "_dedupe_addr"], keep="first")
    duplicates_removed = rows_before_dedupe - len(df)

    # -- location display name -----------------------------------------------
    df["location"] = df["location_key"].map(_pick_display_spelling(df))

    # -- assign id and check uniqueness --------------------------------------
    df["id"] = [make_id(n, a) for n, a in zip(df["name"], df["address"])]
    dup_ids = df[df["id"].duplicated(keep=False)]
    if not dup_ids.empty:
        collisions = dup_ids.groupby("id")["name"].apply(list).to_dict()
        details = "; ".join(f"id={k}: {' vs '.join(v)}" for k, v in collisions.items())
        raise ValueError(f"ID collision detected: {details}")

    # -- select canonical columns in a deterministic order --------------------
    df = df.sort_values(["location_key", "_dedupe_name", "id"], kind="mergesort")
    out = df[CANONICAL_COLUMNS].reset_index(drop=True)

    stats = {
        "rows_in": rows_in,
        "rows_out": len(out),
        "dropped_empty_name": dropped_empty_name,
        "dropped_empty_location": dropped_empty_location,
        "duplicates_removed": int(duplicates_removed),
        "rating_numeric_looking": rating_numeric,
        "rating_parsed": rating_parsed,
        "cost_present": cost_present,
        "cost_parsed": cost_parsed,
    }
    out.attrs["stats"] = stats
    logger.info("clean_frame_done", extra=stats)
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    raw_rows: int,
    clean: pd.DataFrame,
    source: dict[str, str],
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Build the ingestion manifest with statistics and provenance.

    Parameters
    ----------
    raw_rows : int
        Number of rows given to :func:`clean_frame`.
    clean : DataFrame
        The cleaned DataFrame (output of :func:`clean_frame`).  Its
        ``attrs["stats"]`` supplies drop counts and raw parse rates.
    source : dict
        Must include ``repo``, ``revision``, ``filename``.
    sample_size : int or None
        If the ingest was run with ``--sample``, the number of rows sampled.

    Returns
    -------
    dict
        Manifest data to be written as JSON (all values JSON-serialisable).
    """
    stats: dict[str, int] = dict(clean.attrs.get("stats", {}))
    rows_after = len(clean)

    rating_numeric = stats.get("rating_numeric_looking", int(clean["rating"].notna().sum()))
    rating_parsed = stats.get("rating_parsed", int(clean["rating"].notna().sum()))
    cost_present = stats.get("cost_present", rows_after)
    cost_parsed = stats.get("cost_parsed", int(clean["cost_for_two"].notna().sum()))

    cost_values = clean["cost_for_two"].dropna().astype("int64")
    cost_percentiles: dict[str, int] = {}
    if len(cost_values) > 0:
        for pct in (10, 33, 50, 66, 90):
            cost_percentiles[f"p{pct}"] = int(round(float(cost_values.quantile(pct / 100))))

    manifest: dict[str, Any] = {
        "rows_before": int(raw_rows),
        "rows_after": rows_after,
        "dropped_rows": int(raw_rows) - rows_after,
        "dropped": {
            "empty_name": stats.get("dropped_empty_name", 0),
            "empty_location": stats.get("dropped_empty_location", 0),
            "duplicates": stats.get("duplicates_removed", 0),
        },
        "rating_numeric_looking": rating_numeric,
        "rating_count": rating_parsed,
        "rating_parse_rate": _ratio(rating_parsed, rating_numeric),
        "rated_restaurants": int(clean["rating"].notna().sum()),
        "cost_present": cost_present,
        "cost_count": cost_parsed,
        "cost_parse_rate": _ratio(cost_parsed, cost_present),
        "cost_percentiles": cost_percentiles,
        "distinct_locations": int(clean["location_key"].nunique()),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_repo": source.get("repo", ""),
        "dataset_revision": source.get("revision", ""),
        "dataset_filename": source.get("filename", ""),
        "schema_columns": list(CANONICAL_COLUMNS),
    }
    if sample_size is not None:
        manifest["sample_size"] = int(sample_size)

    return manifest
