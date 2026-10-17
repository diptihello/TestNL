"""Download and preprocess the Zomato Bangalore dataset into a parquet + manifest.

Usage (from the project root)::

    python scripts/ingest.py
    python scripts/ingest.py --sample 2000     # quick local run
    python scripts/ingest.py --revision abc123  # pin a commit

The script is idempotent: the same input produces identical output and ids.
It never leaves a half-written file at the final path, and it writes nothing
when a quality gate fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, repo_info

# Ensure the src/ directory is importable even without pip install -e .
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zomato_reco.config import ConfigError, load_settings  # noqa: E402
from zomato_reco.data.preprocess import (  # noqa: E402
    REQUIRED_SOURCE_COLUMNS,
    SOURCE_COLUMNS,
    build_manifest,
    clean_frame,
)

logger = logging.getLogger(__name__)

# Quality gates: catch a changed upstream schema early.
MIN_RATING_PARSE_RATE = 0.95  # of raw values that look numeric
MIN_LOCATIONS = 50

# Fixed seed so a --sample run is reproducible.
SAMPLE_SEED = 42

# Retry settings for Windows file replacement
_RENAME_RETRIES = 5
_RENAME_DELAY_SECONDS = 0.5

PARQUET_TMP_SUFFIX = ".tmp"
MANIFEST_NAME = "manifest.json"


class IngestError(Exception):
    """A failure that should stop ingestion with a clear message and exit code 1."""


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _ascii(text: object) -> str:
    """Make text safe for any Windows console code page."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _say(text: object = "", *, err: bool = False) -> None:
    print(_ascii(text), file=sys.stderr if err else sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# Source discovery and reading
# ---------------------------------------------------------------------------


def discover_csv(repo_id: str, revision: str = "main") -> tuple[str, str]:
    """Find the single CSV file in the dataset repo and resolve the commit hash.

    Returns ``(filename, commit_sha)``.  Raises :class:`IngestError` listing the
    files found when there is not exactly one CSV candidate.
    """
    files = sorted(list_repo_files(repo_id, repo_type="dataset", revision=revision))
    csv_files = [f for f in files if Path(f).suffix.lower() == ".csv"]
    if not csv_files:
        raise IngestError(f"No CSV files found in {repo_id}@{revision}. Files present: {files}")
    if len(csv_files) > 1:
        raise IngestError(
            f"Expected exactly one CSV file in {repo_id}@{revision}, "
            f"found {len(csv_files)}: {csv_files}"
        )
    info = repo_info(repo_id, repo_type="dataset", revision=revision)
    return csv_files[0], info.sha or revision


def download_csv(repo_id: str, filename: str, revision: str) -> Path:
    """Download the CSV file (cached by huggingface_hub between runs)."""
    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision
        )
    )


def read_raw(csv_path: Path) -> pd.DataFrame:
    """Read only the needed columns as strings. Never loads ``reviews_list``/``menu_item``."""
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:  # pandas raises several types for a bad file
        raise IngestError(f"Cannot read CSV {csv_path.name}: {exc}") from exc

    available = set(header.columns)
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in available]
    if missing:
        raise IngestError(
            f"Required columns missing from CSV: {missing}. "
            f"Available columns: {sorted(available)}"
        )
    use_cols = [c for c in SOURCE_COLUMNS if c in available]
    return pd.read_csv(csv_path, usecols=use_cols, dtype=str, keep_default_na=True)


def take_sample(raw: pd.DataFrame, n: int) -> pd.DataFrame:
    """Reproducible random sample that keeps the original row order.

    A random (not head) sample is used because the source file is grouped by
    area, so its first N rows would cover only a handful of locations.
    """
    if n >= len(raw):
        return raw
    return raw.sample(n=n, random_state=SAMPLE_SEED).sort_index()


# ---------------------------------------------------------------------------
# Gates and atomic output
# ---------------------------------------------------------------------------


def check_gates(
    manifest: dict[str, Any],
    *,
    min_rating_rate: float = MIN_RATING_PARSE_RATE,
    min_locations: int = MIN_LOCATIONS,
) -> None:
    """Raise :class:`IngestError` if the cleaned data looks like a changed schema."""
    rate = manifest["rating_parse_rate"]
    if rate < min_rating_rate:
        raise IngestError(
            f"Rating parse rate {rate:.1%} is below the {min_rating_rate:.0%} gate "
            f"({manifest['rating_count']} of {manifest['rating_numeric_looking']} "
            "numeric-looking values parsed). The upstream format may have changed."
        )
    locations = manifest["distinct_locations"]
    if locations < min_locations:
        raise IngestError(
            f"Only {locations} distinct locations survived (need >= {min_locations}). "
            "The upstream data may have changed."
        )


def _safe_rename(src: Path, dst: Path) -> None:
    """Rename *src* to *dst*, retrying because Windows cannot replace an open file."""
    for attempt in range(1, _RENAME_RETRIES + 1):
        try:
            src.replace(dst)
            return
        except OSError:
            if attempt == _RENAME_RETRIES:
                raise
            logger.warning("rename_retry", extra={"attempt": attempt, "dst": dst.name})
            time.sleep(_RENAME_DELAY_SECONDS)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("tmp_cleanup_failed", extra={"path": path.name})


def write_outputs(clean: pd.DataFrame, manifest: dict[str, Any], parquet_path: Path) -> Path:
    """Write parquet and manifest to ``.tmp`` names, then rename into place, manifest last.

    The existing manifest is removed before the parquet is replaced, so a crash
    between the two renames leaves the app in the "not ready" state instead of
    pairing a new parquet with an old manifest.  Temporary files are removed on
    any failure.  Returns the manifest path.
    """
    out_dir = parquet_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    tmp_parquet = parquet_path.with_name(parquet_path.name + PARQUET_TMP_SUFFIX)
    tmp_manifest = manifest_path.with_name(manifest_path.name + PARQUET_TMP_SUFFIX)

    try:
        frame = clean.copy()
        frame.attrs = {}  # keep parquet metadata small and stable
        frame.to_parquet(tmp_parquet, index=False, engine="pyarrow")
        tmp_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if manifest_path.exists():
            _unlink_quietly(manifest_path)
        _safe_rename(tmp_parquet, parquet_path)
        _safe_rename(tmp_manifest, manifest_path)
    finally:
        _unlink_quietly(tmp_parquet)
        _unlink_quietly(tmp_manifest)
    return manifest_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    raw: pd.DataFrame,
    parquet_path: Path,
    source: dict[str, str],
    *,
    sample_size: int | None = None,
    min_rating_rate: float = MIN_RATING_PARSE_RATE,
    min_locations: int = MIN_LOCATIONS,
) -> dict[str, Any]:
    """Clean, gate and write. Nothing is written if any gate fails."""
    if sample_size is not None:
        raw = take_sample(raw, sample_size)
    clean = clean_frame(raw)
    manifest = build_manifest(
        raw_rows=len(raw), clean=clean, source=source, sample_size=sample_size
    )
    check_gates(manifest, min_rating_rate=min_rating_rate, min_locations=min_locations)
    write_outputs(clean, manifest, parquet_path)
    return manifest


def format_summary(manifest: dict[str, Any], parquet_path: Path) -> str:
    """Build an ASCII-safe summary table for the console."""
    dropped = manifest.get("dropped", {})
    lines = [
        "",
        "=" * 52,
        "  Ingestion complete",
        "=" * 52,
        f"  Rows (raw):            {manifest['rows_before']:>10,}",
        f"  Rows (clean):          {manifest['rows_after']:>10,}",
        f"    dropped empty name:  {dropped.get('empty_name', 0):>10,}",
        f"    dropped empty loc.:  {dropped.get('empty_location', 0):>10,}",
        f"    duplicates removed:  {dropped.get('duplicates', 0):>10,}",
        f"  Rating parse rate:     {manifest['rating_parse_rate']:>10.1%}",
        f"  Cost parse rate:       {manifest['cost_parse_rate']:>10.1%}",
        f"  Rated restaurants:     {manifest['rated_restaurants']:>10,}",
        f"  Distinct locations:    {manifest['distinct_locations']:>10,}",
        "",
    ]
    pct = manifest.get("cost_percentiles") or {}
    if pct:
        lines.append("  Cost for two (INR) percentiles:")
        lines.extend(f"    {key}: {value:>8,}" for key, value in pct.items())
        lines.append("")
    if manifest.get("sample_size") is not None:
        lines.extend([f"  ** SAMPLE RUN: {manifest['sample_size']:,} rows **", ""])
    lines.extend(
        [
            f"  Source:  {manifest['dataset_repo']} / {manifest['dataset_filename']}",
            f"  Commit:  {manifest['dataset_revision']}",
            f"  Output:  {parquet_path}",
            f"  Built:   {manifest['generated_at']}",
            "=" * 52,
        ]
    )
    return _ascii("\n".join(lines))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess the Zomato Bangalore dataset."
    )
    parser.add_argument("--repo", default=None, help="Dataset repo id (default: DATASET_REPO).")
    parser.add_argument(
        "--revision", default=None, help="Branch, tag or commit (default: DATASET_REVISION)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder for the parquet and manifest (default: folder of PROCESSED_PATH).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use a reproducible random sample of N raw rows (quick local runs).",
    )
    args = parser.parse_args(argv)
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive number of rows")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion pipeline. Returns 0 on success, 1 on failure."""
    args = _parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        _say(f"ERROR: {exc}", err=True)
        return 1

    repo_id = args.repo or settings.dataset_repo
    revision = args.revision or settings.dataset_revision
    parquet_name = settings.processed_path.name
    out_dir = Path(args.out).resolve() if args.out else settings.processed_path.parent
    parquet_path = out_dir / parquet_name

    # A sample cannot be expected to cover every neighbourhood.
    min_locations = MIN_LOCATIONS
    if args.sample is not None:
        min_locations = 1
        _say("Sample mode: the 50-location gate is relaxed; the app will show a sample badge.")

    try:
        _say(f"Discovering CSV in {repo_id}@{revision} ...")
        filename, commit = discover_csv(repo_id, revision)
        _say(f"  Found {filename} at commit {commit[:12]}")

        _say("Downloading (cached by huggingface_hub after the first run) ...")
        csv_path = download_csv(repo_id, filename, commit)

        _say("Reading the needed columns ...")
        raw = read_raw(csv_path)
        _say(f"  Read {len(raw):,} rows.")

        _say("Cleaning, de-duplicating and writing ...")
        manifest = run_pipeline(
            raw,
            parquet_path,
            {"repo": repo_id, "revision": commit, "filename": filename},
            sample_size=args.sample,
            min_locations=min_locations,
        )
    except IngestError as exc:
        _say(f"ERROR: {exc}", err=True)
        _say("Nothing was written.", err=True)
        return 1
    except Exception as exc:  # network, disk, unexpected data
        _say(f"ERROR: ingestion failed: {type(exc).__name__}: {exc}", err=True)
        _say("Nothing was written to the final paths.", err=True)
        return 1

    _say(format_summary(manifest, parquet_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
