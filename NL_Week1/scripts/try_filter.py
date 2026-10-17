"""Smoke run of Phase 2 filtering and ranking over the real parquet.

Usage (from project root, with the package installed)::

    python scripts/try_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zomato_reco.config import get_settings
from zomato_reco.data.repository import load_repository
from zomato_reco.domain.filters import apply, build_empty_result
from zomato_reco.domain.models import Budget, Preferences
from zomato_reco.domain.ranking import select


def main() -> int:
    settings = get_settings()
    if not settings.dataset_ready:
        print(f"Dataset not ready at {settings.processed_path}. Run scripts/ingest.py.")
        return 1

    repo = load_repository(settings.processed_path)
    df = repo.all()
    print(f"Loaded {len(df)} restaurants, {len(repo.locations())} locations.")

    sample = Preferences(
        location=repo.locations()[0],
        budget=Budget.medium,
        min_rating=4.0,
        cuisine=None,
        extra_preferences=None,
    )
    filtered, counts = apply(df, sample, settings)
    print(f"\nSample query: {sample.model_dump(mode='json')}")
    print(f"Step counts: {counts}")
    candidates, total = select(filtered, sample, settings)
    print(f"Candidates: {len(candidates)} of {total} matches")
    for i, cand in enumerate(candidates, 1):
        r = cand.restaurant
        print(f"  {i:2d}. {r.name}  rating={r.rating} votes={r.votes} cost={r.cost_for_two}")

    # Strict but relaxable: high rating + rare cuisine usually empties, while
    # dropping cuisine or lowering the rating yields matches.
    impossible = Preferences(
        location=repo.locations()[0],
        budget=Budget.medium,
        min_rating=4.5,
        cuisine="ThisCuisineDoesNotExistXYZ",
    )
    empty_filtered, empty_counts = apply(df, impossible, settings)
    print(f"\nImpossible query: {impossible.model_dump(mode='json')}")
    print(f"Impossible query step counts: {empty_counts}")
    assert empty_filtered.empty
    empty = build_empty_result(df, impossible, settings, empty_counts)
    print(f"EmptyResult reason: {empty.reason}")
    if empty.relaxations:
        for hint in empty.relaxations:
            print(f"  hint: {hint.label} -> {hint.would_match} matches")
    else:
        print("  (no single-change hints)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
