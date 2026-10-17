"""Candidate selection, vote-aware ranking and deterministic fallback."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from zomato_reco.config import Settings
from zomato_reco.data.preprocess import normalize_key
from zomato_reco.domain.models import Candidate, Preferences, Recommendation

# Keyword groups: whole-word match in extra_preferences -> column to boost
_ONLINE_KEYWORDS = ("online", "delivery")
_BOOK_KEYWORDS = ("book", "reserve", "table")
_NEGATIONS = frozenset({"no", "not", "without", "don't", "dont"})

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text)]


def _keyword_active(tokens: list[str], keywords: tuple[str, ...]) -> bool:
    """True when a keyword appears as a whole word without a nearby negation."""
    for i, token in enumerate(tokens):
        if token not in keywords:
            continue
        window = tokens[max(0, i - 3) : i]
        if any(w in _NEGATIONS for w in window):
            continue
        return True
    return False


def _boost_flags(notes: str | None) -> tuple[bool, bool]:
    """Return (want_online, want_book) from free-text notes."""
    if not notes:
        return False, False
    tokens = _tokenize(notes)
    return (
        _keyword_active(tokens, _ONLINE_KEYWORDS),
        _keyword_active(tokens, _BOOK_KEYWORDS),
    )


def _vote_aware_score(
    rating: pd.Series,
    votes: pd.Series,
    mean_rating: float,
    m: int,
) -> pd.Series:
    """Bayesian average: (v/(v+m))*rating + (m/(v+m))*mean_rating."""
    v = votes.astype("float64")
    r = rating.astype("float64")
    return (v / (v + m)) * r + (m / (v + m)) * mean_rating


def _apply_chain_cap(df: pd.DataFrame, max_per_chain: int) -> pd.DataFrame:
    """Keep at most ``max_per_chain`` rows per normalized restaurant name."""
    if df.empty or max_per_chain <= 0:
        return df
    keys = df["name"].map(normalize_key)
    keep_idx: list[Any] = []
    counts: dict[str, int] = {}
    for idx, key in zip(df.index, keys):
        n = counts.get(key, 0)
        if n < max_per_chain:
            keep_idx.append(idx)
            counts[key] = n + 1
    return df.loc[keep_idx]


def select(
    filtered: pd.DataFrame,
    prefs: Preferences,
    settings: Settings,
) -> tuple[list[Candidate], int]:
    """Rank filtered rows, apply chain cap and candidate ceiling.

    Returns ``(candidates, total_matches)`` where ``total_matches`` is the size
    of ``filtered`` before capping.
    """
    total = int(len(filtered))
    if total == 0:
        return [], 0

    df = filtered.copy()
    want_online, want_book = _boost_flags(prefs.extra_preferences)

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    votes = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    rated_mask = ratings.notna()
    mean_rating = float(ratings.loc[rated_mask].mean()) if rated_mask.any() else 0.0
    m = int(settings.vote_weight)

    scores = pd.Series(float("-inf"), index=df.index, dtype="float64")
    if rated_mask.any():
        scores.loc[rated_mask] = _vote_aware_score(
            ratings.loc[rated_mask],
            votes.loc[rated_mask],
            mean_rating,
            m,
        )

    boost = pd.Series(0, index=df.index, dtype="int64")
    if want_online:
        boost = boost + df["online_order"].astype(bool).astype("int64")
    if want_book:
        boost = boost + df["book_table"].astype(bool).astype("int64")

    df = df.assign(_score=scores, _boost=boost, _unrated=~rated_mask)
    df = df.sort_values(
        by=["_boost", "_unrated", "_score", "votes", "name", "id"],
        ascending=[False, True, False, False, True, True],
        kind="mergesort",
    )

    df = _apply_chain_cap(df, settings.max_per_chain)
    df = df.head(settings.max_candidates)

    candidates = [Candidate.from_row(row) for _, row in df.iterrows()]
    return candidates, total


def fallback_recommendations(
    candidates: list[Candidate],
    settings: Settings,
) -> list[Recommendation]:
    """Top ``MAX_RECOMMENDATIONS`` candidates in deterministic order, no explanations."""
    limit = min(settings.max_recommendations, len(candidates))
    return [
        Recommendation(
            rank=i + 1,
            restaurant=candidates[i].restaurant,
            explanation=None,
            matched=[],
        )
        for i in range(limit)
    ]
