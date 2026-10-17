"""Structured filtering and relaxation hints. Pure pandas, no LLM."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from zomato_reco.config import Settings
from zomato_reco.domain.models import Budget, EmptyResult, Preferences, Relaxation

STEP_LOCATION = "location"
STEP_RATING = "rating"
STEP_BUDGET = "budget"
STEP_CUISINE = "cuisine"

_WORD_BOUNDARY = r"(?<![\w]){pat}(?![\w])"


def _cuisine_tokens(cuisine: str | None) -> list[str]:
    if not cuisine:
        return []
    return [p.strip() for p in cuisine.split(",") if p.strip()][:3]


def _cuisine_mask(df: pd.DataFrame, cuisine: str | None) -> pd.Series:
    """Whole-word, case-insensitive match against the ``cuisines_key`` column."""
    tokens = _cuisine_tokens(cuisine)
    if not tokens:
        return pd.Series(True, index=df.index)

    if "cuisines_key" not in df.columns:
        keys = df["cuisines"].map(_join_cuisines_key)
    else:
        keys = df["cuisines_key"].fillna("")

    mask = pd.Series(False, index=df.index)
    for token in tokens:
        pattern = _WORD_BOUNDARY.format(pat=re.escape(token.casefold()))
        mask = mask | keys.str.contains(pattern, regex=True, na=False)
    return mask


def _join_cuisines_key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).casefold()
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list):
            return " ".join(str(v) for v in items).casefold()
    return str(value).casefold()


def _rating_series(df: pd.DataFrame) -> pd.Series:
    """Numeric rating series (NaN for missing), safe for object-dtype columns."""
    return pd.to_numeric(df["rating"], errors="coerce")


def _cost_series(df: pd.DataFrame) -> pd.Series:
    """Numeric cost series (NaN for missing), safe for object-dtype columns."""
    return pd.to_numeric(df["cost_for_two"], errors="coerce")


def _rating_mask(df: pd.DataFrame, min_rating: float) -> pd.Series:
    if min_rating <= 0:
        return pd.Series(True, index=df.index)
    ratings = _rating_series(df)
    rated = ratings.notna()
    return rated & (ratings.round(1) >= float(min_rating))


def _budget_mask(df: pd.DataFrame, budget: Budget, settings: Settings) -> pd.Series:
    costs = _cost_series(df)
    low_max = settings.budget_low_max
    med_max = settings.budget_medium_max
    has_cost = costs.notna()
    if budget is Budget.low:
        return has_cost & (costs <= low_max)
    if budget is Budget.medium:
        return has_cost & (costs > low_max) & (costs <= med_max)
    return has_cost & (costs > med_max)


def _location_mask(df: pd.DataFrame, location: str) -> pd.Series:
    key = location.strip().casefold()
    return df["location_key"] == key


def apply(
    df: pd.DataFrame,
    prefs: Preferences,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the four filter masks in order and return the frame plus step counts.

    Counts keys: ``after_location``, ``after_rating``, ``after_budget``,
    ``after_cuisine``, and ``unrated_hidden`` (unrated rows in the location
    when ``min_rating > 0``, else 0).
    """
    loc = _location_mask(df, prefs.location)
    after_location = df[loc]
    unrated_hidden = 0
    if prefs.min_rating > 0 and not after_location.empty:
        unrated_hidden = int(_rating_series(after_location).isna().sum())

    rated = after_location[_rating_mask(after_location, prefs.min_rating)]
    budgeted = rated[_budget_mask(rated, prefs.budget, settings)]
    if prefs.cuisine:
        filtered = budgeted[_cuisine_mask(budgeted, prefs.cuisine)]
    else:
        filtered = budgeted

    counts = {
        "after_location": int(len(after_location)),
        "after_rating": int(len(rated)),
        "after_budget": int(len(budgeted)),
        "after_cuisine": int(len(filtered)),
        "unrated_hidden": unrated_hidden,
    }
    return filtered.copy(), counts


def _count_with(
    located: pd.DataFrame,
    *,
    min_rating: float,
    budget: Budget,
    cuisine: str | None,
    settings: Settings,
) -> int:
    if located.empty:
        return 0
    step = located[_rating_mask(located, min_rating)]
    step = step[_budget_mask(step, budget, settings)]
    if cuisine:
        step = step[_cuisine_mask(step, cuisine)]
    return int(len(step))


def suggest(
    df: pd.DataFrame,
    prefs: Preferences,
    settings: Settings,
) -> list[Relaxation]:
    """Suggest single-field relaxations that would produce at least one match.

    Location stays fixed. Tries: lower rating by 0.5 steps down to 3.0, each
    other budget tier, and drop cuisine. Returns at most 4, smallest change
    first. Never mutates ``prefs``.
    """
    located = df[_location_mask(df, prefs.location)]
    if located.empty:
        return []

    candidates: list[tuple[int, int, Relaxation]] = []
    # sort key: (priority group, magnitude, Relaxation)
    # rating changes first (smallest drop first), then budget, then drop cuisine

    if prefs.min_rating > 0:
        rating = prefs.min_rating
        step_index = 0
        while True:
            next_rating = round(rating - 0.5, 1)
            if next_rating < 3.0:
                break
            n = _count_with(
                located,
                min_rating=next_rating,
                budget=prefs.budget,
                cuisine=prefs.cuisine,
                settings=settings,
            )
            if n >= 1:
                candidates.append(
                    (
                        0,
                        step_index,
                        Relaxation(
                            label=f"Lower minimum rating to {next_rating:.1f}",
                            field="min_rating",
                            value=next_rating,
                            would_match=n,
                        ),
                    )
                )
            rating = next_rating
            step_index += 1

    for other in (Budget.low, Budget.medium, Budget.high):
        if other is prefs.budget:
            continue
        n = _count_with(
            located,
            min_rating=prefs.min_rating,
            budget=other,
            cuisine=prefs.cuisine,
            settings=settings,
        )
        if n >= 1:
            candidates.append(
                (
                    1,
                    0,
                    Relaxation(
                        label=f"Try budget {other.value}",
                        field="budget",
                        value=other.value,
                        would_match=n,
                    ),
                )
            )

    if prefs.cuisine:
        n = _count_with(
            located,
            min_rating=prefs.min_rating,
            budget=prefs.budget,
            cuisine=None,
            settings=settings,
        )
        if n >= 1:
            candidates.append(
                (
                    2,
                    0,
                    Relaxation(
                        label="Drop cuisine filter",
                        field="cuisine",
                        value=None,
                        would_match=n,
                    ),
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], -item[2].would_match))
    return [item[2] for item in candidates[:4]]


def closest_single_change(
    df: pd.DataFrame,
    prefs: Preferences,
    settings: Settings,
) -> Relaxation | None:
    """Best single-field change by match count, even when the count is zero."""
    located = df[_location_mask(df, prefs.location)]
    if located.empty:
        return None

    options: list[Relaxation] = []

    if prefs.min_rating > 0:
        rating = prefs.min_rating
        while True:
            next_rating = round(rating - 0.5, 1)
            if next_rating < 3.0:
                break
            n = _count_with(
                located,
                min_rating=next_rating,
                budget=prefs.budget,
                cuisine=prefs.cuisine,
                settings=settings,
            )
            options.append(
                Relaxation(
                    label=f"Lower minimum rating to {next_rating:.1f}",
                    field="min_rating",
                    value=next_rating,
                    would_match=n,
                )
            )
            rating = next_rating

    for other in (Budget.low, Budget.medium, Budget.high):
        if other is prefs.budget:
            continue
        n = _count_with(
            located,
            min_rating=prefs.min_rating,
            budget=other,
            cuisine=prefs.cuisine,
            settings=settings,
        )
        options.append(
            Relaxation(
                label=f"Try budget {other.value}",
                field="budget",
                value=other.value,
                would_match=n,
            )
        )

    if prefs.cuisine:
        n = _count_with(
            located,
            min_rating=prefs.min_rating,
            budget=prefs.budget,
            cuisine=None,
            settings=settings,
        )
        options.append(
            Relaxation(
                label="Drop cuisine filter",
                field="cuisine",
                value=None,
                would_match=n,
            )
        )

    if not options:
        return None
    return max(options, key=lambda r: r.would_match)


def failing_step(counts: dict[str, int]) -> str:
    """Name the filter step where the count first reached zero."""
    if counts.get("after_location", 0) == 0:
        return STEP_LOCATION
    if counts.get("after_rating", 0) == 0:
        return STEP_RATING
    if counts.get("after_budget", 0) == 0:
        return STEP_BUDGET
    if counts.get("after_cuisine", 0) == 0:
        return STEP_CUISINE
    return STEP_CUISINE


def build_empty_result(
    df: pd.DataFrame,
    prefs: Preferences,
    settings: Settings,
    counts: dict[str, int],
) -> EmptyResult:
    """Build an EmptyResult with a reason and optional relaxation hints."""
    relaxations = suggest(df, prefs, settings)
    step = failing_step(counts)

    if counts.get("after_location", 0) == 0:
        reason = f"No restaurants found in {prefs.location}."
        return EmptyResult(reason=reason, relaxations=[])

    if relaxations:
        step_labels = {
            STEP_RATING: f"rating {prefs.min_rating:.1f}+",
            STEP_BUDGET: f"budget {prefs.budget.value}",
            STEP_CUISINE: f"cuisine {prefs.cuisine}",
        }
        detail = step_labels.get(step, step)
        reason = (
            f"No restaurants in {prefs.location} match {detail}. "
            f"Try one of the suggested changes."
        )
        return EmptyResult(reason=reason, relaxations=relaxations)

    closest = closest_single_change(df, prefs, settings)
    if closest is not None:
        reason = (
            f"No restaurants match this combination in {prefs.location}. "
            f"Closest single change: {closest.label} "
            f"({closest.would_match} match{'es' if closest.would_match != 1 else ''})."
        )
    else:
        reason = f"No restaurants match this combination in {prefs.location}."
    return EmptyResult(reason=reason, relaxations=[])


__all__ = [
    "apply",
    "suggest",
    "closest_single_change",
    "build_empty_result",
    "failing_step",
]
