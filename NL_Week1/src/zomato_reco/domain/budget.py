"""Budget tier mapping from approximate cost for two (INR)."""

from __future__ import annotations

from zomato_reco.config import Settings
from zomato_reco.domain.models import Budget


def tier_for(cost: int | None, settings: Settings) -> Budget | None:
    """Map a cost-for-two value to a budget tier, or None when cost is missing."""
    if cost is None:
        return None
    if cost <= settings.budget_low_max:
        return Budget.low
    if cost <= settings.budget_medium_max:
        return Budget.medium
    return Budget.high


def tier_range_label(budget: Budget, settings: Settings) -> str:
    """Human-readable INR range for a budget tier (used in prompts and UI)."""
    low = settings.budget_low_max
    medium = settings.budget_medium_max
    if budget is Budget.low:
        return f"INR up to {low} for two"
    if budget is Budget.medium:
        return f"INR {low + 1}-{medium} for two"
    return f"INR above {medium} for two"
