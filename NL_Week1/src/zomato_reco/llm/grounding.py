"""Post-parse checks for numerical claims and restaurant names."""

from __future__ import annotations

import re
from collections.abc import Iterable

from zomato_reco.domain.models import Candidate, Restaurant

_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_RATING_PATTERNS = (
    re.compile(rf"\b({_NUMBER})\s*/\s*5\b", re.IGNORECASE),
    re.compile(rf"\b(?:rating|rated(?:\s+at)?)\s+(?:of\s+)?({_NUMBER})", re.IGNORECASE),
    re.compile(rf"\b({_NUMBER})\s*(?:stars?|rating)\b", re.IGNORECASE),
)
_COST_PATTERNS = (
    re.compile(rf"(?:₹|INR|\$)\s*({_NUMBER})", re.IGNORECASE),
    re.compile(rf"\b(?:cost|price)\s+(?:of\s+)?(?:INR\s*)?({_NUMBER})", re.IGNORECASE),
)
_VOTE_PATTERN = re.compile(rf"\b({_NUMBER})\s+votes?\b", re.IGNORECASE)
_SUMMARY_LEADS = frozenset(
    {"these", "those", "this", "that", "our", "try", "visit", "consider", "choose", "explore"}
)


def escape_text(text: str | None) -> str:
    """Escape Markdown control characters for plain-text UI display."""
    if text is None:
        return ""
    special = set("\\`*_{}[]()#+-.!$")
    return "".join("\\" + char if char in special else char for char in str(text))


def _number(value: str) -> float:
    return float(value.replace(",", ""))


def _matches_numbers(
    text: str, patterns: tuple[re.Pattern[str], ...], expected: float | int | None
) -> bool:
    matches = [match.group(1) for pattern in patterns for match in pattern.finditer(text)]
    if not matches:
        return True
    if expected is None:
        return False
    return all(_number(value) == float(expected) for value in matches)


def check_explanation(text: str, candidate: Candidate | Restaurant) -> bool:
    """Return false when a stated rating, cost or vote count is ungrounded."""
    restaurant = candidate.restaurant if isinstance(candidate, Candidate) else candidate
    return (
        _matches_numbers(text, _RATING_PATTERNS, restaurant.rating)
        and _matches_numbers(text, _COST_PATTERNS, restaurant.cost_for_two)
        and _matches_numbers(text, (_VOTE_PATTERN,), restaurant.votes)
    )


def neutral_explanation(candidate: Candidate | Restaurant) -> str:
    """Build a safe replacement using only the restaurant's known facts."""
    restaurant = candidate.restaurant if isinstance(candidate, Candidate) else candidate
    cuisine = ", ".join(restaurant.cuisines[:2])
    if cuisine and restaurant.rating is not None:
        return f"{restaurant.name} serves {cuisine} and has a {restaurant.rating:.1f} rating."
    if cuisine:
        return f"{restaurant.name} serves {cuisine}."
    if restaurant.rating is not None:
        return f"{restaurant.name} has a {restaurant.rating:.1f} rating."
    return f"{restaurant.name} is an option in {restaurant.location}."


def check_summary(text: str, candidates: Iterable[Candidate | Restaurant]) -> bool:
    """Reject capitalized name-like phrases not present among candidates."""
    names = [
        item.restaurant.name if isinstance(item, Candidate) else item.name
        for item in candidates
    ]
    allowed = {name.casefold() for name in names}
    pattern = re.compile(r"\b[A-Z][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)*")
    for match in pattern.finditer(text):
        phrase = match.group()
        normalized = phrase.strip(" .,:;!?")
        words = normalized.split()
        if len(words) > 1 and words[0].casefold() in _SUMMARY_LEADS:
            normalized = " ".join(words[1:])
        if not normalized or normalized.casefold() in allowed:
            continue
        prefix = text[: match.start()].rstrip()
        if (
            (not prefix or prefix.endswith((".", "!", "?")))
            and len(normalized.split()) == 1
            and "'" not in normalized
            and "’" not in normalized
            and normalized.casefold() in _SUMMARY_LEADS
        ):
            continue
        if normalized.casefold() in {"I", "INR"}:
            continue
        looks_like_name = " " in normalized or "'" in normalized or "’" in normalized
        if looks_like_name and not any(
            name.casefold() in normalized.casefold() for name in names
        ):
            return False
    return True
