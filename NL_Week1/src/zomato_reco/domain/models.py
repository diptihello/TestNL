"""Domain models for preferences, restaurants, candidates and results.

Pure pydantic models with no Streamlit or Groq imports. Location membership is
validated by the service, not here, so the models stay repository-independent.
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class Budget(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Preferences(BaseModel):
    """User search preferences. Cuisine and notes are optional."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    location: str
    budget: Budget
    min_rating: float = Field(ge=0.0, le=5.0)
    cuisine: str | None = None
    extra_preferences: str | None = None

    @field_validator("location")
    @classmethod
    def _location_required(cls, value: str) -> str:
        if not value:
            raise ValueError("location must not be empty")
        return value

    @field_validator("cuisine", mode="before")
    @classmethod
    def _normalize_cuisine(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return None
        if len(parts) > 3:
            raise ValueError("cuisine accepts at most 3 comma-separated values")
        for part in parts:
            if len(part) > 40:
                raise ValueError("each cuisine value must be at most 40 characters")
        return ", ".join(parts)

    @field_validator("extra_preferences", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = _CONTROL_CHARS_RE.sub("", str(value)).strip()
        if not text:
            return None
        return text[:300]


class Restaurant(BaseModel):
    """Canonical restaurant record matching the processed parquet schema."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    address: str
    location: str
    location_key: str
    cuisines: list[str] = Field(default_factory=list)
    cost_for_two: int | None = None
    rating: float | None = None
    votes: int = 0
    rest_type: str | None = None
    dish_liked: list[str] = Field(default_factory=list)
    online_order: bool = False
    book_table: bool = False

    @classmethod
    def from_row(cls, row: Any) -> Restaurant:
        """Build a Restaurant from a pandas Series or mapping."""
        data = row if isinstance(row, dict) else dict(row)
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            address=str(data.get("address") or ""),
            location=str(data.get("location") or ""),
            location_key=str(data.get("location_key") or ""),
            cuisines=_as_str_list(data.get("cuisines")),
            cost_for_two=_as_optional_int(data.get("cost_for_two")),
            rating=_as_optional_float(data.get("rating")),
            votes=int(data.get("votes") or 0),
            rest_type=_as_optional_str(data.get("rest_type")),
            dish_liked=_as_str_list(data.get("dish_liked")),
            online_order=bool(data.get("online_order", False)),
            book_table=bool(data.get("book_table", False)),
        )


class Candidate(BaseModel):
    """Compact candidate sent to the LLM, backed by a full Restaurant record."""

    model_config = ConfigDict(extra="forbid")

    restaurant: Restaurant

    @classmethod
    def from_row(cls, row: Any) -> Candidate:
        return cls(restaurant=Restaurant.from_row(row))

    @property
    def id(self) -> str:
        return self.restaurant.id

    def to_prompt_line(self) -> str:
        """Compact JSON line for the prompt (no spaces after separators).

        Name is capped at 80 characters, each dish at 40, and the joined
        cuisines / liked strings at 80 so a line stays under 400 characters.
        """
        r = self.restaurant
        cuisines = ", ".join(r.cuisines)[:80]
        liked = ", ".join(d[:40] for d in r.dish_liked)[:80]
        rest_type = (r.rest_type or "")[:40] or None
        payload = {
            "id": r.id,
            "name": r.name[:80],
            "cuisines": cuisines,
            "rating": r.rating,
            "votes": r.votes,
            "cost_for_two": r.cost_for_two,
            "type": rest_type,
            "online_order": r.online_order,
            "book_table": r.book_table,
            "liked": liked,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    restaurant: Restaurant
    explanation: str | None = None
    matched: list[str] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendations: list[Recommendation]
    extra_recommendations: list[Recommendation] = Field(default_factory=list)
    summary: str | None = None
    total_matches: int
    candidates_shown: int
    source: Literal["llm", "cache", "fallback"]
    usage: dict[str, Any] | None = None
    notice: str | None = None


class Relaxation(BaseModel):
    """A single suggested preference change that would yield matches."""

    model_config = ConfigDict(extra="forbid")

    label: str
    field: str
    value: Any
    would_match: int


class EmptyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    relaxations: list[Relaxation] = Field(default_factory=list)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas.NA and similar
    try:
        import pandas as pd

        if value is pd.NA:
            return True
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
    except (ImportError, TypeError, ValueError):
        pass
    return False


def _as_str_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list):
            return [str(v) for v in items if not _is_missing(v)]
    return []


def _as_optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _as_optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None
