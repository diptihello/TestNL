"""Strict parsing and normalization of model-generated recommendation JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from zomato_reco.config import Settings
from zomato_reco.llm.client import LLMResponseError


class LLMRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(strict=True)
    rank: int
    explanation: str = Field(strict=True)
    matched: list[str] = Field(default_factory=list)

    @field_validator("rank", mode="before")
    @classmethod
    def _numeric_text_rank(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return value

    @field_validator("matched", mode="before")
    @classmethod
    def _safe_matched(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError("matched must be a list of strings")


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str | None = Field(default=None, strict=True)
    recommendations: list[LLMRecommendation]
    extra_recommendations: list[LLMRecommendation] = Field(default_factory=list)


@dataclass(frozen=True)
class ParseStats:
    dropped_unknown_ids: int = 0
    dropped_duplicates: int = 0
    dropped_invalid: int = 0
    dropped_empty_explanations: int = 0
    extra_recommendations: int = 0


def _extract_json(raw: str) -> str:
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise LLMResponseError("The AI response did not contain a JSON object.")

    in_string = False
    escaped = False
    depth = 0
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0 or text[:start].strip() or text[end:].strip():
        raise LLMResponseError("Unexpected text around the AI JSON response.")
    return text[start:end]


def _truncate_explanation(text: str) -> str:
    clean = text.strip()
    if len(clean) <= 300:
        return clean
    prefix = clean[:299]
    if " " in prefix:
        prefix = prefix.rsplit(None, 1)[0]
    return prefix.rstrip() + "…"


def _truncate_summary(text: str | None) -> str | None:
    if not text or not text.strip():
        return None
    clean = text.strip()
    words = clean.split()
    if len(words) <= 60 and len(clean) <= 400:
        return clean
    prefix = " ".join(words[:60])[:399]
    boundary = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
    if boundary >= int(len(prefix) * 0.55):
        prefix = prefix[: boundary + 1]
    elif " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip() + "…"


def parse_output(
    raw_content: str,
    candidate_ids: set[str] | list[str] | tuple[str, ...],
    settings: Settings,
) -> tuple[LLMOutput, ParseStats]:
    """Parse, validate and filter the output; preserve extra valid recs for UI."""
    if not raw_content.strip():
        raise LLMResponseError("The AI response was empty.")
    try:
        payload = json.loads(_extract_json(raw_content))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LLMResponseError("The AI response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError("The AI response must be a JSON object.")
    raw_recommendations = payload.get("recommendations")
    if not isinstance(raw_recommendations, list) or not raw_recommendations:
        raise LLMResponseError("The AI response contained no recommendations.")

    valid_ids = set(candidate_ids)
    accepted: list[LLMRecommendation] = []
    seen: set[str] = set()
    unknown = duplicates = invalid = empty_explanations = 0
    for item in raw_recommendations:
        if not isinstance(item, dict):
            invalid += 1
            continue
        try:
            recommendation = LLMRecommendation.model_validate(item)
        except ValidationError:
            invalid += 1
            continue
        recommendation.id = recommendation.id.strip()
        if recommendation.id not in valid_ids:
            unknown += 1
            continue
        if recommendation.id in seen:
            duplicates += 1
            continue
        explanation = _truncate_explanation(recommendation.explanation)
        if not explanation:
            empty_explanations += 1
            continue
        recommendation.explanation = explanation
        recommendation.rank = len(accepted) + 1
        seen.add(recommendation.id)
        accepted.append(recommendation)

    if not accepted:
        raise LLMResponseError("The AI response had no valid recommendations.")
    cap = settings.max_recommendations
    summary = _truncate_summary(payload.get("summary") if isinstance(payload.get("summary"), str) else None)
    result = LLMOutput(
        summary=summary,
        recommendations=accepted[:cap],
        extra_recommendations=accepted[cap:],
    )
    return result, ParseStats(
        dropped_unknown_ids=unknown,
        dropped_duplicates=duplicates,
        dropped_invalid=invalid,
        dropped_empty_explanations=empty_explanations,
        extra_recommendations=len(accepted[cap:]),
    )
