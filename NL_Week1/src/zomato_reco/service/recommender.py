"""Recommendation orchestration with grounded LLM output and safe fallback."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from zomato_reco.config import Settings
from zomato_reco.data.repository import Repository
from zomato_reco.domain import filters
from zomato_reco.domain.models import (
    Candidate,
    EmptyResult,
    Preferences,
    Recommendation,
    RecommendationResult,
)
from zomato_reco.domain.ranking import fallback_recommendations, select
from zomato_reco.llm.client import (
    LLMClient,
    LLMError,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    RawLLMResponse,
    make_client,
)
from zomato_reco.llm.grounding import (
    check_explanation,
    check_summary,
    neutral_explanation,
)
from zomato_reco.llm.guard import (
    BudgetExceeded,
    BudgetGuard,
    ResultCache,
    UsageStore,
)
from zomato_reco.llm.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
    estimate_tokens,
)
from zomato_reco.llm.schema import LLMOutput, parse_output

logger = logging.getLogger(__name__)

_NO_KEY = "AI explanations unavailable: no API key configured."
_PROVIDER_FAILURE = "The AI service did not respond; showing results sorted by rating."
_RESPONSE_FAILURE = "The AI response could not be read; showing results sorted by rating."
_GENERIC_FAILURE = "AI recommendations are temporarily unavailable; showing results sorted by rating."


class LocationNotFoundError(ValueError):
    """Raised when a preference location is absent from the repository."""


class SearchOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prefs: Preferences
    unrated_hidden: int
    value: RecommendationResult | EmptyResult
    retryable: bool = False
    cached_age_seconds: float | None = None


@dataclass
class _SearchStats:
    counts: dict[str, int]
    candidates_sent: int = 0
    tokens: int = 0
    latency_ms: int = 0
    source: str = "fallback"
    error_class: str | None = None


class RecommendationService:
    """Filter and rank source data, then optionally add grounded AI text."""

    def __init__(
        self,
        repository: Repository,
        settings: Settings,
        *,
        client: LLMClient | None = None,
        guard: BudgetGuard | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.client = client or make_client(settings)
        self.guard = guard or BudgetGuard(UsageStore(settings.usage_path), settings)
        self.cache = cache or ResultCache(settings.cache_dir)
        self.call_count = 0

    def _facts(self, ids: list[str]) -> dict[str, Candidate]:
        rows = self.repository.by_ids(ids)
        return {
            str(row["id"]): Candidate.from_row(row)
            for _, row in rows.iterrows()
        }

    def _fallback(
        self,
        candidates: list[Candidate],
        total_matches: int,
        notice: str,
    ) -> RecommendationResult:
        source_facts = self._facts([candidate.id for candidate in candidates])
        recs = fallback_recommendations(candidates, self.settings)
        grounded = [
            rec.model_copy(update={"restaurant": source_facts[rec.restaurant.id]})
            for rec in recs
            if rec.restaurant.id in source_facts
        ]
        return RecommendationResult(
            recommendations=grounded,
            total_matches=total_matches,
            candidates_shown=len(grounded),
            source="fallback",
            notice=notice,
        )

    def _from_llm(
        self,
        output: LLMOutput,
        candidates: list[Candidate],
        total_matches: int,
        source: Literal["llm", "cache"],
    ) -> RecommendationResult:
        candidate_map = {candidate.id: candidate for candidate in candidates}
        facts = self._facts([item.id for item in output.recommendations + output.extra_recommendations])
        primary: list[Recommendation] = []
        extras: list[Recommendation] = []
        for item in output.recommendations + output.extra_recommendations:
            candidate = candidate_map.get(item.id)
            source_candidate = facts.get(item.id)
            if candidate is None or source_candidate is None:
                continue
            explanation = (
                item.explanation
                if check_explanation(item.explanation, source_candidate)
                else neutral_explanation(source_candidate)
            )
            recommendation = Recommendation(
                rank=len(primary) + len(extras) + 1,
                restaurant=source_candidate.restaurant,
                explanation=explanation,
                matched=item.matched,
            )
            (primary if len(primary) < self.settings.max_recommendations else extras).append(
                recommendation
            )
        if not primary:
            raise LLMResponseError("The AI response had no recommendations in the dataset.")
        summary = output.summary
        if summary and not check_summary(summary, candidates):
            summary = None
        primary_ids = {rec.restaurant.id for rec in primary}
        # Cached entries are checked against this request's candidates before assembly.
        partial = len(primary_ids) < min(self.settings.max_recommendations, len(candidates))
        notice = (
            f"AI explanations available for {len(primary_ids)} of "
            f"{min(self.settings.max_recommendations, len(candidates))}."
            if partial
            else None
        )
        return RecommendationResult(
            recommendations=primary,
            extra_recommendations=extras,
            summary=summary,
            total_matches=total_matches,
            candidates_shown=len(primary),
            source=source,
            notice=notice,
        )

    def _cache_key(self, prefs: Preferences, candidates: list[Candidate]) -> str:
        return self.cache.key(prefs, candidates, self.settings.groq_model, PROMPT_VERSION)

    def _cached_output(
        self, prefs: Preferences, candidates: list[Candidate]
    ) -> tuple[LLMOutput, float] | None:
        if self.settings.demo_mode:
            return None
        found = self.cache.get_with_age(self._cache_key(prefs, candidates))
        if found is None:
            return None
        output, age = found
        eligible = {candidate.id for candidate in candidates}
        known = self._facts(
            [item.id for item in output.recommendations + output.extra_recommendations]
        )
        retained = [
            item
            for item in output.recommendations + output.extra_recommendations
            if item.id in eligible and item.id in known
        ]
        if not retained:
            return None
        cap = self.settings.max_recommendations
        validated = LLMOutput(
            summary=output.summary,
            recommendations=retained[:cap],
            extra_recommendations=retained[cap:],
        )
        return validated, age

    def recommend(self, prefs: Preferences, *, bypass_cache: bool = False) -> SearchOutcome:
        self.call_count += 1
        all_rows = self.repository.all()
        locations = {str(value).casefold() for value in self.repository.locations()}
        if prefs.location.casefold() not in locations:
            raise LocationNotFoundError(
                f"{prefs.location} is not in the dataset. Choose a listed location."
            )

        filtered, counts = filters.apply(all_rows, prefs, self.settings)
        stats = _SearchStats(counts=counts)
        try:
            if filtered.empty:
                stats.source = "empty"
                return SearchOutcome(
                    prefs=prefs,
                    unrated_hidden=counts["unrated_hidden"],
                    value=filters.build_empty_result(all_rows, prefs, self.settings, counts),
                )
            candidates, total_matches = select(filtered, prefs, self.settings)
            stats.candidates_sent = len(candidates)
            if not candidates:
                result = self._fallback([], total_matches, _GENERIC_FAILURE)
                return SearchOutcome(prefs=prefs, unrated_hidden=counts["unrated_hidden"], value=result)

            if not self.settings.demo_mode and not self.settings.groq_api_key.get_secret_value():
                stats.error_class = "MissingAPIKey"
                result = self._fallback(candidates, total_matches, _NO_KEY)
                return SearchOutcome(prefs=prefs, unrated_hidden=counts["unrated_hidden"], value=result)

            if not bypass_cache:
                cached = self._cached_output(prefs, candidates)
                if cached is not None:
                    output, age = cached
                    result = self._from_llm(output, candidates, total_matches, "cache")
                    stats.source = "cache"
                    return SearchOutcome(
                        prefs=prefs,
                        unrated_hidden=counts["unrated_hidden"],
                        value=result,
                        cached_age_seconds=age,
                    )

            user_prompt = build_user_prompt(prefs, candidates, self.settings)
            estimated = estimate_tokens(SYSTEM_PROMPT + "\n" + user_prompt)
            exceeded = self.guard.check_and_reserve(
                estimated, self.settings.llm_max_output_tokens
            )
            if exceeded:
                stats.error_class = "BudgetExceeded"
                result = self._fallback(
                    candidates,
                    total_matches,
                    f"Rate limit reached. Try again in {exceeded.wait_seconds} seconds.",
                )
                return SearchOutcome(
                    prefs=prefs,
                    unrated_hidden=counts["unrated_hidden"],
                    value=result,
                )

            reserved = True
            try:
                raw: RawLLMResponse = self.client.rank(
                    SYSTEM_PROMPT, user_prompt, self.settings.llm_max_output_tokens
                )
                stats.tokens = raw.total_tokens
                stats.latency_ms = raw.latency_ms
                self.guard.record(raw.total_tokens)
                reserved = False
                output, parse_stats = parse_output(
                    raw.content, {candidate.id for candidate in candidates}, self.settings
                )
                result = self._from_llm(output, candidates, total_matches, "llm")
                if parse_stats.dropped_unknown_ids:
                    logger.info("llm_unknown_ids_dropped", extra={"count": parse_stats.dropped_unknown_ids})
                if not self.settings.demo_mode:
                    self.cache.put(self._cache_key(prefs, candidates), output)
                stats.source = "llm"
                return SearchOutcome(
                    prefs=prefs,
                    unrated_hidden=counts["unrated_hidden"],
                    value=result,
                )
            except LLMResponseError as exc:
                if reserved:
                    self.guard.record_failure(estimated)
                stats.error_class = type(exc).__name__
                result = self._fallback(candidates, total_matches, _RESPONSE_FAILURE)
                return SearchOutcome(prefs=prefs, unrated_hidden=counts["unrated_hidden"], value=result)
            except LLMError as exc:
                if reserved:
                    self.guard.record_failure(estimated)
                stats.error_class = type(exc).__name__
                result = self._fallback(candidates, total_matches, _PROVIDER_FAILURE)
                return SearchOutcome(
                    prefs=prefs,
                    unrated_hidden=counts["unrated_hidden"],
                    value=result,
                    retryable=isinstance(
                        exc, (LLMTimeout, LLMUnavailable, LLMRateLimited)
                    ),
                )
            except Exception:
                if reserved:
                    self.guard.record_failure(estimated)
                raise
        except Exception as exc:
            stats.error_class = type(exc).__name__
            logger.error("recommend_unexpected_failure\n%s", traceback.format_exc())
            result = self._fallback(
                locals().get("candidates", []),
                locals().get("total_matches", 0),
                _GENERIC_FAILURE,
            )
            return SearchOutcome(prefs=prefs, unrated_hidden=counts["unrated_hidden"], value=result)
        finally:
            if "stats" in locals():
                logger.info(
                    "recommendation_search",
                    extra={
                        "source": stats.source,
                        "step_counts": stats.counts,
                        "candidates_sent": stats.candidates_sent,
                        "tokens": stats.tokens,
                        "latency_ms": stats.latency_ms,
                        "error_class": stats.error_class,
                    },
                )
