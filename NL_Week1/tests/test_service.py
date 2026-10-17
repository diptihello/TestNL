"""Unit tests for the Phase 3 recommendation service."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from zomato_reco.config import Settings, load_settings
from zomato_reco.data.preprocess import CANONICAL_COLUMNS, clean_frame
from zomato_reco.data.repository import load_repository
from zomato_reco.domain.budget import tier_for
from zomato_reco.domain.models import (
    Budget,
    EmptyResult,
    Preferences,
    RecommendationResult,
)
from zomato_reco.domain import filters
from zomato_reco.domain.ranking import select
from zomato_reco.llm.client import FakeLLMClient, LLMTimeout
from zomato_reco.llm.guard import BudgetGuard, ResultCache, UsageStore
from zomato_reco.llm.schema import LLMOutput, LLMRecommendation
from zomato_reco.service.recommender import (
    LocationNotFoundError,
    RecommendationService,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "zomato_sample.csv"


@pytest.fixture
def service(
    tmp_path: Path,
) -> tuple[RecommendationService, pd.DataFrame, Settings]:
    frame = clean_frame(pd.read_csv(FIXTURE_PATH, dtype=str))
    parquet = tmp_path / "restaurants.parquet"
    frame.to_parquet(parquet, index=False)
    manifest = {
        "rows_after": len(frame),
        "schema_columns": list(CANONICAL_COLUMNS),
        "distinct_locations": int(frame["location"].nunique()),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    repository = load_repository(parquet)
    settings = load_settings(_env_file=tmp_path / "missing.env")
    guard = BudgetGuard(UsageStore(tmp_path / "usage.json"), settings)
    cache = ResultCache(tmp_path / "cache")
    return (
        RecommendationService(repository, settings, guard=guard, cache=cache),
        frame,
        settings,
    )


def test_recommend_returns_deterministic_fallback(service) -> None:
    recommender, frame, settings = service
    row = frame.loc[frame["cost_for_two"].notna()].iloc[0]
    budget = tier_for(int(row["cost_for_two"]), settings)
    assert budget is not None
    prefs = Preferences(
        location=str(row["location"]),
        budget=budget,
        min_rating=0.0,
        cuisine="",
        extra_preferences=None,
    )

    outcome = recommender.recommend(prefs)

    assert isinstance(outcome.value, RecommendationResult)
    assert outcome.value.source == "fallback"
    assert outcome.value.notice == "AI explanations unavailable: no API key configured."
    assert outcome.value.total_matches >= len(outcome.value.recommendations) > 0
    assert [rec.rank for rec in outcome.value.recommendations] == list(
        range(1, len(outcome.value.recommendations) + 1)
    )
    assert recommender.call_count == 1


def test_recommend_returns_empty_result_with_relaxations(service) -> None:
    recommender, _, _ = service
    prefs = Preferences(
        location="BTM Layout",
        budget=Budget.low,
        min_rating=3.0,
        cuisine="ZzNotACuisine999",
    )

    outcome = recommender.recommend(prefs)

    assert isinstance(outcome.value, EmptyResult)
    assert outcome.value.relaxations
    assert all(hint.would_match > 0 for hint in outcome.value.relaxations)


def test_recommend_validates_location_case_insensitively(service) -> None:
    recommender, frame, _ = service
    location = str(frame.iloc[0]["location"])
    prefs = Preferences(location=location.swapcase(), budget=Budget.low, min_rating=0.0)

    outcome = recommender.recommend(prefs)

    assert outcome.prefs.location == location.swapcase()


def test_recommend_rejects_unknown_location(service) -> None:
    recommender, _, _ = service
    prefs = Preferences(location="Not a Bangalore area", budget=Budget.low, min_rating=0.0)

    with pytest.raises(LocationNotFoundError, match="not in the dataset"):
        recommender.recommend(prefs)


def _ready_service(service, tmp_path, client, **overrides):
    original, _, _ = service
    settings = load_settings(
        _env_file=tmp_path / "missing.env",
        groq_api_key="test-key",
        **overrides,
    )
    guard = BudgetGuard(UsageStore(tmp_path / "usage.json"), settings)
    cache = ResultCache(tmp_path / "cache")
    return (
        RecommendationService(
            original.repository,
            settings,
            client=client,
            guard=guard,
            cache=cache,
        ),
        settings,
        guard,
        cache,
    )


def _prefs_with_candidates(recommender, settings, minimum=1):
    rows = recommender.repository.all()
    for location in recommender.repository.locations():
        for budget in Budget:
            prefs = Preferences(location=location, budget=budget, min_rating=0)
            filtered, _ = filters.apply(rows, prefs, settings)
            candidates, _ = select(filtered, prefs, settings)
            if len(candidates) >= minimum:
                return prefs, candidates
    pytest.skip("fixture dataset has too few matching candidates")


def test_ai_results_use_repository_facts_and_cache_hit_skips_guard(service, tmp_path) -> None:
    client = FakeLLMClient()
    recommender, settings, guard, _ = _ready_service(
        service, tmp_path, client, max_recommendations=2
    )
    prefs, _ = _prefs_with_candidates(recommender, settings, minimum=2)
    reserve_calls = 0
    original_reserve = guard.check_and_reserve

    def counted_reserve(prompt_tokens, output_tokens):
        nonlocal reserve_calls
        reserve_calls += 1
        return original_reserve(prompt_tokens, output_tokens)

    guard.check_and_reserve = counted_reserve
    first = recommender.recommend(prefs)
    for _ in range(settings.rpm_limit):
        guard.record(0)
    second = recommender.recommend(prefs)

    assert first.value.source == "llm"
    assert second.value.source == "cache"
    assert reserve_calls == 1
    assert second.cached_age_seconds is not None
    assert first.value.recommendations
    rows = recommender.repository.by_ids(
        [rec.restaurant.id for rec in first.value.recommendations]
    )
    facts_by_id = {str(row["id"]): row for _, row in rows.iterrows()}
    for rec in first.value.recommendations:
        assert rec.restaurant == type(rec.restaurant).from_row(facts_by_id[rec.restaurant.id])


def test_timeout_returns_retryable_fallback_and_records_estimated_usage(service, tmp_path) -> None:
    client = FakeLLMClient(scripted={"error": LLMTimeout})
    recommender, settings, guard, cache = _ready_service(service, tmp_path, client)
    prefs, _ = _prefs_with_candidates(recommender, settings)

    outcome = recommender.recommend(prefs)

    assert outcome.value.source == "fallback"
    assert outcome.value.notice == (
        "The AI service did not respond; showing results sorted by rating."
    )
    assert outcome.retryable
    assert outcome.value.recommendations
    assert guard.snapshot()["requests_day"] == 1
    assert guard.snapshot()["tokens_day"] > 0
    assert not cache.directory.exists() or not list(cache.directory.glob("*.json"))


def test_malformed_response_falls_back_without_retry(service, tmp_path) -> None:
    recommender, settings, _, _ = _ready_service(
        service, tmp_path, FakeLLMClient(scripted={"malformed": True})
    )
    prefs, _ = _prefs_with_candidates(recommender, settings)

    outcome = recommender.recommend(prefs)

    assert outcome.value.source == "fallback"
    assert outcome.value.notice == (
        "The AI response could not be read; showing results sorted by rating."
    )
    assert not outcome.retryable


def test_partial_ai_output_is_labeled_and_extras_are_retained(service, tmp_path) -> None:
    recommender, settings, _, _ = _ready_service(
        service, tmp_path, FakeLLMClient(), max_recommendations=2
    )
    prefs, candidates = _prefs_with_candidates(recommender, settings, minimum=3)
    content = json.dumps(
        {
            "recommendations": [
                {
                    "id": candidates[0].id,
                    "rank": 1,
                    "explanation": "A good option based on the available data.",
                    "matched": [],
                }
            ]
        }
    )
    recommender.client = FakeLLMClient(scripted={"content": content})

    outcome = recommender.recommend(prefs)

    assert outcome.value.notice == "AI explanations available for 1 of 2."
    assert outcome.value.source == "llm"


def test_demo_mode_skips_result_cache(service, tmp_path) -> None:
    class CacheSpy(ResultCache):
        reads = writes = 0

        def get_with_age(self, key):
            self.reads += 1
            return super().get_with_age(key)

        def put(self, key, output):
            self.writes += 1
            super().put(key, output)

    original, _, _ = service
    settings = load_settings(
        _env_file=tmp_path / "missing.env",
        demo_mode=True,
        max_recommendations=2,
    )
    cache = CacheSpy(tmp_path / "cache")
    recommender = RecommendationService(
        original.repository,
        settings,
        client=FakeLLMClient(),
        guard=BudgetGuard(UsageStore(tmp_path / "usage.json"), settings),
        cache=cache,
    )
    prefs, _ = _prefs_with_candidates(recommender, settings)

    assert recommender.recommend(prefs).value.source == "llm"
    assert cache.reads == cache.writes == 0


def test_cached_ids_are_revalidated_against_current_candidates(service, tmp_path) -> None:
    recommender, settings, _, cache = _ready_service(
        service, tmp_path, FakeLLMClient(), max_recommendations=2
    )
    prefs, candidates = _prefs_with_candidates(recommender, settings)
    key = cache.key(prefs, candidates, settings.groq_model)
    cache.put(
        key,
        LLMOutput(
            recommendations=[
                LLMRecommendation(
                    id="not-in-the-repository",
                    rank=1,
                    explanation="Stale result",
                )
            ]
        ),
    )

    outcome = recommender.recommend(prefs)

    assert outcome.value.source == "llm"
    assert outcome.value.recommendations[0].restaurant.id in {
        candidate.id for candidate in candidates
    }


def test_budget_block_keeps_fallback_cards_and_wait_notice(service, tmp_path) -> None:
    original, _, _ = service
    settings = load_settings(
        _env_file=tmp_path / "missing.env",
        groq_api_key="test-key",
        rpm_limit=1,
    )
    guard = BudgetGuard(UsageStore(tmp_path / "usage.json"), settings)
    guard.record(12)
    recommender = RecommendationService(
        original.repository,
        settings,
        client=FakeLLMClient(),
        guard=guard,
        cache=ResultCache(tmp_path / "cache"),
    )
    prefs, _ = _prefs_with_candidates(recommender, settings)

    outcome = recommender.recommend(prefs)

    assert outcome.value.source == "fallback"
    assert "Rate limit reached. Try again in 60 seconds." == outcome.value.notice
    assert outcome.value.recommendations


def test_wrong_numbers_are_replaced_and_outsider_summary_is_dropped(
    service, tmp_path, caplog
) -> None:
    recommender, settings, _, _ = _ready_service(
        service, tmp_path, FakeLLMClient(), max_recommendations=1
    )
    prefs, candidates = _prefs_with_candidates(recommender, settings)
    restaurant = candidates[0].restaurant
    wrong_rating = 0 if restaurant.rating != 0 else 5
    wrong_cost = 1 if restaurant.cost_for_two != 1 else 2
    content = json.dumps(
        {
            "summary": "Visit Totally Unknown Place for a meal.",
            "recommendations": [
                {
                    "id": restaurant.id,
                    "rank": 1,
                    "explanation": f"Rated {wrong_rating}/5 and costs INR {wrong_cost}.",
                    "matched": [],
                }
            ],
        }
    )
    recommender.client = FakeLLMClient(scripted={"content": content})
    prefs = prefs.model_copy(update={"extra_preferences": "PRIVATE_NOTE_NEVER_LOG"})
    caplog.set_level("INFO")

    outcome = recommender.recommend(prefs)

    rec = outcome.value.recommendations[0]
    assert rec.explanation != content
    assert f"{wrong_rating}/5" not in rec.explanation
    assert f"INR {wrong_cost}" not in rec.explanation
    assert outcome.value.summary is None
    assert "PRIVATE_NOTE_NEVER_LOG" not in caplog.text
    assert "test-key" not in caplog.text
    assert "USER PREFERENCES" not in caplog.text
    assert "recommendation_search" in caplog.text


def test_unexpected_client_failure_returns_generic_fallback(service, tmp_path) -> None:
    recommender, settings, guard, _ = _ready_service(
        service,
        tmp_path,
        FakeLLMClient(scripted={"error": RuntimeError("private details")}),
    )
    prefs, _ = _prefs_with_candidates(recommender, settings)

    outcome = recommender.recommend(prefs)

    assert outcome.value.source == "fallback"
    assert outcome.value.notice == (
        "AI recommendations are temporarily unavailable; showing results sorted by rating."
    )
    assert outcome.value.recommendations
    assert guard.snapshot()["requests_day"] == 1
