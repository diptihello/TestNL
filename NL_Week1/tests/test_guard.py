"""Rolling budget, persistence and result cache tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from zomato_reco.config import load_settings
from zomato_reco.domain.models import Budget, Candidate, Preferences
from zomato_reco.llm.guard import BudgetExceeded, BudgetGuard, ResultCache, UsageStore
from zomato_reco.llm.schema import LLMOutput, LLMRecommendation


class Clock:
    def __init__(self, value: float = 100_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _settings(tmp_path, **overrides):
    return load_settings(_env_file=tmp_path / "missing.env", **overrides)


def _guard(tmp_path, clock, **settings):
    return BudgetGuard(
        UsageStore(tmp_path / ".usage.json", clock),
        _settings(tmp_path, **settings),
        clock,
    )


def _candidate(candidate_id: str) -> Candidate:
    return Candidate.from_row(
        {
            "id": candidate_id,
            "name": "Place",
            "address": "",
            "location": "BTM",
            "location_key": "btm",
            "cuisines": [],
            "cost_for_two": 400,
            "rating": 4.0,
            "votes": 5,
            "rest_type": None,
            "dish_liked": [],
            "online_order": False,
            "book_table": False,
        }
    )


def _output() -> LLMOutput:
    return LLMOutput(
        summary="Nice.",
        recommendations=[
            LLMRecommendation(id="r_1", rank=1, explanation="Good.", matched=[])
        ],
    )


def test_31st_request_in_minute_is_blocked_and_wait_is_reported(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock)
    for _ in range(30):
        guard.record(0)

    blocked = guard.check_and_reserve(1, 0)

    assert isinstance(blocked, BudgetExceeded)
    assert blocked.limit_name == "requests per minute"
    assert blocked.wait_seconds == 60


def test_minute_token_limit_blocks_even_when_request_limit_does_not(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock, tpm_limit=1000)
    guard.record(950)

    blocked = guard.check_and_reserve(30, 21)

    assert isinstance(blocked, BudgetExceeded)
    assert blocked.limit_name == "tokens per minute"
    assert "tokens per minute" in str(blocked)


def test_daily_token_limit_blocks(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock, tpd_limit=1000)
    guard.record(950)

    blocked = guard.check_and_reserve(30, 21)

    assert isinstance(blocked, BudgetExceeded)
    assert blocked.limit_name == "tokens per day"


def test_worst_case_includes_max_output_and_failure_records_estimate(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock, tpm_limit=100)
    assert guard.check_and_reserve(60, 41).limit_name == "tokens per minute"

    guard = _guard(tmp_path, clock)
    assert guard.check_and_reserve(60, 41) is None
    guard.record_failure(60)
    assert guard.snapshot()["tokens_minute"] == 60
    assert guard.snapshot()["requests_minute"] == 1


def test_concurrent_reservation_cannot_exceed_request_limit(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock, rpm_limit=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: guard.check_and_reserve(1, 0), range(2)))
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, BudgetExceeded) for outcome in outcomes) == 1


def test_missing_corrupt_future_and_expired_usage_events(tmp_path) -> None:
    clock = Clock()
    path = tmp_path / ".usage.json"
    store = UsageStore(path, clock)
    assert store.events == []

    path.write_text("{broken", encoding="utf-8")
    store.reload()
    assert (tmp_path / ".usage.json.bad").exists()
    assert store.events == []

    path.write_text(
        json.dumps([
            {"id": "future", "timestamp": clock.value + 500, "tokens": 4},
            {"id": "old", "timestamp": clock.value - 90_000, "tokens": 7},
        ]),
        encoding="utf-8",
    )
    store.reload()
    assert len(store.events) == 1
    assert store.events[0]["timestamp"] == clock.value


def test_usage_persists_across_store_instances_and_prunes_after_day(tmp_path) -> None:
    clock = Clock()
    guard = _guard(tmp_path, clock)
    assert guard.check_and_reserve(20, 10) is None
    guard.record({"total_tokens": 18})

    reloaded = BudgetGuard(UsageStore(tmp_path / ".usage.json", clock), _settings(tmp_path), clock)
    assert reloaded.snapshot()["requests_day"] == 1
    clock.value += 86_401
    assert reloaded.snapshot()["requests_day"] == 0


def test_result_cache_normalizes_key_and_honors_model_version_ttl(tmp_path) -> None:
    clock = Clock()
    cache = ResultCache(tmp_path / "cache", clock=clock, ttl_seconds=100)
    candidates = [_candidate("r_1")]
    prefs_a = Preferences(
        location=" BTM  Layout ",
        budget=Budget.low,
        min_rating=4,
        cuisine=" North Indian ",
        extra_preferences="Family   friendly",
    )
    prefs_b = Preferences(
        location="btm layout",
        budget=Budget.low,
        min_rating=4,
        cuisine="north indian",
        extra_preferences="family friendly",
    )
    key = cache.key(prefs_a, candidates, "model")
    assert cache.key(prefs_b, candidates, "model") == key
    cache.put(key, _output())
    assert cache.get(key) == _output()
    model_key = cache.key(prefs_a, candidates, "other-model")
    version_key = cache.key(prefs_a, candidates, "model", "next")
    assert model_key != key
    assert version_key != key
    assert cache.get(model_key) is None
    assert cache.get(version_key) is None
    clock.value += 101
    assert cache.get(key) is None


def test_result_cache_handles_corrupt_entry_and_enforces_entry_cap(tmp_path) -> None:
    clock = Clock()
    cache = ResultCache(tmp_path / "cache", clock=clock, max_entries=2)
    cache.directory.mkdir()
    corrupt = cache.directory / ("a" * 64 + ".json")
    corrupt.write_text("bad", encoding="utf-8")
    assert cache.get("a" * 64) is None
    assert not corrupt.exists()
    for index in range(3):
        cache.put(f"{index:064x}", _output())
        clock.value += 1
    assert len(list(cache.directory.glob("*.json"))) == 2
