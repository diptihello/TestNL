"""UI tests for Phase 3 using Streamlit AppTest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from zomato_reco.config import get_settings
from zomato_reco.data.preprocess import CANONICAL_COLUMNS, clean_frame
from zomato_reco.data.repository import load_repository
from zomato_reco.domain.budget import tier_for
from zomato_reco.domain.models import Preferences, RecommendationResult
from zomato_reco.service.recommender import RecommendationService
from zomato_reco.ui.components import _placeholder, escape_text

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "zomato_sample.csv"
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Build a small parquet + manifest for AppTest."""
    raw = pd.read_csv(FIXTURE_CSV, dtype=str)
    clean = clean_frame(raw)
    # Inject a name with markdown specials for escape tests.
    clean = clean.copy()
    clean.loc[clean.index[0], "name"] = "Cafe $pecial *Grill*"
    clean.loc[clean.index[0], "votes"] = 100_000
    clean.at[clean.index[0], "cuisines"] = []
    clean.loc[clean.index[0], "rest_type"] = None
    clean.loc[clean.index[0], "address"] = ""
    path = tmp_path / "restaurants.parquet"
    clean.to_parquet(path, index=False)
    manifest = {
        "rows_after": len(clean),
        "distinct_locations": int(clean["location"].nunique()),
        "generated_at": "2026-10-15T00:00:00+00:00",
        "schema_columns": list(CANONICAL_COLUMNS),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture
def app_env(processed_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROCESSED_PATH", str(processed_dir / "restaurants.parquet"))
    monkeypatch.setenv("MIN_SUBMIT_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("GROQ_API_KEY", "  ")
    get_settings.cache_clear()
    # Clear Streamlit resource cache between tests when possible
    try:
        from zomato_reco.ui.state import _cached_repository

        _cached_repository.clear()
    except Exception:
        pass
    yield processed_dir
    get_settings.cache_clear()


def _run_app(timeout: float = 10) -> AppTest:
    get_settings.cache_clear()
    at = AppTest.from_file(str(APP_PATH), default_timeout=timeout)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    return at


def _submit(at: AppTest) -> AppTest:
    buttons = [b for b in at.button if "recommend" in (b.label or "").lower()]
    assert buttons, f"submit button not found; labels={[b.label for b in at.button]}"
    buttons[0].click().run()
    assert not at.exception, f"App raised after submit: {at.exception}"
    return at


class TestEscapeText:
    def test_escapes_markdown_and_dollar(self) -> None:
        assert "$" in escape_text("Cafe $pecial")
        assert escape_text("Cafe $pecial") != "Cafe $pecial" or "\\$" in escape_text(
            "Cafe $pecial"
        )
        assert "\\*" in escape_text("a*b")
        assert "\\$" in escape_text("$")


class TestAppLoad:
    def test_loads_and_location_options(self, app_env: Path) -> None:
        repo = load_repository(app_env / "restaurants.parquet")
        at = _run_app()
        # Location selectbox is the first selectbox
        assert at.selectbox
        loc = at.selectbox[0]
        assert list(loc.options) == repo.locations()
        assert loc.value is None

    def test_missing_processed_data_shows_ingest_command(
        self, app_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROCESSED_PATH", str(app_env / "missing.parquet"))
        get_settings.cache_clear()

        at = _run_app()

        assert any("dataset has not been built" in error.value for error in at.error)
        assert any("python scripts\\ingest.py" in item.value for item in at.code)

    def test_test_connection_reports_missing_key_without_calling_provider(
        self, app_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (app_env / "restaurants.parquet").exists()
        monkeypatch.setenv("GROQ_API_KEY", "  ")
        get_settings.cache_clear()
        at = _run_app()
        connection_button = next(
            button for button in at.button if button.label == "Test connection"
        )
        connection_button.click().run()
        assert not at.exception
        assert any("Configure GROQ_API_KEY" in error.value for error in at.error)

    def test_test_connection_uses_client_and_records_usage(
        self, app_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (app_env / "restaurants.parquet").exists()
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "T" * 40)
        get_settings.cache_clear()

        from zomato_reco.ui import components
        from zomato_reco.llm.client import FakeLLMClient

        calls = []

        class GuardStub:
            def __init__(self, store, settings):
                calls.append(("init", store, settings))

            def check_and_reserve(self, prompt_tokens, output_tokens):
                calls.append(("reserve", prompt_tokens, output_tokens))
                return None

            def record(self, usage):
                calls.append(("record", usage))

        def fake_client(settings):
            calls.append(("client", settings.groq_model))
            return FakeLLMClient()

        monkeypatch.setattr(components, "UsageStore", lambda path: path)
        monkeypatch.setattr(components, "BudgetGuard", GuardStub)
        monkeypatch.setattr(components, "make_client", fake_client)

        at = _run_app()
        next(button for button in at.button if button.label == "Test connection").click().run()

        assert not at.exception
        assert any("Demo connection successful." in item.value for item in at.success)
        assert ("reserve", 20, 5) in calls
        assert any(call[0] == "record" for call in calls)
        assert any(call[0] == "client" for call in calls)


class TestSearch:
    def test_submit_blank_cuisine_yields_cards(self, app_env: Path) -> None:
        repo = load_repository(app_env / "restaurants.parquet")
        row = repo.all().loc[lambda frame: frame["cost_for_two"].notna()].iloc[0]
        budget = tier_for(int(row["cost_for_two"]), get_settings())
        at = _run_app()
        at.selectbox[0].set_value(str(row["location"]))
        at.radio[0].set_value(budget.value)
        at.slider[0].set_value(0.0)
        _submit(at)

        assert not at.exception
        result = at.session_state.get("last_result")
        assert result is not None
        assert isinstance(result.value, RecommendationResult)
        assert result.value.recommendations
        assert result.value.source == "llm"
        assert any(rec.explanation for rec in result.value.recommendations)
        body = " ".join(m.value for m in at.markdown)
        assert "Showing best" in body
        assert "**From dataset**" in body
        assert "**AI explanation**" in body
        assert "Ranked by AI" in body
        assert r"Cafe \$pecial \*Grill\*" in body
        assert "Cuisine not listed" in body
        assert "Address not available" in body
        if result.value.extra_recommendations:
            assert any(
                expander.label == "Show more recommendations"
                for expander in at.expander
            )

    def test_invalid_cuisine_is_reported_without_service_call(
        self, app_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected_call():
            raise AssertionError("service should not run for invalid form input")

        monkeypatch.setattr(RecommendationService, "recommend", unexpected_call)
        repo = load_repository(app_env / "restaurants.parquet")
        row = repo.all().loc[lambda frame: frame["cost_for_two"].notna()].iloc[0]
        at = _run_app()
        at.selectbox[0].set_value(str(row["location"]))
        at.radio[0].set_value(tier_for(int(row["cost_for_two"]), get_settings()).value)
        at.text_input[0].set_value("Italian, Chinese, Thai, Mexican")
        _submit(at)

        assert not at.exception
        assert any("at most 3 comma-separated values" in item.value for item in at.error)

    def test_missing_key_shows_fallback_without_ai_column_or_retry(
        self, app_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEMO_MODE", "false")
        monkeypatch.setenv("GROQ_API_KEY", "  ")
        get_settings.cache_clear()
        repo = load_repository(app_env / "restaurants.parquet")
        row = repo.all().loc[lambda frame: frame["cost_for_two"].notna()].iloc[0]
        at = _run_app()
        at.selectbox[0].set_value(str(row["location"]))
        at.radio[0].set_value(tier_for(int(row["cost_for_two"]), get_settings()).value)
        at.slider[0].set_value(0.0)
        _submit(at)

        result = at.session_state["last_result"]
        assert result.value.source == "fallback"
        assert result.value.notice == "AI explanations unavailable: no API key configured."
        body = " ".join(item.value for item in at.markdown)
        assert "Sorted by rating" in body
        assert "**AI explanation**" not in body
        assert not any(button.label == "Retry AI" for button in at.button)

    def test_rerun_without_submit_keeps_result(self, app_env: Path) -> None:
        repo = load_repository(app_env / "restaurants.parquet")
        location = repo.locations()[0]
        at = _run_app()
        at.selectbox[0].set_value(location)
        at.slider[0].set_value(0.0)
        _submit(at)
        first = at.session_state["last_result"]
        calls = at.session_state.get("recommend_calls", 0)

        at.run()
        assert at.session_state["last_result"] is first
        assert calls == 1
        assert at.session_state.get("recommend_calls", 0) == 1

    def test_impossible_query_shows_relaxation(self, app_env: Path) -> None:
        repo = load_repository(app_env / "restaurants.parquet")
        at = _run_app()
        at.selectbox[0].set_value(next(
            location for location in repo.locations() if location == "BTM Layout"
        ))
        at.radio[0].set_value("low")
        at.slider[0].set_value(3.0)
        at.text_input[0].set_value("ZzNotACuisine999")
        _submit(at)

        assert not at.exception
        outcome = at.session_state.get("last_result")
        assert outcome is not None
        from zomato_reco.domain.models import EmptyResult

        assert isinstance(outcome.value, EmptyResult)
        assert outcome.value.relaxations
        relax_buttons = [
            b for b in at.button if b.key and str(b.key).startswith("relax_")
        ]
        assert relax_buttons
        relax_buttons[0].click().run()
        assert not at.exception
        assert at.session_state["pref_cuisine"] == ""

    def test_escaped_text_and_missing_value_placeholders(self) -> None:
        assert escape_text("Cafe $pecial *Grill*") == r"Cafe \$pecial \*Grill\*"
        assert _placeholder(None, "Not yet rated") == "Not yet rated"
        assert _placeholder(float("nan"), "Cost not available") == "Cost not available"

    def test_busy_ignores_second_submit(self, app_env: Path) -> None:
        repo = load_repository(app_env / "restaurants.parquet")
        at = _run_app()
        at.session_state["busy"] = True
        at.run()
        submit = [b for b in at.button if "recommend" in (b.label or "").lower()]
        assert submit and submit[0].disabled
        assert at.session_state.get("recommend_calls", 0) == 0

        at.session_state["busy"] = False
        at.run()
        at.selectbox[0].set_value(repo.locations()[0])
        at.slider[0].set_value(0.0)
        _submit(at)
        assert at.session_state.get("busy") is False
        assert at.session_state.get("recommend_calls", 0) == 1


def test_busy_resets_after_search_exception(
    app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_search(
        service: RecommendationService,
        prefs: Preferences,
        *,
        bypass_cache: bool = False,
    ):
        assert isinstance(service, RecommendationService)
        assert prefs.location == "BTM Layout"
        assert not bypass_cache
        raise RuntimeError("test failure")

    assert (app_env / "restaurants.parquet").exists()
    monkeypatch.setattr(RecommendationService, "recommend", fail_search)
    at = _run_app()
    at.selectbox[0].set_value("BTM Layout")
    at.radio[0].set_value("low")
    at.slider[0].set_value(0.0)
    _submit(at)

    assert not at.exception
    assert at.session_state["busy"] is False
    assert any(
        error.value == "Something went wrong while searching. Please try again."
        for error in at.error
    )


def test_retry_button_bypasses_cache_once(
    app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RecommendationService.recommend
    calls = []

    def fail_once(
        service: RecommendationService,
        prefs: Preferences,
        *,
        bypass_cache: bool = False,
    ):
        calls.append(bypass_cache)
        outcome = original(service, prefs, bypass_cache=bypass_cache)
        if bypass_cache:
            return outcome
        fallback = outcome.value.model_copy(
            update={
                "source": "fallback",
                "notice": (
                    "The AI service did not respond; showing results sorted by rating."
                ),
            }
        )
        return outcome.model_copy(
            update={"value": fallback, "retryable": True}
        )

    monkeypatch.setattr(RecommendationService, "recommend", fail_once)
    at = _run_app()
    repo = load_repository(app_env / "restaurants.parquet")
    row = repo.all().loc[lambda frame: frame["cost_for_two"].notna()].iloc[0]
    at.selectbox[0].set_value(str(row["location"]))
    at.radio[0].set_value(tier_for(int(row["cost_for_two"]), get_settings()).value)
    at.slider[0].set_value(0.0)
    _submit(at)

    assert calls == [False]
    assert any(button.label == "Retry AI" for button in at.button)
    next(button for button in at.button if button.label == "Retry AI").click().run()

    assert not at.exception
    assert calls == [False, True]
    assert at.session_state["last_result"].value.source == "llm"


def test_budget_block_is_rendered_without_provider_retry(
    app_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zomato_reco.llm.guard import BudgetExceeded, BudgetGuard

    def blocked_reservation(*_args, **_kwargs):
        del _args, _kwargs
        return BudgetExceeded(17, "requests per minute")

    monkeypatch.setattr(BudgetGuard, "check_and_reserve", blocked_reservation)
    repo = load_repository(app_env / "restaurants.parquet")
    row = repo.all().loc[lambda frame: frame["cost_for_two"].notna()].iloc[0]
    at = _run_app()
    at.selectbox[0].set_value(str(row["location"]))
    at.radio[0].set_value(tier_for(int(row["cost_for_two"]), get_settings()).value)
    at.slider[0].set_value(0.0)
    _submit(at)

    result = at.session_state["last_result"]
    assert result.value.source == "fallback"
    assert result.value.notice == "Rate limit reached. Try again in 17 seconds."
    assert result.value.recommendations
    assert not any(button.label == "Retry AI" for button in at.button)
