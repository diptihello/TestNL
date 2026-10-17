"""Streamlit UI components for the recommendation form and results."""

from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st
from pydantic import ValidationError

from zomato_reco.config import Settings
from zomato_reco.data.repository import Repository
from zomato_reco.domain.models import (
    Budget,
    EmptyResult,
    Preferences,
    Recommendation,
    RecommendationResult,
)
from zomato_reco.llm.client import LLMError, make_client
from zomato_reco.llm.guard import BudgetGuard, UsageStore
from zomato_reco.llm.grounding import escape_text
from zomato_reco.ui import state as ui_state

logger = logging.getLogger(__name__)


def _placeholder(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and value != value:  # NaN
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return fallback
    return text


def render_sidebar(settings: Settings, repository: Repository) -> None:
    """Dataset stats, model label, connection test and rolling usage counters."""
    manifest = repository.manifest()
    with st.sidebar:
        st.subheader("Model")
        st.write(settings.groq_model)
        test_connection = st.button(
            "Test connection",
            use_container_width=True,
            disabled=ui_state.is_busy()
            or bool(st.session_state.get(ui_state.KEY_CONNECTION_BUSY, False)),
        )
        if test_connection:
            st.session_state[ui_state.KEY_CONNECTION_BUSY] = True
            try:
                if not settings.groq_api_key.get_secret_value():
                    st.error("Configure GROQ_API_KEY to test the AI connection.")
                elif settings.demo_mode:
                    st.error("Connection testing is unavailable in demo mode.")
                else:
                    guard: BudgetGuard | None = None
                    reserved = False
                    try:
                        guard = BudgetGuard(
                            UsageStore(settings.usage_path), settings
                        )
                        exceeded = guard.check_and_reserve(20, 5)
                        if exceeded:
                            st.error(
                                f"{exceeded.limit_name} reached. Try again in "
                                f"{exceeded.wait_seconds} seconds."
                            )
                        else:
                            reserved = True
                            status = make_client(settings).test_connection()
                            guard.record(status.tokens_used)
                            st.success(
                                f"{status.message} {status.model} · "
                                f"{status.latency_ms} ms · {status.tokens_used} tokens"
                            )
                    except LLMError as exc:
                        if guard is not None and reserved:
                            guard.record_failure(20)
                        st.error(str(exc))
                    except Exception:
                        if guard is not None and reserved:
                            guard.record_failure(20)
                        logger.exception("connection_test_failed")
                        st.error(
                            "The AI connection test failed. Check the application logs."
                        )
            finally:
                st.session_state[ui_state.KEY_CONNECTION_BUSY] = False
        st.caption("One minimal request; only runs when clicked.")

        st.subheader("Usage today")
        try:
            snapshot = BudgetGuard(UsageStore(settings.usage_path), settings).snapshot()
            st.write(
                f"requests {snapshot['requests_day']} / "
                f"{snapshot['requests_day_limit']}"
            )
            st.write(f"tokens {snapshot['tokens_day']:,} / {snapshot['tokens_day_limit']:,}")
            st.caption(
                f"Last minute: {snapshot['requests_minute']} requests · "
                f"{snapshot['tokens_minute']:,} tokens"
            )
        except Exception:
            logger.exception("usage_snapshot_failed")
            st.caption("Usage counters are temporarily unavailable.")

        st.subheader("Dataset")
        rows = manifest.get("rows_after", len(repository.all()))
        locs = manifest.get("distinct_locations", len(repository.locations()))
        built = manifest.get("generated_at", "unknown")
        if isinstance(built, str) and "T" in built:
            built = built.split("T", 1)[0]
        st.write(f"{rows:,} restaurants")
        st.write(f"{locs} locations")
        st.write(f"built {built}")
        if manifest.get("sample_size") is not None:
            st.warning("sample data")


def render_form(repository: Repository, settings: Settings) -> Preferences | None:
    """Render the search form. Returns Preferences on a valid submit, else None."""
    locations = repository.locations()
    cuisine_examples = ", ".join(repository.cuisines()[:8]) or "Italian, Chinese"

    with st.form("search_form", clear_on_submit=False):
        st.selectbox(
            "Location",
            options=locations,
            index=None,
            placeholder="Select a location",
            key=ui_state.KEY_LOCATION,
            help="Bangalore areas from the dataset. Required.",
        )
        st.radio(
            "Budget",
            options=[b.value for b in Budget],
            horizontal=True,
            key=ui_state.KEY_BUDGET,
            help="Approximate cost for two people.",
        )
        st.slider(
            "Minimum rating",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            key=ui_state.KEY_MIN_RATING,
        )
        st.text_input(
            "Cuisine (optional)",
            key=ui_state.KEY_CUISINE,
            help=f"Examples from the dataset: {cuisine_examples}. Up to 3 comma-separated values.",
        )
        st.text_area(
            "Additional preferences (optional)",
            key=ui_state.KEY_NOTES,
            max_chars=300,
            height=80,
        )
        submitted = st.form_submit_button(
            "Get recommendations",
            type="primary",
            disabled=ui_state.is_busy(),
        )

    if not submitted:
        return None

    if ui_state.is_busy():
        return None

    interval = settings.min_submit_interval_seconds
    now = time.monotonic()
    last = float(st.session_state.get(ui_state.KEY_LAST_SUBMIT_AT, 0.0))
    if interval > 0 and last > 0 and (now - last) < interval:
        st.warning(f"Please wait {interval} seconds between searches.")
        return None

    location = st.session_state.get(ui_state.KEY_LOCATION)
    if location is None or location == "":
        st.error("Please choose a location.")
        return None

    try:
        prefs = Preferences(
            location=str(location),
            budget=st.session_state.get(ui_state.KEY_BUDGET, "medium"),
            min_rating=float(st.session_state.get(ui_state.KEY_MIN_RATING, 4.0)),
            cuisine=st.session_state.get(ui_state.KEY_CUISINE) or None,
            extra_preferences=st.session_state.get(ui_state.KEY_NOTES) or None,
        )
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "input"
            st.error(f"{loc}: {err.get('msg', 'invalid value')}")
        return None

    return prefs


def _source_badge(source: str, cached_age_seconds: float | None = None) -> str:
    if source == "llm":
        return "Ranked by AI"
    if source == "cache":
        age = max(0, int(cached_age_seconds or 0))
        if age < 60:
            age_label = f"{age}s ago"
        elif age < 3600:
            age_label = f"{age // 60}m ago"
        else:
            age_label = f"{age // 3600}h ago"
        return f"Ranked by AI (cached {age_label})"
    return "Sorted by rating"


def render_result_header(
    result: RecommendationResult,
    *,
    unrated_hidden: int,
    sample: bool = False,
    cached_age_seconds: float | None = None,
) -> None:
    """Show match counts, unrated-hidden note, and source badge."""
    # Never claim more results than total matches (e.g. "5 of 3").
    card_n = min(len(result.recommendations), result.total_matches)
    st.markdown(
        f"Showing best **{card_n}** of **{result.total_matches}** matches"
        f" · {_source_badge(result.source, cached_age_seconds)}"
    )
    if unrated_hidden > 0:
        st.caption(f"{unrated_hidden} unrated restaurants hidden")
    if sample:
        st.info("sample data")


def render_notice(result: RecommendationResult) -> None:
    if result.notice:
        st.info(escape_text(result.notice))


def render_summary(result: RecommendationResult) -> None:
    if result.summary:
        st.markdown("**Summary (AI)**")
        st.write(escape_text(result.summary))


def _queue_retry(prefs: dict[str, Any]) -> None:
    st.session_state[ui_state.KEY_RETRY_PREFS] = prefs


def render_card(rec: Recommendation, show_ai: bool) -> None:
    """One result card: dataset facts first, optional AI column."""
    r = rec.restaurant
    st.markdown(f"### #{rec.rank}  {escape_text(r.name)}")

    if show_ai:
        left, right = st.columns(2, gap="medium")
    else:
        left = st.container()
        right = None

    with left:
        st.markdown("**From dataset**")
        cuisines = ", ".join(r.cuisines) if r.cuisines else ""
        st.write(escape_text(_placeholder(cuisines, "Cuisine not listed")))
        if r.rating is None:
            st.write("Not yet rated")
        else:
            st.write(f"{r.rating:.1f} ({r.votes} votes)")
        if r.cost_for_two is None:
            st.write("Cost not available")
        else:
            st.write(f"INR {r.cost_for_two} for two")
        st.write(escape_text(_placeholder(r.rest_type, "—")))
        flags = []
        if r.online_order:
            flags.append("Online order")
        if r.book_table:
            flags.append("Table booking")
        if flags:
            st.write(", ".join(flags))
        with st.expander("Address"):
            st.write(escape_text(_placeholder(r.address, "Address not available")))

    if right is not None:
        with right:
            st.markdown("**AI explanation**")
            if rec.explanation:
                st.write(escape_text(rec.explanation))
                if rec.matched:
                    st.caption("Matched: " + escape_text(", ".join(rec.matched)))
            else:
                st.caption("AI explanations are unavailable for this result.")

    st.divider()


def _apply_relaxation(field: str, value: Any) -> None:
    """Callback: write a suggested value into the matching widget session key."""
    if field == "min_rating":
        st.session_state[ui_state.KEY_MIN_RATING] = float(value)
    elif field == "budget":
        st.session_state[ui_state.KEY_BUDGET] = str(value)
    elif field == "cuisine":
        st.session_state[ui_state.KEY_CUISINE] = "" if value is None else str(value)
    elif field == "location":
        st.session_state[ui_state.KEY_LOCATION] = value


def render_empty(empty: EmptyResult) -> None:
    """Empty-state message plus clickable relaxation hints."""
    st.warning(escape_text(empty.reason))
    if not empty.relaxations:
        return
    st.markdown("**Try a change** (then press Get recommendations):")
    for i, hint in enumerate(empty.relaxations):
        st.button(
            f"{hint.label} ({hint.would_match})",
            key=f"relax_{i}_{hint.field}_{hint.value}",
            on_click=_apply_relaxation,
            args=(hint.field, hint.value),
        )


def render_outcome(
    outcome: Any,
    *,
    sample: bool = False,
    show_ai: bool = True,
) -> None:
    """Render either an EmptyResult or RecommendationResult from a SearchOutcome."""
    from zomato_reco.service.recommender import SearchOutcome

    if outcome is None:
        return
    if not isinstance(outcome, SearchOutcome):
        return

    value = outcome.value
    if isinstance(value, EmptyResult):
        render_empty(value)
        return

    if isinstance(value, RecommendationResult):
        render_notice(value)
        render_result_header(
            value,
            unrated_hidden=outcome.unrated_hidden if outcome.prefs.min_rating > 0 else 0,
            sample=sample,
            cached_age_seconds=outcome.cached_age_seconds,
        )
        for rec in value.recommendations:
            render_card(rec, show_ai=show_ai)
        render_summary(value)
        if value.extra_recommendations:
            with st.expander("Show more recommendations"):
                for rec in value.extra_recommendations:
                    render_card(rec, show_ai=show_ai)
        if outcome.retryable:
            st.button(
                "Retry AI",
                disabled=ui_state.is_busy(),
                on_click=_queue_retry,
                args=(outcome.prefs.model_dump(mode="python"),),
            )
