"""Streamlit entry point: form, deterministic search, and result cards (Phase 3)."""

from __future__ import annotations

import logging
import time
import traceback

import streamlit as st

st.set_page_config(page_title="Restaurant recommendations", layout="wide")

try:
    from zomato_reco.config import ConfigError, get_settings
    from zomato_reco.logging_setup import configure_logging
except ModuleNotFoundError as exc:
    if (exc.name or "").startswith("zomato_reco"):
        st.error(
            "The zomato_reco package is not installed. "
            "Run `pip install -e .` in the project folder, then restart the app."
        )
    else:
        st.error(
            f"A required library is missing ({exc.name}). "
            "Run `pip install -r requirements.txt`, then restart the app."
        )
    st.stop()

try:
    settings = get_settings()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

configure_logging(settings)
logger = logging.getLogger("zomato_reco.app")

try:
    from zomato_reco.service.recommender import (
        LocationNotFoundError,
        RecommendationService,
    )
    from zomato_reco.ui import state as ui_state
    from zomato_reco.ui.components import render_form, render_outcome, render_sidebar
except Exception:  # noqa: BLE001
    logger.exception("import_failed")
    st.error("Something went wrong while starting the app. Check the logs.")
    st.stop()


def main() -> None:
    logger.info(
        "app_loaded",
        extra={
            "dataset_ready": settings.dataset_ready,
            "llm_enabled": settings.llm_enabled,
            "demo_mode": settings.demo_mode,
        },
    )

    st.title("Find a restaurant in Bangalore")
    st.caption(
        "Search the Zomato Bangalore dataset by location, budget and rating. "
        "AI recommendations are grounded in the restaurant dataset."
    )

    if not settings.dataset_ready:
        st.error("The restaurant dataset has not been built yet.")
        st.markdown("Run this command in the project folder, then reload the page:")
        st.code("python scripts\\ingest.py", language="powershell")
        st.stop()

    ui_state.init_session()

    try:
        repository = ui_state.get_repository(settings)
    except Exception:  # noqa: BLE001
        logger.exception("repository_load_failed")
        st.error("Data could not be read. Run `python scripts\\ingest.py` and reload.")
        st.stop()

    render_sidebar(settings, repository)
    service = RecommendationService(repository, settings)

    prefs = render_form(repository, settings)
    retry_values = st.session_state.get(ui_state.KEY_RETRY_PREFS)
    retry = prefs is None and isinstance(retry_values, dict)
    if retry:
        st.session_state[ui_state.KEY_RETRY_PREFS] = None
        try:
            from zomato_reco.domain.models import Preferences

            prefs = Preferences.model_validate(retry_values)
        except Exception:
            logger.exception("retry_preferences_invalid")
            st.error("The previous search could not be retried. Submit the form again.")

    if prefs is not None:
        interval = settings.min_submit_interval_seconds
        now = time.monotonic()
        last = float(st.session_state.get(ui_state.KEY_LAST_SUBMIT_AT, 0.0))
        if interval > 0 and last > 0 and now - last < interval:
            st.warning(f"Please wait {interval} seconds between searches.")
            prefs = None

    if prefs is not None:
        ui_state.set_busy(True)
        try:
            with st.spinner("Filtering restaurants... asking the model..."):
                outcome = service.recommend(prefs, bypass_cache=retry)
            ui_state.set_result(outcome)
            st.session_state[ui_state.KEY_LAST_SUBMIT_AT] = time.monotonic()
            st.session_state[ui_state.KEY_RECOMMEND_CALLS] = (
                int(st.session_state.get(ui_state.KEY_RECOMMEND_CALLS, 0)) + 1
            )
            if settings.log_level == "DEBUG":
                logger.debug(
                    "recommend_audit",
                    extra={
                        "recommend_calls": st.session_state[ui_state.KEY_RECOMMEND_CALLS]
                    },
                )
        except LocationNotFoundError as exc:
            st.error(str(exc))
        except Exception:  # noqa: BLE001
            logger.error("recommend_failed\n%s", traceback.format_exc())
            st.error("Something went wrong while searching. Please try again.")
        finally:
            ui_state.set_busy(False)

    outcome = ui_state.get_result()
    sample = repository.manifest().get("sample_size") is not None
    render_outcome(
        outcome,
        sample=sample,
        show_ai=bool(
            outcome
            and getattr(outcome.value, "source", None) in {"llm", "cache"}
        ),
    )

    if settings.log_level == "DEBUG" and outcome is not None:
        st.sidebar.caption(
            f"debug: recommend() calls this session = "
            f"{st.session_state.get(ui_state.KEY_RECOMMEND_CALLS, 0)}"
        )


try:
    main()
except Exception:  # noqa: BLE001
    logger.error("unhandled\n%s", traceback.format_exc())
    st.error("Something went wrong. Please reload the page.")
