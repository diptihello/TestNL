"""Session state helpers and cached repository access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from zomato_reco.config import Settings, get_settings
from zomato_reco.data.repository import Repository, load_repository
from zomato_reco.service.recommender import SearchOutcome

# Widget keys shared with form and relaxation callbacks
KEY_LOCATION = "pref_location"
KEY_BUDGET = "pref_budget"
KEY_MIN_RATING = "pref_min_rating"
KEY_CUISINE = "pref_cuisine"
KEY_NOTES = "pref_notes"

KEY_LAST_RESULT = "last_result"
KEY_BUSY = "busy"
KEY_LAST_SUBMIT_AT = "last_submit_at"
KEY_RECOMMEND_CALLS = "recommend_calls"
KEY_RETRY_PREFS = "retry_prefs"
KEY_CONNECTION_BUSY = "connection_busy"


def _parquet_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def _cached_repository(path_str: str, mtime: float) -> Repository:
    """Load the parquet once per path + modification time."""
    return load_repository(Path(path_str))


def get_repository(settings: Settings | None = None) -> Repository:
    """Return the process-wide repository, refreshing when the parquet changes."""
    settings = settings or get_settings()
    path = settings.processed_path
    return _cached_repository(str(path.resolve()), _parquet_mtime(path))


def init_session() -> None:
    """Set default form and result keys once per browser session."""
    defaults: dict[str, Any] = {
        KEY_LOCATION: None,
        KEY_BUDGET: "medium",
        KEY_MIN_RATING: 4.0,
        KEY_CUISINE: "",
        KEY_NOTES: "",
        KEY_LAST_RESULT: None,
        KEY_BUSY: False,
        KEY_LAST_SUBMIT_AT: 0.0,
        KEY_RECOMMEND_CALLS: 0,
        KEY_RETRY_PREFS: None,
        KEY_CONNECTION_BUSY: False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_result(outcome: SearchOutcome | None) -> None:
    st.session_state[KEY_LAST_RESULT] = outcome


def get_result() -> SearchOutcome | None:
    return st.session_state.get(KEY_LAST_RESULT)


def set_busy(value: bool) -> None:
    st.session_state[KEY_BUSY] = bool(value)


def is_busy() -> bool:
    return bool(st.session_state.get(KEY_BUSY, False))
