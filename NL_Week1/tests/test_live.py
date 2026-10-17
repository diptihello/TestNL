"""Opt-in smoke tests against the configured Groq endpoint."""

from __future__ import annotations

import json

import pytest

from zomato_reco.config import get_settings
from zomato_reco.data.repository import load_repository
from zomato_reco.domain.models import Budget, Candidate, Preferences
from zomato_reco.llm.client import make_client
from zomato_reco.llm.prompts import SYSTEM_PROMPT, build_user_prompt, estimate_tokens


@pytest.mark.live
def test_groq_connection_and_ranking_smoke() -> None:
    settings = get_settings()
    if not settings.groq_api_key.get_secret_value():
        pytest.skip("GROQ_API_KEY is not configured.")
    if not settings.dataset_ready:
        pytest.skip("Processed restaurant dataset is not available.")

    repository = load_repository(settings.processed_path)
    frame = repository.all()
    eligible = frame.loc[
        frame["rating"].notna()
        & frame["cost_for_two"].gt(settings.budget_low_max)
        & frame["cost_for_two"].le(settings.budget_medium_max)
    ]
    if eligible.empty:
        pytest.skip("No rated medium-budget restaurants are available.")
    location_key = str(eligible.iloc[0]["location_key"])
    sample = eligible.loc[eligible["location_key"] == location_key].head(3)
    candidates = [Candidate.from_row(row) for _, row in sample.iterrows()]
    if len(candidates) != 3:
        pytest.skip("Three rated restaurants are required for the smoke test.")

    prefs = Preferences(
        location=candidates[0].restaurant.location,
        budget=Budget.medium,
        min_rating=0.0,
    )
    user_prompt = build_user_prompt(prefs, candidates, settings)
    client = make_client(settings)
    status = client.test_connection()
    response = client.rank(SYSTEM_PROMPT, user_prompt, settings.llm_max_output_tokens)

    parsed = json.loads(response.content)
    assert status.ok
    assert isinstance(parsed, dict)
    assert isinstance(parsed.get("recommendations"), list)
    assert response.finish_reason == "stop"
    assert response.total_tokens < 1500
    print(
        "Groq smoke usage: "
        f"prompt={response.prompt_tokens}, completion={response.completion_tokens}, "
        f"total={response.total_tokens}, finish_reason={response.finish_reason}, "
        f"estimated_prompt={estimate_tokens(SYSTEM_PROMPT + chr(10) + user_prompt)}"
    )
