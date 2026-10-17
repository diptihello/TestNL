"""Strict model-output parsing and grounding tests."""

from __future__ import annotations

import json

import pytest

from zomato_reco.config import load_settings
from zomato_reco.domain.models import Candidate
from zomato_reco.llm.client import LLMResponseError
from zomato_reco.llm.grounding import check_explanation, check_summary, neutral_explanation
from zomato_reco.llm.schema import parse_output


@pytest.fixture
def settings(tmp_path):
    return load_settings(_env_file=tmp_path / "missing.env")


def _candidate(index: int, **overrides) -> Candidate:
    row = {
        "id": f"r_{index}",
        "name": f"Restaurant {index}",
        "address": "Road",
        "location": "BTM",
        "location_key": "btm",
        "cuisines": ["North Indian"],
        "cost_for_two": 400,
        "rating": 4.2,
        "votes": 100,
        "rest_type": "Casual dining",
        "dish_liked": ["Pasta"],
        "online_order": True,
        "book_table": False,
    }
    row.update(overrides)
    return Candidate.from_row(row)


def _item(candidate_id: str, **overrides) -> dict:
    value = {
        "id": candidate_id,
        "rank": 99,
        "explanation": "A good option for this search.",
        "matched": ["budget"],
    }
    value.update(overrides)
    return value


def _raw(recommendations, summary="A useful set of choices.") -> str:
    return json.dumps({"summary": summary, "recommendations": recommendations})


def test_valid_output_reassigns_rank_and_accepts_numeric_text(settings) -> None:
    output, stats = parse_output(_raw([_item(" r_1 ", rank="9")]), {"r_1"}, settings)
    assert output.recommendations[0].rank == 1
    assert output.recommendations[0].id == "r_1"
    assert stats.dropped_unknown_ids == 0


def test_unknown_duplicate_and_invalid_items_are_dropped_and_counted(settings) -> None:
    output, stats = parse_output(
        _raw([
            _item("unknown"),
            _item("r_1"),
            _item("r_1"),
            {"id": "r_2", "rank": "1.5", "explanation": "Bad rank"},
        ]),
        {"r_1"},
        settings,
    )
    assert [item.id for item in output.recommendations] == ["r_1"]
    assert stats.dropped_unknown_ids == 1
    assert stats.dropped_duplicates == 1
    assert stats.dropped_invalid == 1


def test_explanation_is_word_boundary_truncated(settings) -> None:
    text = "word " * 70
    output, _ = parse_output(_raw([_item("r_1", explanation=text)]), {"r_1"}, settings)
    explanation = output.recommendations[0].explanation
    assert len(explanation) <= 300
    assert explanation.endswith("…")
    assert not explanation.endswith("wo…")


def test_empty_explanation_dropped(settings) -> None:
    output, stats = parse_output(
        _raw([_item("r_1", explanation="  "), _item("r_2")]),
        {"r_1", "r_2"},
        settings,
    )
    assert [item.id for item in output.recommendations] == ["r_2"]
    assert stats.dropped_empty_explanations == 1


def test_code_fence_is_accepted_but_other_surrounding_text_is_rejected(settings) -> None:
    payload = _raw([_item("r_1")])
    output, _ = parse_output(f"```json\n{payload}\n```", {"r_1"}, settings)
    assert output.recommendations
    with pytest.raises(LLMResponseError):
        parse_output(f"Here you go: {payload}", {"r_1"}, settings)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[]",
        '{"summary":"x"}',
        '{"recommendations":[]}',
        '{"recommendations":[{"id":"r_1","rank":1,"explanation":"  "}]}',
    ],
)
def test_invalid_or_all_invalid_output_raises(settings, raw) -> None:
    with pytest.raises(LLMResponseError):
        parse_output(raw, {"r_1"}, settings)


def test_missing_summary_is_accepted(settings) -> None:
    output, _ = parse_output(
        json.dumps({"recommendations": [_item("r_1")]}), {"r_1"}, settings
    )
    assert output.summary is None


def test_summary_is_truncated(settings) -> None:
    output, _ = parse_output(
        _raw([_item("r_1")], "Useful sentence. " + ("extra " * 80)),
        {"r_1"},
        settings,
    )
    assert output.summary is not None
    assert len(output.summary.split()) <= 60
    assert len(output.summary) <= 400


def test_recommendations_are_capped_and_extra_valid_items_retained(settings) -> None:
    candidates = {f"r_{index}" for index in range(7)}
    output, stats = parse_output(
        _raw([_item(candidate_id) for candidate_id in sorted(candidates)]),
        candidates,
        settings,
    )
    assert len(output.recommendations) == settings.max_recommendations
    assert len(output.extra_recommendations) == len(candidates) - settings.max_recommendations
    assert stats.extra_recommendations == len(output.extra_recommendations)


def test_wrong_rating_or_cost_is_not_grounded() -> None:
    candidate = _candidate(1)
    assert not check_explanation("Rated 4.8 with cost INR 900.", candidate)
    assert check_explanation("Rated 4.2 with cost INR 400.", candidate)
    replacement = neutral_explanation(candidate)
    assert "Restaurant 1" in replacement
    assert "4.2" in replacement


def test_summary_outsider_name_is_rejected() -> None:
    candidate = _candidate(1)
    assert check_summary("These choices are a good fit.", [candidate])
    assert not check_summary("Try McDonald's for a better match.", [candidate])
