"""Unit tests for ``zomato_reco.domain.models``."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from zomato_reco.domain.models import (
    Budget,
    Candidate,
    Preferences,
    Restaurant,
)


class TestPreferences:
    def test_minimal_valid(self) -> None:
        prefs = Preferences(location="Banashankari", budget=Budget.low, min_rating=4.0)
        assert prefs.cuisine is None
        assert prefs.extra_preferences is None

    def test_empty_cuisine_becomes_none(self) -> None:
        prefs = Preferences(
            location="Banashankari",
            budget="medium",
            min_rating=3.5,
            cuisine="   ",
        )
        assert prefs.cuisine is None

    def test_whitespace_cuisine_becomes_none(self) -> None:
        prefs = Preferences(
            location="Banashankari", budget=Budget.high, min_rating=0, cuisine=""
        )
        assert prefs.cuisine is None

    def test_cuisine_trimmed_and_joined(self) -> None:
        prefs = Preferences(
            location="BTM",
            budget=Budget.low,
            min_rating=4.0,
            cuisine=" Italian , Chinese ",
        )
        assert prefs.cuisine == "Italian, Chinese"

    def test_cuisine_rejects_more_than_three(self) -> None:
        with pytest.raises(ValidationError):
            Preferences(
                location="BTM",
                budget=Budget.low,
                min_rating=4.0,
                cuisine="a, b, c, d",
            )

    def test_cuisine_rejects_long_token(self) -> None:
        with pytest.raises(ValidationError):
            Preferences(
                location="BTM",
                budget=Budget.low,
                min_rating=4.0,
                cuisine="x" * 41,
            )

    def test_notes_truncated_at_300(self) -> None:
        prefs = Preferences(
            location="BTM",
            budget=Budget.low,
            min_rating=4.0,
            extra_preferences="a" * 350,
        )
        assert prefs.extra_preferences is not None
        assert len(prefs.extra_preferences) == 300

    def test_notes_control_chars_stripped_before_cap(self) -> None:
        raw = "hello\x00\x01world" + ("x" * 295)
        prefs = Preferences(
            location="BTM",
            budget=Budget.low,
            min_rating=4.0,
            extra_preferences=raw,
        )
        assert prefs.extra_preferences is not None
        assert "\x00" not in prefs.extra_preferences
        assert prefs.extra_preferences.startswith("helloworld")
        assert len(prefs.extra_preferences) == 300

    def test_blank_notes_become_none(self) -> None:
        prefs = Preferences(
            location="BTM",
            budget=Budget.low,
            min_rating=4.0,
            extra_preferences="  \n\t  ",
        )
        assert prefs.extra_preferences is None

    @pytest.mark.parametrize("rating", [-0.1, 5.1, 6.0])
    def test_rating_bounds(self, rating: float) -> None:
        with pytest.raises(ValidationError):
            Preferences(location="BTM", budget=Budget.low, min_rating=rating)

    def test_rating_edges_ok(self) -> None:
        Preferences(location="BTM", budget=Budget.low, min_rating=0.0)
        Preferences(location="BTM", budget=Budget.low, min_rating=5.0)

    def test_invalid_budget_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Preferences(location="BTM", budget="luxury", min_rating=4.0)  # type: ignore[arg-type]

    def test_empty_location_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Preferences(location="  ", budget=Budget.low, min_rating=4.0)


class TestRestaurantAndCandidate:
    def test_from_row_and_prompt_line(self) -> None:
        row = {
            "id": "r_abc",
            "name": "Café München",
            "address": "1 Road",
            "location": "BTM",
            "location_key": "btm",
            "cuisines": ["North Indian", "Chinese"],
            "cost_for_two": 400,
            "rating": 4.2,
            "votes": 100,
            "rest_type": "Casual Dining",
            "dish_liked": ["Pasta", "Biryani"],
            "online_order": True,
            "book_table": False,
        }
        cand = Candidate.from_row(row)
        line = cand.to_prompt_line()
        assert " " not in line.split(":")[0]  # compact separators overall
        assert ": " not in line
        parsed = json.loads(line)
        assert parsed["id"] == "r_abc"
        assert parsed["name"] == "Café München"
        assert "\\u" not in line  # ensure_ascii=False
        assert parsed["rating"] == 4.2

    def test_null_numbers_as_json_null(self) -> None:
        row = {
            "id": "r_x",
            "name": "X",
            "address": "",
            "location": "BTM",
            "location_key": "btm",
            "cuisines": [],
            "cost_for_two": None,
            "rating": None,
            "votes": 0,
            "rest_type": None,
            "dish_liked": [],
            "online_order": False,
            "book_table": False,
        }
        line = Candidate.from_row(row).to_prompt_line()
        assert "NaN" not in line
        assert "null" in line
        Restaurant.from_row(row)
