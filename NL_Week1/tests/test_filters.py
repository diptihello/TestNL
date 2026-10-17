"""Unit tests for ``zomato_reco.domain.filters``."""

from __future__ import annotations

import pandas as pd
import pytest

from zomato_reco.config import load_settings
from zomato_reco.domain.filters import apply, build_empty_result, suggest
from zomato_reco.domain.models import Budget, Preferences


@pytest.fixture
def settings(tmp_path):
    return load_settings(_env_file=tmp_path / "missing.env")


def _row(
    *,
    rid: str,
    name: str,
    location: str,
    cuisines: list[str],
    cost: int | None,
    rating: float | None,
    votes: int = 10,
) -> dict:
    return {
        "id": rid,
        "name": name,
        "address": f"{name} addr",
        "location": location,
        "location_key": location.casefold(),
        "cuisines": cuisines,
        "cuisines_key": " ".join(cuisines).casefold(),
        "cost_for_two": cost,
        "rating": rating,
        "votes": votes,
        "rest_type": "Casual Dining",
        "dish_liked": [],
        "online_order": True,
        "book_table": False,
    }


@pytest.fixture
def frame() -> pd.DataFrame:
    """20-row in-memory frame covering filter edge cases."""
    rows = [
        # Banashankari - varied
        _row(rid="r01", name="A", location="Banashankari", cuisines=["North Indian"], cost=250, rating=4.5),
        _row(rid="r02", name="B", location="Banashankari", cuisines=["Chinese"], cost=400, rating=4.0),
        _row(rid="r03", name="C", location="Banashankari", cuisines=["Italian"], cost=600, rating=3.5),
        _row(rid="r04", name="D", location="Banashankari", cuisines=["Indo-Chinese"], cost=350, rating=4.2),
        _row(rid="r05", name="E", location="Banashankari", cuisines=["Steak"], cost=300, rating=4.8),
        _row(rid="r06", name="F", location="Banashankari", cuisines=["Cafe"], cost=200, rating=None),
        _row(rid="r07", name="G", location="Banashankari", cuisines=["Italian", "Chinese"], cost=450, rating=4.1),
        _row(rid="r08", name="H", location="Banashankari", cuisines=["C++ Special"], cost=280, rating=3.8),
        _row(rid="r09", name="I", location="Banashankari", cuisines=["(Experimental)"], cost=500, rating=4.0),
        _row(rid="r10", name="J", location="Banashankari", cuisines=["Tea"], cost=150, rating=3.9),
        # BTM
        _row(rid="r11", name="K", location="BTM", cuisines=["South Indian"], cost=200, rating=4.3),
        _row(rid="r12", name="L", location="BTM", cuisines=["Chinese"], cost=800, rating=4.6),
        _row(rid="r13", name="M", location="BTM", cuisines=["Italian"], cost=400, rating=None),
        _row(rid="r14", name="N", location="BTM", cuisines=["North Indian"], cost=None, rating=4.0),
        _row(rid="r15", name="O", location="BTM", cuisines=["Fast Food"], cost=300, rating=3.0),
        # Jayanagar
        _row(rid="r16", name="P", location="Jayanagar", cuisines=["Chinese"], cost=350, rating=4.9, votes=5),
        _row(rid="r17", name="Q", location="Jayanagar", cuisines=["Italian"], cost=550, rating=4.4),
        _row(rid="r18", name="R", location="Jayanagar", cuisines=["North Indian"], cost=250, rating=4.0),
        _row(rid="r19", name="S", location="Jayanagar", cuisines=["Mexican"], cost=420, rating=3.2),
        _row(rid="r20", name="T", location="Jayanagar", cuisines=["Chinese"], cost=480, rating=4.7),
    ]
    return pd.DataFrame(rows)


def _prefs(**kwargs) -> Preferences:
    base = {
        "location": "Banashankari",
        "budget": Budget.medium,
        "min_rating": 4.0,
    }
    base.update(kwargs)
    return Preferences(**base)


class TestApplySteps:
    def test_location_case_insensitive(self, frame, settings) -> None:
        filtered, counts = apply(frame, _prefs(location="banashankari"), settings)
        assert counts["after_location"] == 10
        assert len(filtered) <= 10

    def test_null_rating_kept_only_when_min_zero(self, frame, settings) -> None:
        _, counts_hi = apply(frame, _prefs(min_rating=4.0, budget=Budget.low), settings)
        assert counts_hi["unrated_hidden"] == 1  # F is unrated in Banashankari
        filtered0, counts0 = apply(
            frame, _prefs(min_rating=0.0, budget=Budget.low, cuisine=None), settings
        )
        assert counts0["unrated_hidden"] == 0
        assert "r06" in set(filtered0["id"])

    def test_rating_at_boundary_included(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=4.0, cuisine="Chinese"),
            settings,
        )
        # B has Chinese, cost 400, rating 4.0
        assert "r02" in set(filtered["id"])

    def test_step_counts(self, frame, settings) -> None:
        _, counts = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=4.0, cuisine="Italian"),
            settings,
        )
        assert counts["after_location"] == 10
        assert counts["after_rating"] < counts["after_location"] or counts["after_rating"] == 9
        assert counts["after_budget"] <= counts["after_rating"]
        assert counts["after_cuisine"] <= counts["after_budget"]
        assert "unrated_hidden" in counts


class TestCuisineMatching:
    def test_tea_does_not_match_steak(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.low, min_rating=0, cuisine="Tea"),
            settings,
        )
        ids = set(filtered["id"])
        assert "r10" in ids
        assert "r05" not in ids

    def test_chinese_matches_indo_chinese(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=0, cuisine="Chinese"),
            settings,
        )
        ids = set(filtered["id"])
        assert "r04" in ids  # Indo-Chinese
        assert "r02" in ids

    def test_multi_cuisine_any_match(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=0, cuisine="Italian, Chinese"),
            settings,
        )
        ids = set(filtered["id"])
        assert "r02" in ids or "r07" in ids
        assert "r07" in ids  # Italian + Chinese, cost 450

    def test_regex_chars_do_not_raise(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.low, min_rating=0, cuisine="C++"),
            settings,
        )
        assert "r08" in set(filtered["id"])
        filtered2, _ = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=0, cuisine="("),
            settings,
        )
        # literal "(" should match "(Experimental)" via whole-word-ish boundary
        assert isinstance(filtered2, pd.DataFrame)

    def test_case_insensitive_cuisine(self, frame, settings) -> None:
        filtered, _ = apply(
            frame,
            _prefs(budget=Budget.medium, min_rating=0, cuisine="ITALIAN"),
            settings,
        )
        assert "r07" in set(filtered["id"])


class TestSuggest:
    def test_ordered_nonempty_capped(self, frame, settings) -> None:
        # Empty because rating is too strict; lowering it (or other single
        # changes) should yield matches in Banashankari.
        prefs = _prefs(
            budget=Budget.medium,
            min_rating=5.0,
            cuisine=None,
        )
        filtered, _ = apply(frame, prefs, settings)
        assert filtered.empty
        hints = suggest(frame, prefs, settings)
        assert hints
        assert len(hints) <= 4
        assert all(h.would_match >= 1 for h in hints)
        # Smallest rating drop should come first among rating hints
        rating_hints = [h for h in hints if h.field == "min_rating"]
        if len(rating_hints) >= 2:
            assert rating_hints[0].value > rating_hints[1].value

    def test_no_rating_hint_below_3(self, frame, settings) -> None:
        prefs = _prefs(min_rating=3.5, budget=Budget.high, cuisine="Mexican")
        hints = suggest(frame, prefs, settings)
        for h in hints:
            if h.field == "min_rating":
                assert float(h.value) >= 3.0

    def test_no_rating_hint_when_already_zero(self, frame, settings) -> None:
        prefs = _prefs(min_rating=0.0, budget=Budget.high, cuisine="Mexican")
        hints = suggest(frame, prefs, settings)
        assert all(h.field != "min_rating" for h in hints)

    def test_empty_when_location_has_no_rows(self, frame, settings) -> None:
        prefs = _prefs(location="Delhi", min_rating=4.0)
        assert suggest(frame, prefs, settings) == []

    def test_closest_change_reason_when_no_single_helps(self, frame, settings) -> None:
        # Location with rows, but every single relaxation still yields 0:
        # Banashankari + cuisine that doesn't exist + extreme combo where
        # dropping cuisine / changing budget / lowering rating alone isn't enough
        # because we need a cuisine that exists nowhere in Banashankari AND
        # constraints that stay empty.
        # Build a tiny dedicated frame where only a *double* change helps.
        rows = [
            _row(
                rid="only",
                name="Only",
                location="Solo",
                cuisines=["Thai"],
                cost=200,
                rating=3.5,
            ),
        ]
        tiny = pd.DataFrame(rows)
        prefs = Preferences(
            location="Solo",
            budget=Budget.high,  # only row is low
            min_rating=5.0,  # only row is 3.5
            cuisine="Mexican",  # only row is Thai
        )
        filtered, counts = apply(tiny, prefs, settings)
        assert filtered.empty
        hints = suggest(tiny, prefs, settings)
        assert hints == []
        empty = build_empty_result(tiny, prefs, settings, counts)
        assert empty.relaxations == []
        assert "closest" in empty.reason.lower() or "combination" in empty.reason.lower()
        assert empty.reason

    def test_unrated_hidden_count(self, frame, settings) -> None:
        _, counts = apply(frame, _prefs(min_rating=4.0), settings)
        assert counts["unrated_hidden"] == 1
