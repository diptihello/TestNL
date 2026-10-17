"""Unit tests for ``zomato_reco.domain.ranking``."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from zomato_reco.config import load_settings
from zomato_reco.domain.models import Budget, Candidate, Preferences
from zomato_reco.domain.ranking import fallback_recommendations, select


@pytest.fixture
def settings(tmp_path):
    return load_settings(_env_file=tmp_path / "missing.env")


def _row(**kwargs) -> dict:
    base = {
        "id": "r_x",
        "name": "Place",
        "address": "1 Road",
        "location": "BTM",
        "location_key": "btm",
        "cuisines": ["North Indian"],
        "cost_for_two": 400,
        "rating": 4.0,
        "votes": 100,
        "rest_type": "Casual Dining",
        "dish_liked": ["Pasta"],
        "online_order": False,
        "book_table": False,
    }
    base.update(kwargs)
    return base


def _prefs(**kwargs) -> Preferences:
    base = {
        "location": "BTM",
        "budget": Budget.medium,
        "min_rating": 0.0,
    }
    base.update(kwargs)
    return Preferences(**base)


class TestSelectOrdering:
    def test_low_votes_high_rating_ranks_below(self, settings) -> None:
        # Extra mid-rated rows keep the filtered-set mean near 4.0 so the
        # vote-aware score prefers 4.6 with 3,000 votes over 5.0 with 3.
        df = pd.DataFrame(
            [
                _row(id="r_hi", name="Flashy", rating=5.0, votes=3),
                _row(id="r_lo", name="Steady", rating=4.6, votes=3000),
                _row(id="r_a", name="A", rating=3.8, votes=100),
                _row(id="r_b", name="B", rating=3.9, votes=100),
                _row(id="r_c", name="C", rating=4.0, votes=100),
                _row(id="r_d", name="D", rating=4.1, votes=100),
            ]
        )
        cands, total = select(df, _prefs(), settings)
        assert total == 6
        ids = [c.id for c in cands]
        assert ids.index("r_lo") < ids.index("r_hi")

    def test_unrated_sort_last(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r_new", name="New Place", rating=None, votes=0),
                _row(id="r_ok", name="Rated", rating=3.5, votes=50),
            ]
        )
        cands, _ = select(df, _prefs(), settings)
        assert cands[-1].id == "r_new"
        assert cands[0].id == "r_ok"

    def test_cap_at_max_candidates(self, settings) -> None:
        rows = [
            _row(id=f"r_{i:02d}", name=f"Place {i}", rating=4.0 + (i % 5) * 0.1, votes=100 + i)
            for i in range(20)
        ]
        df = pd.DataFrame(rows)
        cands, total = select(df, _prefs(), settings)
        assert total == 20
        assert len(cands) == settings.max_candidates

    def test_chain_cap(self, settings) -> None:
        rows = [
            _row(id=f"r_{i}", name="Domino's Pizza", rating=4.5, votes=1000 - i, address=f"{i} St")
            for i in range(5)
        ]
        rows.append(_row(id="r_other", name="Other Cafe", rating=4.0, votes=10))
        df = pd.DataFrame(rows)
        cands, _ = select(df, _prefs(), settings)
        domino = [c for c in cands if c.restaurant.name == "Domino's Pizza"]
        assert len(domino) <= settings.max_per_chain
        assert any(c.id == "r_other" for c in cands)

    def test_tie_break_name_then_id(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r_b", name="Beta", rating=4.5, votes=100),
                _row(id="r_a", name="Alpha", rating=4.5, votes=100),
            ]
        )
        cands, _ = select(df, _prefs(), settings)
        assert [c.id for c in cands] == ["r_a", "r_b"]


class TestKeywordBoost:
    def test_delivery_boosts_online_order(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r_hi", name="High Rated", rating=4.8, votes=500, online_order=False),
                _row(id="r_del", name="Delivery Spot", rating=4.0, votes=50, online_order=True),
            ]
        )
        cands, _ = select(df, _prefs(extra_preferences="need delivery tonight"), settings)
        assert cands[0].id == "r_del"

    def test_vegetable_does_not_trigger_table(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r_hi", name="High", rating=4.8, votes=500, book_table=False),
                _row(id="r_book", name="Bookable", rating=4.0, votes=50, book_table=True),
            ]
        )
        cands, _ = select(df, _prefs(extra_preferences="love vegetable dishes"), settings)
        assert cands[0].id == "r_hi"

    def test_negation_skips_boost(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r_hi", name="High", rating=4.8, votes=500, online_order=False),
                _row(id="r_on", name="Online", rating=4.0, votes=50, online_order=True),
            ]
        )
        cands, _ = select(
            df, _prefs(extra_preferences="no online ordering please"), settings
        )
        assert cands[0].id == "r_hi"


class TestPromptLine:
    def test_long_row_under_400_no_nan_unicode(self, settings) -> None:
        row = _row(
            id="r_long",
            name="あ" * 100 + " Café",
            dish_liked=["D" * 60, "E" * 60, "F" * 60, "G" * 60, "H" * 60],
            cuisines=["North Indian", "Chinese", "Continental", "Mughlai"],
            rating=None,
            cost_for_two=None,
        )
        line = Candidate.from_row(row).to_prompt_line()
        assert len(line) < 400
        assert "NaN" not in line
        assert "\\u" not in line
        parsed = json.loads(line)
        assert parsed["rating"] is None
        assert parsed["cost_for_two"] is None
        assert len(parsed["name"]) <= 80


class TestFallback:
    def test_fallback_order_and_no_explanation(self, settings) -> None:
        df = pd.DataFrame(
            [
                _row(id="r1", name="A", rating=4.9, votes=200),
                _row(id="r2", name="B", rating=4.8, votes=150),
                _row(id="r3", name="C", rating=4.7, votes=100),
                _row(id="r4", name="D", rating=4.6, votes=90),
                _row(id="r5", name="E", rating=4.5, votes=80),
                _row(id="r6", name="F", rating=4.4, votes=70),
            ]
        )
        cands, _ = select(df, _prefs(), settings)
        recs = fallback_recommendations(cands, settings)
        assert len(recs) == settings.max_recommendations
        assert [r.rank for r in recs] == list(range(1, len(recs) + 1))
        assert all(r.explanation is None for r in recs)
        assert [r.restaurant.id for r in recs] == [c.id for c in cands[: len(recs)]]
