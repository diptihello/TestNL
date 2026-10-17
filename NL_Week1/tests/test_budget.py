"""Unit tests for ``zomato_reco.domain.budget``."""

from __future__ import annotations

import pytest

from zomato_reco.config import load_settings
from zomato_reco.domain.budget import tier_for, tier_range_label
from zomato_reco.domain.models import Budget


@pytest.fixture
def settings(tmp_path):
    return load_settings(_env_file=tmp_path / "missing.env")


class TestTierFor:
    @pytest.mark.parametrize(
        ("cost", "expected"),
        [
            (300, Budget.low),
            (1, Budget.low),
            (301, Budget.medium),
            (500, Budget.medium),
            (501, Budget.high),
            (2000, Budget.high),
            (None, None),
        ],
    )
    def test_boundaries(self, settings, cost, expected) -> None:
        assert tier_for(cost, settings) is expected


class TestTierRangeLabel:
    def test_labels_use_settings(self, settings) -> None:
        assert "300" in tier_range_label(Budget.low, settings)
        assert "301" in tier_range_label(Budget.medium, settings) or "300" in tier_range_label(
            Budget.medium, settings
        )
        med = tier_range_label(Budget.medium, settings)
        assert "500" in med
        assert "INR" in med
        high = tier_range_label(Budget.high, settings)
        assert "500" in high
        assert "INR" in high

    def test_label_tracks_threshold_change(self, tmp_path) -> None:
        custom = load_settings(
            _env_file=tmp_path / "missing.env",
            budget_low_max=200,
            budget_medium_max=800,
        )
        assert "200" in tier_range_label(Budget.low, custom)
        assert "800" in tier_range_label(Budget.medium, custom)
