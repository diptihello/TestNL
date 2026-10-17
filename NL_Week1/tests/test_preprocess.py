"""Unit tests for ``zomato_reco.data.preprocess``.

Coverage targets: every parser including edge values, ``clean_frame`` on the
fixture CSV, id stability and uniqueness, location key casefolding, display
spelling, column correctness and the manifest statistics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from zomato_reco.data.preprocess import (
    CANONICAL_COLUMNS,
    CANONICAL_DTYPES,
    build_manifest,
    clean_frame,
    cost_parse_stats,
    make_id,
    normalize_key,
    parse_cost,
    parse_rating,
    parse_votes,
    parse_yes_no,
    rating_parse_stats,
    split_list,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "zomato_sample.csv"
FIXTURE_RAW_ROWS = 50
FIXTURE_CLEAN_ROWS = 43  # 50 - 1 empty name - 1 empty location - 5 duplicates
SOURCE = {"repo": "test/repo", "revision": "abc123", "filename": "data.csv"}


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, dtype=str)


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_frame(raw_df)


def _row(clean: pd.DataFrame, name: str) -> pd.Series:
    rows = clean[clean["name"] == name]
    assert len(rows) == 1, f"expected exactly one {name!r}, found {len(rows)}"
    return rows.iloc[0]


def _raw(**overrides: str | None) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "name": "Place",
        "address": "1 Road",
        "online_order": "Yes",
        "book_table": "No",
        "rate": "4.0/5",
        "votes": "10",
        "location": "Area",
        "rest_type": "Cafe",
        "dish_liked": "Coffee",
        "cuisines": "Cafe",
        "approx_cost(for two people)": "300",
        "listed_in(type)": "Delivery",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# parse_rating
# ---------------------------------------------------------------------------


class TestParseRating:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("4.1/5", 4.1),
            ("4/5", 4.0),
            ("4.1 /5", 4.1),
            ("4.1 / 5", 4.1),
            (" 4.1/5 ", 4.1),
            ("3.95/5", 4.0),
            ("0/5", 0.0),
            ("5/5", 5.0),
        ],
    )
    def test_valid(self, value, expected):
        assert parse_rating(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["NEW", "new", "-", "", "   ", None, float("nan"), "6/5", "5.1/5", "-1/5", "4.1", "great"],
    )
    def test_invalid_is_none(self, value):
        assert parse_rating(value) is None

    def test_pd_na(self):
        assert parse_rating(pd.NA) is None


# ---------------------------------------------------------------------------
# parse_cost
# ---------------------------------------------------------------------------


class TestParseCost:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("800", 800), ("1,200", 1200), ("800.0", 800), (" 800 ", 800), ("1 200", 1200)],
    )
    def test_valid(self, value, expected):
        assert parse_cost(value) == expected

    @pytest.mark.parametrize(
        "value", ["BBQ", "Rs 800", "abc", None, float("nan"), "", "0", "-100", "inf"]
    )
    def test_invalid_is_none(self, value):
        assert parse_cost(value) is None


# ---------------------------------------------------------------------------
# split_list
# ---------------------------------------------------------------------------


class TestSplitList:
    def test_basic(self):
        assert split_list("North Indian, Mughlai, Chinese") == [
            "North Indian",
            "Mughlai",
            "Chinese",
        ]

    def test_strip_and_dedup_case_insensitive(self):
        assert split_list("Italian, italian , ITALIAN") == ["Italian"]

    def test_drop_empties(self):
        assert split_list("A, , B,") == ["A", "B"]

    def test_title_case(self):
        assert split_list("south indian") == ["South Indian"]

    def test_keep_short_uppercase(self):
        assert split_list("BBQ") == ["BBQ"]

    def test_bbq_in_multi(self):
        assert split_list("BBQ, north indian") == ["BBQ", "North Indian"]

    def test_long_uppercase_is_title_cased(self):
        assert split_list("CONTINENTAL") == ["Continental"]

    def test_apostrophe_and_hyphen(self):
        assert split_list("chef's special, indo-chinese") == ["Chef's Special", "Indo-Chinese"]

    @pytest.mark.parametrize("value", [None, float("nan"), "", "  "])
    def test_missing(self, value):
        assert split_list(value) == []

    def test_limit(self):
        assert split_list("A, B, C, D, E, F", limit=5) == ["A", "B", "C", "D", "E"]

    def test_no_limit(self):
        assert len(split_list("A, B, C, D, E, F")) == 6


# ---------------------------------------------------------------------------
# parse_yes_no / parse_votes
# ---------------------------------------------------------------------------


class TestParseYesNo:
    @pytest.mark.parametrize("value", ["Yes", "yes", " YES "])
    def test_yes(self, value):
        assert parse_yes_no(value) is True

    @pytest.mark.parametrize("value", ["No", None, float("nan"), "maybe", "", "Y"])
    def test_not_yes(self, value):
        assert parse_yes_no(value) is False


class TestParseVotes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("775", 775),
            ("775.5", 775),
            ("2,000", 2000),
            (None, 0),
            (float("nan"), 0),
            ("", 0),
            ("abc", 0),
            ("-5", 0),
        ],
    )
    def test_values(self, value, expected):
        assert parse_votes(value) == expected


# ---------------------------------------------------------------------------
# normalize_key / make_id
# ---------------------------------------------------------------------------


class TestNormalizeKey:
    def test_casefold_and_strip(self):
        assert normalize_key("  Banashankari  ") == "banashankari"

    def test_casefold_not_lower(self):
        assert normalize_key("Straße") == "strasse"

    def test_missing(self):
        assert normalize_key(None) == ""


class TestMakeId:
    def test_deterministic(self):
        assert make_id("Jalsa", "942 21st Main Road") == make_id("Jalsa", "942 21st Main Road")

    def test_format(self):
        rid = make_id("Test", "Addr")
        assert rid.startswith("r_") and len(rid) == 12

    def test_known_value(self):
        # Pins the hash recipe so ids stay stable across releases.
        import hashlib

        expected = "r_" + hashlib.sha1(b"jalsa|addr").hexdigest()[:10]
        assert make_id(" Jalsa ", "ADDR") == expected

    def test_case_and_space_insensitive(self):
        assert make_id("Jalsa ", "Addr") == make_id("jalsa", " addr")

    def test_different_address_differs(self):
        assert make_id("Jalsa", "A") != make_id("Jalsa", "B")


# ---------------------------------------------------------------------------
# Parse-rate helpers
# ---------------------------------------------------------------------------


class TestParseStats:
    def test_rating_stats_ignore_unrated_markers(self):
        series = pd.Series(["4.1/5", "NEW", "-", None, "6/5", "3/5"])
        # numeric-looking: 4.1/5, 6/5, 3/5 ; parsed: 4.1/5, 3/5
        assert rating_parse_stats(series) == (3, 2)

    def test_cost_stats(self):
        series = pd.Series(["800", "1,200", "", None, "abc"])
        assert cost_parse_stats(series) == (3, 2)


# ---------------------------------------------------------------------------
# clean_frame on the fixture CSV
# ---------------------------------------------------------------------------


class TestCleanFrame:
    def test_canonical_columns_in_order(self, clean_df):
        assert list(clean_df.columns) == CANONICAL_COLUMNS

    def test_dtypes(self, clean_df):
        actual = {c: str(clean_df[c].dtype) for c in CANONICAL_COLUMNS}
        assert actual == CANONICAL_DTYPES

    @pytest.mark.parametrize(
        "column", ["reviews_list", "phone", "menu_item", "url", "listed_in(type)", "rate"]
    )
    def test_dropped_source_columns(self, clean_df, column):
        assert column not in clean_df.columns

    def test_row_count_after_dedupe(self, raw_df, clean_df):
        assert len(raw_df) == FIXTURE_RAW_ROWS
        assert len(clean_df) == FIXTURE_CLEAN_ROWS

    def test_empty_name_and_location_dropped(self, clean_df):
        assert (clean_df["name"].str.len() > 0).all()
        assert (clean_df["location"].str.len() > 0).all()
        assert "Nameless Cafe" not in set(clean_df["name"])

    def test_dedupe_keeps_higher_votes(self, clean_df):
        assert _row(clean_df, "Jalsa")["votes"] == 775

    def test_dedupe_higher_votes_wins_even_when_later(self, clean_df):
        # The second Spice Junction row has more votes and a different cost.
        row = _row(clean_df, "Spice Junction")
        assert row["votes"] == 460
        assert row["cost_for_two"] == 550

    def test_vote_tie_keeps_first_row(self, clean_df):
        assert _row(clean_df, "Tie Cafe")["cost_for_two"] == 400

    def test_case_and_trailing_space_names_are_one_restaurant(self, clean_df):
        keys = clean_df["name"].str.strip().str.casefold()
        assert (keys == "jalsa").sum() == 1

    def test_display_keeps_original_spelling(self, clean_df):
        assert "Jalsa" in set(clean_df["name"])
        assert "jalsa" not in set(clean_df["name"])

    def test_ids_unique(self, clean_df):
        assert clean_df["id"].is_unique

    def test_ids_match_recipe(self, clean_df):
        for _, row in clean_df.iterrows():
            assert row["id"] == make_id(row["name"], row["address"])

    def test_ids_and_output_stable_across_runs(self, raw_df):
        first = clean_frame(raw_df)
        second = clean_frame(raw_df)
        pd.testing.assert_frame_equal(first, second)

    def test_output_independent_of_input_order(self, raw_df):
        shuffled = raw_df.sample(frac=1, random_state=7)
        # Different input order may change which tie wins, but ids must match.
        assert set(clean_frame(shuffled)["id"]) == set(clean_frame(raw_df)["id"])

    def test_location_key_casefolded(self, clean_df):
        assert (clean_df["location_key"] == clean_df["location"].str.casefold()).all()

    def test_one_display_per_location_key(self, clean_df):
        assert (clean_df.groupby("location_key")["location"].nunique() == 1).all()

    def test_most_frequent_spelling_wins(self, clean_df):
        btm = clean_df[clean_df["location_key"] == "btm layout"]
        assert set(btm["location"]) == {"BTM Layout"}
        assert len(btm) == 3

    def test_whitespace_padded_location_stripped(self, clean_df):
        assert _row(clean_df, "Spice Junction")["location"] == "Indiranagar"
        assert (clean_df["location"] == clean_df["location"].str.strip()).all()

    def test_alphabetical_tiebreak_for_spelling(self):
        raw = pd.DataFrame(
            [
                _raw(name="A", location="hsr layout"),
                _raw(name="B", location="HSR Layout"),
            ]
        )
        assert set(clean_frame(raw)["location"]) == {"HSR Layout"}

    def test_cuisines_are_lists(self, clean_df):
        assert all(isinstance(v, list) for v in clean_df["cuisines"])
        assert _row(clean_df, "Jalsa")["cuisines"] == ["North Indian", "Mughlai", "Chinese"]

    def test_null_cuisine_is_empty_list(self, clean_df):
        assert _row(clean_df, "Plain Eats")["cuisines"] == []

    def test_acronym_cuisine_kept_uppercase(self, clean_df):
        assert _row(clean_df, "The Grill House")["cuisines"] == ["BBQ", "North Indian"]

    def test_lowercase_cuisine_title_cased(self, clean_df):
        assert _row(clean_df, "Btm Bites")["cuisines"] == ["Fast Food", "Rolls"]

    def test_dish_liked_capped_at_5(self, clean_df):
        assert all(len(v) <= 5 for v in clean_df["dish_liked"])
        assert len(_row(clean_df, "Jalsa")["dish_liked"]) == 5

    def test_empty_dish_liked(self, clean_df):
        assert _row(clean_df, "Plain Eats")["dish_liked"] == []

    def test_cost_with_comma(self, clean_df):
        assert _row(clean_df, "Royal Treat")["cost_for_two"] == 1200

    def test_cost_float_string(self, clean_df):
        assert _row(clean_df, "Pasta Palace")["cost_for_two"] == 800

    def test_null_cost_is_na(self, clean_df):
        assert pd.isna(_row(clean_df, "Café Udupi Ruchi")["cost_for_two"])

    def test_spaced_rating(self, clean_df):
        assert _row(clean_df, "Pasta Palace")["rating"] == 4.1

    def test_new_rating_is_missing(self, clean_df):
        row = _row(clean_df, "Cafe Corner")
        assert math.isnan(row["rating"])
        assert row["votes"] == 0  # empty votes become 0

    def test_dash_rating_is_missing(self, clean_df):
        assert math.isnan(_row(clean_df, "Chai Point")["rating"])

    def test_votes_with_comma(self, clean_df):
        assert _row(clean_df, "Empire Restaurant")["votes"] == 2000

    def test_booleans(self, clean_df):
        row = _row(clean_df, "Royal Treat")
        assert row["online_order"] is False or row["online_order"] == False  # noqa: E712
        assert bool(row["book_table"]) is True

    def test_non_ascii_name_preserved(self, clean_df):
        assert "Café Udupi Ruchi" in set(clean_df["name"])

    def test_empty_rest_type_is_none(self):
        clean = clean_frame(pd.DataFrame([_raw(rest_type="  ")]))
        assert clean.iloc[0]["rest_type"] is None

    def test_listed_in_type_optional(self):
        row = _raw()
        row.pop("listed_in(type)")
        assert len(clean_frame(pd.DataFrame([row]))) == 1

    def test_missing_required_column_raises(self):
        row = _raw()
        row.pop("cuisines")
        with pytest.raises(KeyError, match="cuisines"):
            clean_frame(pd.DataFrame([row]))

    def test_id_collision_names_both(self, monkeypatch):
        import zomato_reco.data.preprocess as pre

        monkeypatch.setattr(pre, "make_id", lambda name, address: "r_same")
        raw = pd.DataFrame([_raw(name="First"), _raw(name="Second", address="2 Road")])
        with pytest.raises(ValueError, match="First") as excinfo:
            pre.clean_frame(raw)
        assert "Second" in str(excinfo.value)

    def test_stats_attached(self, clean_df):
        stats = clean_df.attrs["stats"]
        assert stats["rows_in"] == FIXTURE_RAW_ROWS
        assert stats["rows_out"] == FIXTURE_CLEAN_ROWS
        assert stats["dropped_empty_name"] == 1
        assert stats["dropped_empty_location"] == 1
        assert stats["duplicates_removed"] == 5

    def test_input_not_mutated(self, raw_df):
        before = raw_df.copy()
        clean_frame(raw_df)
        pd.testing.assert_frame_equal(raw_df, before)

    def test_parquet_round_trip(self, clean_df, tmp_path):
        path = tmp_path / "r.parquet"
        frame = clean_df.copy()
        frame.attrs = {}
        frame.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == CANONICAL_COLUMNS
        assert len(loaded) == FIXTURE_CLEAN_ROWS
        assert str(loaded["cost_for_two"].dtype) == "Int64"
        assert list(loaded.loc[loaded["name"] == "Jalsa", "cuisines"].iloc[0]) == [
            "North Indian",
            "Mughlai",
            "Chinese",
        ]


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    @pytest.fixture()
    def manifest(self, clean_df):
        return build_manifest(raw_rows=FIXTURE_RAW_ROWS, clean=clean_df, source=SOURCE)

    def test_counts(self, manifest):
        assert manifest["rows_before"] == FIXTURE_RAW_ROWS
        assert manifest["rows_after"] == FIXTURE_CLEAN_ROWS
        assert manifest["dropped_rows"] == FIXTURE_RAW_ROWS - FIXTURE_CLEAN_ROWS
        assert manifest["dropped"] == {"empty_name": 1, "empty_location": 1, "duplicates": 5}

    def test_rating_parse_rate_uses_raw_numeric_values(self, manifest):
        # 48 raw ratings contain a digit and all of them parse; NEW and - are excluded.
        assert manifest["rating_numeric_looking"] == 48
        assert manifest["rating_count"] == 48
        assert manifest["rating_parse_rate"] == 1.0

    def test_cost_parse_rate(self, manifest):
        assert manifest["cost_present"] == 49
        assert manifest["cost_parse_rate"] == 1.0

    def test_cost_percentiles(self, manifest):
        pct = manifest["cost_percentiles"]
        assert list(pct) == ["p10", "p33", "p50", "p66", "p90"]
        assert pct["p10"] <= pct["p33"] <= pct["p50"] <= pct["p66"] <= pct["p90"]
        assert all(isinstance(v, int) for v in pct.values())

    def test_provenance(self, manifest):
        assert manifest["dataset_repo"] == "test/repo"
        assert manifest["dataset_revision"] == "abc123"
        assert manifest["dataset_filename"] == "data.csv"
        assert manifest["generated_at"].endswith("+00:00")
        assert manifest["schema_columns"] == CANONICAL_COLUMNS

    def test_distinct_locations(self, manifest, clean_df):
        assert manifest["distinct_locations"] == clean_df["location_key"].nunique()

    def test_json_serialisable(self, manifest):
        import json

        json.dumps(manifest)

    def test_sample_size(self, clean_df):
        manifest = build_manifest(FIXTURE_RAW_ROWS, clean_df, SOURCE, sample_size=50)
        assert manifest["sample_size"] == 50

    def test_no_sample_key_when_none(self, manifest):
        assert "sample_size" not in manifest
