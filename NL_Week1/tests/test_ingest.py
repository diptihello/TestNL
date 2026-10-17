"""Tests for ``scripts/ingest.py`` (Phase 1, task 5).

Covers: zero or several CSV files, missing columns, reading only the needed
columns, the parse-rate and location gates, no final file left after a
simulated crash before the rename, the sample badge in the manifest, the
parquet schema, and the end-to-end ``main()`` with the network mocked out.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from zomato_reco.data.preprocess import CANONICAL_COLUMNS, CANONICAL_DTYPES, clean_frame

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "zomato_sample.csv"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SOURCE = {"repo": "test/repo", "revision": "abc123", "filename": "zomato.csv"}


@pytest.fixture(scope="module")
def ingest():
    """Import scripts/ingest.py as a module."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module("ingest")
    return importlib.reload(module)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH, dtype=str)


def _manifest(**overrides) -> dict:
    base = {
        "rating_parse_rate": 0.99,
        "rating_count": 99,
        "rating_numeric_looking": 100,
        "distinct_locations": 80,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_zero_csv_files(self, ingest) -> None:
        with mock.patch.object(ingest, "list_repo_files", return_value=["data.json", "README.md"]):
            with pytest.raises(ingest.IngestError, match="No CSV files") as info:
                ingest.discover_csv("test/repo")
        assert "data.json" in str(info.value)  # lists the files it found

    def test_multiple_csv_files(self, ingest) -> None:
        with mock.patch.object(ingest, "list_repo_files", return_value=["b.csv", "a.CSV", "x.md"]):
            with pytest.raises(ingest.IngestError, match="exactly one CSV") as info:
                ingest.discover_csv("test/repo")
        assert "a.CSV" in str(info.value) and "b.csv" in str(info.value)

    def test_single_csv_returns_name_and_commit(self, ingest) -> None:
        with (
            mock.patch.object(ingest, "list_repo_files", return_value=["README.md", "zomato.csv"]),
            mock.patch.object(ingest, "repo_info", return_value=SimpleNamespace(sha="deadbeef")),
        ):
            assert ingest.discover_csv("test/repo", "main") == ("zomato.csv", "deadbeef")

    def test_missing_sha_falls_back_to_revision(self, ingest) -> None:
        with (
            mock.patch.object(ingest, "list_repo_files", return_value=["zomato.csv"]),
            mock.patch.object(ingest, "repo_info", return_value=SimpleNamespace(sha=None)),
        ):
            assert ingest.discover_csv("test/repo", "v1") == ("zomato.csv", "v1")


# ---------------------------------------------------------------------------
# Reading the raw CSV
# ---------------------------------------------------------------------------


class TestReadRaw:
    def test_reads_only_needed_columns(self, ingest, tmp_path, raw) -> None:
        frame = raw.copy()
        frame["reviews_list"] = "[('Rated 5.0', 'long text')]"
        frame["menu_item"] = "[]"
        frame["phone"] = "080 1234"
        path = tmp_path / "wide.csv"
        frame.to_csv(path, index=False)

        result = ingest.read_raw(path)
        assert not {"reviews_list", "menu_item", "phone"} & set(result.columns)
        assert len(result) == len(raw)
        assert all(dtype == object for dtype in result.dtypes)  # read as strings

    def test_missing_required_columns_named(self, ingest, tmp_path, raw) -> None:
        path = tmp_path / "bad.csv"
        raw.drop(columns=["rate", "cuisines"]).to_csv(path, index=False)
        with pytest.raises(ingest.IngestError) as info:
            ingest.read_raw(path)
        message = str(info.value)
        assert "rate" in message and "cuisines" in message

    def test_optional_listed_in_type_may_be_absent(self, ingest, tmp_path, raw) -> None:
        path = tmp_path / "no_type.csv"
        raw.drop(columns=["listed_in(type)"], errors="ignore").to_csv(path, index=False)
        result = ingest.read_raw(path)
        assert "listed_in(type)" not in result.columns
        clean_frame(result)  # still cleans

    def test_unreadable_file(self, ingest, tmp_path) -> None:
        path = tmp_path / "empty.csv"
        path.write_bytes(b"")
        with pytest.raises(ingest.IngestError, match="Cannot read CSV"):
            ingest.read_raw(path)


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


class TestGates:
    def test_good_manifest_passes(self, ingest) -> None:
        ingest.check_gates(_manifest())

    def test_fixture_passes_rating_gate(self, ingest, raw) -> None:
        from zomato_reco.data.preprocess import build_manifest

        manifest = build_manifest(len(raw), clean_frame(raw), SOURCE)
        assert manifest["rating_parse_rate"] >= ingest.MIN_RATING_PARSE_RATE
        ingest.check_gates(manifest, min_locations=1)

    def test_low_rating_parse_rate_fails(self, ingest) -> None:
        with pytest.raises(ingest.IngestError, match="Rating parse rate"):
            ingest.check_gates(_manifest(rating_parse_rate=0.80))

    def test_rate_exactly_at_gate_passes(self, ingest) -> None:
        ingest.check_gates(_manifest(rating_parse_rate=ingest.MIN_RATING_PARSE_RATE))

    def test_too_few_locations_fails(self, ingest) -> None:
        with pytest.raises(ingest.IngestError, match="distinct locations"):
            ingest.check_gates(_manifest(distinct_locations=ingest.MIN_LOCATIONS - 1))

    def test_locations_exactly_at_gate_pass(self, ingest) -> None:
        ingest.check_gates(_manifest(distinct_locations=ingest.MIN_LOCATIONS))


# ---------------------------------------------------------------------------
# Atomic output
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_safe_rename_moves_file(self, ingest, tmp_path) -> None:
        src, dst = tmp_path / "a.tmp", tmp_path / "a.parquet"
        src.write_text("new")
        dst.write_text("old")
        ingest._safe_rename(src, dst)
        assert dst.read_text() == "new" and not src.exists()

    def test_safe_rename_retries_then_succeeds(self, ingest, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ingest, "_RENAME_DELAY_SECONDS", 0)
        src, dst = tmp_path / "a.tmp", tmp_path / "a.parquet"
        src.write_text("new")
        real_replace = Path.replace
        calls = {"n": 0}

        def flaky(self, target):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("file is open")
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", flaky)
        ingest._safe_rename(src, dst)
        assert calls["n"] == 3 and dst.read_text() == "new"

    def test_safe_rename_gives_up(self, ingest, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ingest, "_RENAME_DELAY_SECONDS", 0)
        monkeypatch.setattr(Path, "replace", mock.Mock(side_effect=PermissionError("locked")))
        src = tmp_path / "a.tmp"
        src.write_text("x")
        with pytest.raises(PermissionError):
            ingest._safe_rename(src, tmp_path / "a.parquet")

    def test_writes_parquet_and_manifest(self, ingest, tmp_path, raw) -> None:
        parquet = tmp_path / "out" / "restaurants.parquet"
        ingest.write_outputs(clean_frame(raw), {"k": "v"}, parquet)
        assert parquet.is_file()
        assert json.loads((parquet.parent / "manifest.json").read_text("utf-8")) == {"k": "v"}
        assert not list(parquet.parent.glob("*.tmp"))

    def test_crash_before_rename_leaves_no_final_file(self, ingest, tmp_path, raw, monkeypatch) -> None:
        monkeypatch.setattr(ingest, "_safe_rename", mock.Mock(side_effect=KeyboardInterrupt))
        parquet = tmp_path / "restaurants.parquet"
        with pytest.raises(KeyboardInterrupt):
            ingest.write_outputs(clean_frame(raw), {"k": "v"}, parquet)
        assert not parquet.exists()
        assert not (tmp_path / "manifest.json").exists()
        assert not list(tmp_path.glob("*.tmp"))  # temp files cleaned up

    def test_crash_between_renames_leaves_not_ready(self, ingest, tmp_path, raw, monkeypatch) -> None:
        parquet = tmp_path / "restaurants.parquet"
        manifest = tmp_path / "manifest.json"
        parquet.write_text("old parquet")
        manifest.write_text('{"old": true}')

        real = ingest._safe_rename
        calls = {"n": 0}

        def second_fails(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
            real(src, dst)

        monkeypatch.setattr(ingest, "_safe_rename", second_fails)
        with pytest.raises(OSError):
            ingest.write_outputs(clean_frame(raw), {"new": True}, parquet)
        # The old manifest must not be paired with the new parquet.
        assert not manifest.exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_parquet_schema(self, ingest, tmp_path, raw) -> None:
        parquet = tmp_path / "restaurants.parquet"
        ingest.write_outputs(clean_frame(raw), {}, parquet)
        loaded = pd.read_parquet(parquet)
        assert list(loaded.columns) == CANONICAL_COLUMNS
        for column, dtype in CANONICAL_DTYPES.items():
            assert str(loaded[column].dtype) == dtype, column
        assert all(len(v) >= 0 for v in loaded["cuisines"])  # list-like, not str
        assert not isinstance(loaded["cuisines"].iloc[0], str)


# ---------------------------------------------------------------------------
# Pipeline and sample badge
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pipeline_writes_outputs(self, ingest, tmp_path, raw) -> None:
        parquet = tmp_path / "restaurants.parquet"
        manifest = ingest.run_pipeline(raw, parquet, SOURCE, min_locations=1)
        assert parquet.is_file() and (tmp_path / "manifest.json").is_file()
        assert manifest["rows_before"] == len(raw)
        assert manifest["dataset_revision"] == "abc123"
        assert "sample_size" not in manifest

    def test_gate_failure_writes_nothing(self, ingest, tmp_path, raw) -> None:
        parquet = tmp_path / "restaurants.parquet"
        with pytest.raises(ingest.IngestError):
            ingest.run_pipeline(raw, parquet, SOURCE, min_locations=10_000)
        assert list(tmp_path.iterdir()) == []

    def test_sample_badge(self, ingest, tmp_path, raw) -> None:
        parquet = tmp_path / "restaurants.parquet"
        manifest = ingest.run_pipeline(raw, parquet, SOURCE, sample_size=20, min_locations=1)
        assert manifest["sample_size"] == 20
        assert manifest["rows_before"] == 20
        on_disk = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
        assert on_disk["sample_size"] == 20

    def test_sample_is_reproducible(self, ingest, raw) -> None:
        first = ingest.take_sample(raw, 20)
        second = ingest.take_sample(raw, 20)
        assert first.index.tolist() == second.index.tolist()
        assert first.index.is_monotonic_increasing

    def test_sample_larger_than_data_returns_all(self, ingest, raw) -> None:
        assert len(ingest.take_sample(raw, len(raw) + 100)) == len(raw)

    def test_ids_stable_across_runs(self, ingest, tmp_path, raw) -> None:
        a, b = tmp_path / "a" / "r.parquet", tmp_path / "b" / "r.parquet"
        ingest.run_pipeline(raw, a, SOURCE, min_locations=1)
        ingest.run_pipeline(raw, b, SOURCE, min_locations=1)
        assert pd.read_parquet(a)["id"].tolist() == pd.read_parquet(b)["id"].tolist()

    def test_summary_is_ascii(self, ingest, raw) -> None:
        from zomato_reco.data.preprocess import build_manifest

        manifest = build_manifest(len(raw), clean_frame(raw), SOURCE, sample_size=10)
        manifest["dataset_filename"] = "zomato_\u00e9t\u00e9_\u0939\u093f.csv"
        text = ingest.format_summary(manifest, Path("out") / "r.parquet")
        text.encode("ascii")  # must not raise
        assert "SAMPLE RUN" in text


# ---------------------------------------------------------------------------
# main() end to end, network mocked
# ---------------------------------------------------------------------------


class TestMain:
    def _patch_network(self, ingest, monkeypatch) -> None:
        monkeypatch.setattr(ingest, "discover_csv", lambda repo, rev: ("zomato.csv", "c0ffee"))
        monkeypatch.setattr(ingest, "download_csv", lambda repo, name, rev: FIXTURE_PATH)

    def test_main_sample_run(self, ingest, tmp_path, monkeypatch, capsys) -> None:
        self._patch_network(ingest, monkeypatch)
        code = ingest.main(["--out", str(tmp_path), "--sample", "30"])
        assert code == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
        assert manifest["sample_size"] == 30
        assert manifest["dataset_revision"] == "c0ffee"
        assert (tmp_path / "restaurants.parquet").is_file()
        assert "Ingestion complete" in capsys.readouterr().out

    def test_main_full_run_fails_location_gate_on_fixture(self, ingest, tmp_path, monkeypatch, capsys) -> None:
        self._patch_network(ingest, monkeypatch)
        assert ingest.main(["--out", str(tmp_path)]) == 1
        assert "Nothing was written" in capsys.readouterr().err
        assert not (tmp_path / "restaurants.parquet").exists()

    def test_main_discovery_error_exits_1(self, ingest, tmp_path, monkeypatch, capsys) -> None:
        def boom(repo, rev):
            raise ingest.IngestError("No CSV files found")

        monkeypatch.setattr(ingest, "discover_csv", boom)
        assert ingest.main(["--out", str(tmp_path)]) == 1
        assert "No CSV files found" in capsys.readouterr().err
        assert list(tmp_path.iterdir()) == []

    def test_main_network_error_exits_1(self, ingest, tmp_path, monkeypatch, capsys) -> None:
        def offline(repo, rev):
            raise ConnectionError("offline")

        monkeypatch.setattr(ingest, "discover_csv", offline)
        assert ingest.main(["--out", str(tmp_path)]) == 1
        assert "ConnectionError" in capsys.readouterr().err

    @pytest.mark.parametrize("bad", ["0", "-5"])
    def test_invalid_sample_rejected(self, ingest, bad) -> None:
        with pytest.raises(SystemExit):
            ingest.main(["--sample", bad])
