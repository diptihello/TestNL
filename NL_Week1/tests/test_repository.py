"""Unit tests for ``zomato_reco.data.repository``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from zomato_reco.data.preprocess import CANONICAL_COLUMNS, clean_frame
from zomato_reco.data.repository import RepositoryError, load_repository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "zomato_sample.csv"


def _write_repo(tmp_path: Path, df: pd.DataFrame, manifest: dict | None = None) -> Path:
    path = tmp_path / "restaurants.parquet"
    df.to_parquet(path, index=False)
    man = manifest or {
        "rows_after": len(df),
        "schema_columns": list(CANONICAL_COLUMNS),
        "distinct_locations": int(df["location"].nunique()),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return path


@pytest.fixture
def fixture_clean() -> pd.DataFrame:
    raw = pd.read_csv(FIXTURE_PATH, dtype=str)
    return clean_frame(raw)


class TestLoadRepository:
    def test_loads_lists_and_locations(self, tmp_path, fixture_clean) -> None:
        path = _write_repo(tmp_path, fixture_clean)
        repo = load_repository(path)
        locs = repo.locations()
        assert locs == sorted(set(locs))
        assert len(locs) == len(set(locs))
        assert locs
        frame = repo.all()
        assert isinstance(frame.iloc[0]["cuisines"], list)
        assert isinstance(frame.iloc[0]["dish_liked"], list)
        assert "cuisines_key" in frame.columns
        assert repo.manifest()["rows_after"] == len(fixture_clean)

    def test_by_ids_found_once(self, tmp_path, fixture_clean) -> None:
        path = _write_repo(tmp_path, fixture_clean)
        repo = load_repository(path)
        first = fixture_clean.iloc[0]["id"]
        second = fixture_clean.iloc[1]["id"]
        result = repo.by_ids([second, first, second, "r_missing"])
        assert list(result["id"]) == [second, first]
        assert repo.by_ids([]).empty

    def test_older_schema_raises(self, tmp_path, fixture_clean) -> None:
        bad = fixture_clean.drop(columns=["votes"])
        path = _write_repo(tmp_path, bad)
        with pytest.raises(RepositoryError, match="missing"):
            load_repository(path)

    def test_corrupt_file_raises(self, tmp_path) -> None:
        path = tmp_path / "restaurants.parquet"
        path.write_text("not a parquet", encoding="utf-8")
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RepositoryError, match="could not be read"):
            load_repository(path)

    def test_empty_location_list_raises(self, tmp_path, fixture_clean) -> None:
        empty = fixture_clean.iloc[0:0].copy()
        path = _write_repo(tmp_path, empty)
        with pytest.raises(RepositoryError, match="no locations"):
            load_repository(path)

    def test_missing_manifest_raises(self, tmp_path, fixture_clean) -> None:
        path = tmp_path / "restaurants.parquet"
        fixture_clean.to_parquet(path, index=False)
        with pytest.raises(RepositoryError, match="Manifest"):
            load_repository(path)

    def test_all_returns_copy(self, tmp_path, fixture_clean) -> None:
        path = _write_repo(tmp_path, fixture_clean)
        repo = load_repository(path)
        frame = repo.all()
        frame.loc[frame.index[0], "name"] = "MUTATED"
        assert repo.all().iloc[0]["name"] != "MUTATED"
