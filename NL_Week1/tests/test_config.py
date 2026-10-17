"""Tests for zomato_reco.config and zomato_reco.logging_setup (Phase 0, task 10).

Every test builds settings with ``_env_file=None`` and a scrubbed environment so
a developer's real ``.env`` or shell variables can never leak into the result.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from zomato_reco.config import (
    MAX_CANDIDATES_CEILING,
    PROJECT_ROOT,
    ConfigError,
    Settings,
    get_settings,
    load_settings,
    sanitize_api_key,
)
from zomato_reco.logging_setup import REDACTED, configure_logging, make_redactor

# Built at runtime so scripts/check_no_secrets.py does not flag this file.
KEY_PREFIX = "gsk" + "_"
FAKE_KEY = KEY_PREFIX + "TestKey1234567890abcdefXYZ"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every settings variable from the process environment."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make(**overrides) -> Settings:
    return load_settings(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_match_architecture(self) -> None:
        s = make()
        assert s.groq_model == "openai/gpt-oss-120b"
        assert s.llm_timeout_seconds == 30
        assert s.llm_max_output_tokens == 900
        assert s.max_candidates == 12
        assert s.max_recommendations == 5
        assert s.vote_weight == 50
        assert s.max_per_chain == 2
        assert s.min_submit_interval_seconds == 3
        assert (s.budget_low_max, s.budget_medium_max) == (300, 500)
        assert (s.rpm_limit, s.rpd_limit, s.tpm_limit, s.tpd_limit) == (30, 1000, 8000, 200000)
        assert s.dataset_repo == "ManikaSaini/zomato-restaurant-recommendation"
        assert s.dataset_revision == "main"
        assert s.demo_mode is False
        assert s.log_level == "INFO"

    def test_missing_key_does_not_raise(self) -> None:
        s = make()
        assert s.groq_api_key.get_secret_value() == ""
        assert s.llm_enabled is False
        assert s.secret_values == []

    def test_empty_env_value_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MAX_CANDIDATES", "")
        assert make().max_candidates == 12

    def test_env_variable_is_read(self, monkeypatch) -> None:
        monkeypatch.setenv("MAX_RECOMMENDATIONS", "3")
        assert make().max_recommendations == 3

    def test_env_file_is_read(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("GROQ_MODEL=some/model\nLOG_LEVEL=debug\n", encoding="utf-8")
        s = load_settings(_env_file=env)
        assert s.groq_model == "some/model"
        assert s.log_level == "DEBUG"

    def test_settings_are_frozen(self) -> None:
        s = make()
        with pytest.raises(Exception):
            s.max_candidates = 3  # type: ignore[misc]

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_max_candidates_capped(self) -> None:
        assert make(max_candidates=50).max_candidates == MAX_CANDIDATES_CEILING

    def test_recommendations_must_not_exceed_candidates(self) -> None:
        with pytest.raises(ConfigError, match="MAX_RECOMMENDATIONS"):
            make(max_candidates=4, max_recommendations=5)

    def test_recommendations_equal_to_candidates_ok(self) -> None:
        assert make(max_candidates=5, max_recommendations=5).max_recommendations == 5

    @pytest.mark.parametrize("low, medium", [(500, 500), (600, 500)])
    def test_budget_thresholds_strictly_increasing(self, low, medium) -> None:
        with pytest.raises(ConfigError, match="BUDGET_LOW_MAX"):
            make(budget_low_max=low, budget_medium_max=medium)

    @pytest.mark.parametrize(
        "field",
        [
            "llm_timeout_seconds",
            "llm_max_output_tokens",
            "max_candidates",
            "max_recommendations",
            "vote_weight",
            "max_per_chain",
            "budget_low_max",
            "rpm_limit",
            "rpd_limit",
            "tpm_limit",
            "tpd_limit",
        ],
    )
    @pytest.mark.parametrize("bad", [0, -1])
    def test_limits_must_be_positive(self, field, bad) -> None:
        with pytest.raises(ConfigError) as info:
            make(**{field: bad})
        assert field.upper() in str(info.value)

    def test_submit_interval_zero_allowed_negative_rejected(self) -> None:
        assert make(min_submit_interval_seconds=0).min_submit_interval_seconds == 0
        with pytest.raises(ConfigError, match="MIN_SUBMIT_INTERVAL_SECONDS"):
            make(min_submit_interval_seconds=-1)

    def test_non_numeric_limit_names_variable_not_value(self, monkeypatch) -> None:
        monkeypatch.setenv("MAX_CANDIDATES", "twelve-ish")
        with pytest.raises(ConfigError) as info:
            make()
        assert "MAX_CANDIDATES" in str(info.value)
        assert "twelve-ish" not in str(info.value)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ConfigError, match="LOG_LEVEL"):
            make(log_level="LOUD")

    def test_log_level_normalized(self) -> None:
        assert make(log_level=" warning ").log_level == "WARNING"


class TestBooleans:
    @pytest.mark.parametrize("raw", ["true", "TRUE", "yes", "on", "1"])
    def test_truthy(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("DEMO_MODE", raw)
        assert make().demo_mode is True

    @pytest.mark.parametrize("raw", ["false", "no", "off", "0"])
    def test_falsy(self, monkeypatch, raw) -> None:
        monkeypatch.setenv("DEMO_MODE", raw)
        assert make().demo_mode is False

    def test_garbage_fails_naming_variable_only(self, monkeypatch) -> None:
        monkeypatch.setenv("DEMO_MODE", "perhapsXYZ")
        with pytest.raises(ConfigError) as info:
            make()
        assert "DEMO_MODE" in str(info.value)
        assert "perhapsXYZ" not in str(info.value)


# ---------------------------------------------------------------------------
# API key sanitizing and masking
# ---------------------------------------------------------------------------


class TestApiKey:
    @pytest.mark.parametrize(
        "raw",
        [
            FAKE_KEY,
            f"  {FAKE_KEY}  ",
            f'"{FAKE_KEY}"',
            f"'{FAKE_KEY}'",
            f"Bearer {FAKE_KEY}",
            f"bearer {FAKE_KEY}",
            f'"Bearer {FAKE_KEY}"',
            f"Bearer '{FAKE_KEY}'\n",
        ],
    )
    def test_sanitize(self, raw) -> None:
        assert sanitize_api_key(raw) == FAKE_KEY
        assert make(groq_api_key=raw).groq_api_key.get_secret_value() == FAKE_KEY

    def test_sanitize_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", f'  "Bearer {FAKE_KEY}"  ')
        assert make().groq_api_key.get_secret_value() == FAKE_KEY

    def test_mismatched_quotes_not_stripped(self) -> None:
        assert sanitize_api_key(f"'{FAKE_KEY}\"") == f"'{FAKE_KEY}\""

    def test_key_with_inner_space_rejected_without_echo(self) -> None:
        with pytest.raises(ConfigError) as info:
            make(groq_api_key="gsk_abc def123")
        message = str(info.value)
        assert "GROQ_API_KEY" in message
        assert "gsk_abc" not in message and "def123" not in message

    def test_whitespace_only_key_is_empty(self) -> None:
        assert make(groq_api_key="   ").groq_api_key.get_secret_value() == ""

    def test_key_masked_in_repr_and_str(self) -> None:
        s = make(groq_api_key=FAKE_KEY)
        assert FAKE_KEY not in repr(s)
        assert FAKE_KEY not in str(s)
        assert FAKE_KEY not in str(s.groq_api_key)
        assert FAKE_KEY not in s.model_dump_json()

    def test_secret_values(self) -> None:
        assert make(groq_api_key=FAKE_KEY).secret_values == [FAKE_KEY]


# ---------------------------------------------------------------------------
# llm_enabled
# ---------------------------------------------------------------------------


class TestLlmEnabled:
    @pytest.mark.parametrize(
        "key, demo, expected",
        [
            ("", False, False),
            ("", True, False),
            (FAKE_KEY, False, True),
            (FAKE_KEY, True, False),  # demo mode wins over a configured key
        ],
    )
    def test_matrix(self, key, demo, expected) -> None:
        assert make(groq_api_key=key, demo_mode=demo).llm_enabled is expected


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class TestPaths:
    def test_default_path_resolved_against_project_root(self) -> None:
        s = make()
        assert s.processed_path.is_absolute()
        assert s.processed_path == PROJECT_ROOT / "data" / "processed" / "restaurants.parquet"

    def test_relative_path_from_env_resolved(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSED_PATH", "other/r.parquet")
        assert make().processed_path == PROJECT_ROOT / "other" / "r.parquet"

    def test_absolute_path_kept(self, tmp_path: Path) -> None:
        target = tmp_path / "r.parquet"
        assert make(processed_path=target).processed_path == target

    def test_project_root_is_repo_folder(self) -> None:
        assert (PROJECT_ROOT / "pyproject.toml").is_file()

    def test_manifest_sits_next_to_parquet(self, tmp_path: Path) -> None:
        s = make(processed_path=tmp_path / "r.parquet")
        assert s.manifest_path == tmp_path / "manifest.json"

    def test_dataset_ready_needs_both_files(self, tmp_path: Path) -> None:
        s = make(processed_path=tmp_path / "r.parquet")
        assert s.dataset_ready is False
        (tmp_path / "r.parquet").write_bytes(b"x")
        assert s.dataset_ready is False
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        assert s.dataset_ready is True

    def test_runtime_state_paths_under_data(self) -> None:
        s = make()
        assert s.usage_path == PROJECT_ROOT / "data" / ".usage.json"
        assert s.cache_dir == PROJECT_ROOT / "data" / ".cache"


# ---------------------------------------------------------------------------
# Logging and redaction
# ---------------------------------------------------------------------------


@pytest.fixture
def log_stream():
    """Configure logging into a buffer and restore the root logger afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    stream = io.StringIO()

    def _configure(settings: Settings) -> io.StringIO:
        # pytest attaches its capture handler during the test call, after fixture
        # setup, so detach foreign handlers here; only ours should see records.
        root.handlers[:] = [
            h for h in root.handlers if getattr(h, "_zomato_reco_handler", False)
        ]
        configure_logging(settings, stream=stream)
        return stream

    yield _configure
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


class TestRedaction:
    def test_make_redactor_substring(self) -> None:
        redact = make_redactor([FAKE_KEY])
        assert redact(f"prefix-{FAKE_KEY}-suffix") == f"prefix-{REDACTED}-suffix"

    def test_make_redactor_any_key_shape(self) -> None:
        redact = make_redactor([])
        assert KEY_PREFIX not in redact(f"leaked {KEY_PREFIX}abcdefgh12345678 here")

    def test_short_secrets_ignored(self) -> None:
        assert make_redactor(["ab"])("abc") == "abc"

    def test_key_inside_longer_message_redacted(self, log_stream) -> None:
        stream = log_stream(make(groq_api_key=FAKE_KEY))
        logging.getLogger("t").warning("auth failed for key=%s on retry", FAKE_KEY)
        output = stream.getvalue()
        assert FAKE_KEY not in output
        assert REDACTED in output
        assert "auth failed for key=" in output

    def test_key_in_exception_redacted(self, log_stream) -> None:
        stream = log_stream(make(groq_api_key=FAKE_KEY))
        try:
            raise RuntimeError(f"bad header Authorization: Bearer {FAKE_KEY}")
        except RuntimeError:
            logging.getLogger("t").exception("call failed")
        output = stream.getvalue()
        assert FAKE_KEY not in output
        assert "RuntimeError" in output

    def test_key_in_extra_field_redacted(self, log_stream) -> None:
        stream = log_stream(make(groq_api_key=FAKE_KEY))
        logging.getLogger("t").info("event", extra={"detail": f"x{FAKE_KEY}x"})
        assert FAKE_KEY not in stream.getvalue()

    def test_single_line_json(self, log_stream) -> None:
        stream = log_stream(make())
        logging.getLogger("t").info("hello\nworld", extra={"count": 3})
        lines = stream.getvalue().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["message"] == "hello\nworld"
        assert payload["level"] == "INFO"
        assert payload["count"] == 3

    def test_level_from_settings(self, log_stream) -> None:
        stream = log_stream(make(log_level="WARNING"))
        logging.getLogger("t").info("hidden")
        logging.getLogger("t").warning("shown")
        output = stream.getvalue()
        assert "hidden" not in output and "shown" in output

    def test_configure_twice_keeps_one_handler(self, log_stream) -> None:
        s = make()
        log_stream(s)
        log_stream(s)
        marked = [h for h in logging.getLogger().handlers if getattr(h, "_zomato_reco_handler", False)]
        assert len(marked) == 1

    def test_broken_log_call_does_not_raise(self, log_stream) -> None:
        stream = log_stream(make())
        logging.getLogger("t").info("needs %s %s", "one")
        assert "unformattable" in stream.getvalue()
