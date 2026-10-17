"""Application settings loaded from environment variables and a local .env file.

Rules enforced here (see docs/architecture.md section 9):

* The Groq key is a secret string. It is sanitized on load and never printed.
* Numeric limits must be positive and thresholds strictly increasing.
* Validation errors name the variable and never echo a value.
* A relative PROCESSED_PATH is resolved against the project root.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import (
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"

MAX_CANDIDATES_CEILING = 20
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(Exception):
    """Raised when settings are invalid. The message never contains a value."""


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def sanitize_api_key(raw: str) -> str:
    """Strip whitespace, matching quotes and a leading ``Bearer `` from a pasted key."""
    key = _unquote(raw.strip()).strip()
    if key.lower().startswith("bearer "):
        key = key[len("bearer ") :].strip()
    return _unquote(key).strip()


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case variable names."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_ignore_empty=True,
    )

    # Groq provider
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: PositiveInt = 30
    llm_max_output_tokens: PositiveInt = 900

    # Candidate selection and results
    max_candidates: PositiveInt = 12
    max_recommendations: PositiveInt = 5
    vote_weight: PositiveInt = 50
    max_per_chain: PositiveInt = 2
    min_submit_interval_seconds: int = 3

    # Budget tiers (approximate cost for two, INR). Tuned in Phase 1 to the
    # 33rd/66th cost percentiles of the ingested data (p33=300, p66=500).
    budget_low_max: PositiveInt = 300
    budget_medium_max: PositiveInt = 500

    # Groq limits recorded for the project
    rpm_limit: PositiveInt = 30
    rpd_limit: PositiveInt = 1000
    tpm_limit: PositiveInt = 8000
    tpd_limit: PositiveInt = 200000

    # Dataset
    dataset_repo: str = "ManikaSaini/zomato-restaurant-recommendation"
    dataset_revision: str = "main"
    processed_path: Path = Path("data/processed/restaurants.parquet")

    # Behaviour
    demo_mode: bool = False
    log_level: str = "INFO"

    # ---- validators -------------------------------------------------------

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _sanitize_key(cls, value: Any) -> SecretStr:
        if value is None:
            return SecretStr("")
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        key = sanitize_api_key(raw)
        if re.search(r"\s", key):
            raise ValueError("must not contain spaces")
        return SecretStr(key)

    @field_validator("max_candidates")
    @classmethod
    def _cap_candidates(cls, value: int) -> int:
        return min(value, MAX_CANDIDATES_CEILING)

    @field_validator("min_submit_interval_seconds")
    @classmethod
    def _interval_not_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be zero or greater")
        return value

    @field_validator("processed_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in VALID_LOG_LEVELS:
            raise ValueError("must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return level

    @model_validator(mode="after")
    def _check_relationships(self) -> "Settings":
        if self.max_recommendations > self.max_candidates:
            raise ValueError("MAX_RECOMMENDATIONS must not exceed MAX_CANDIDATES")
        if self.budget_low_max >= self.budget_medium_max:
            raise ValueError("BUDGET_LOW_MAX must be less than BUDGET_MEDIUM_MAX")
        return self

    # ---- derived values ---------------------------------------------------

    @property
    def llm_enabled(self) -> bool:
        """True only when a key is present and demo mode is off."""
        return bool(self.groq_api_key.get_secret_value()) and not self.demo_mode

    @property
    def manifest_path(self) -> Path:
        return self.processed_path.with_name("manifest.json")

    @property
    def dataset_ready(self) -> bool:
        """True when both the processed parquet and its manifest exist."""
        return self.processed_path.is_file() and self.manifest_path.is_file()

    @property
    def usage_path(self) -> Path:
        return DATA_DIR / ".usage.json"

    @property
    def cache_dir(self) -> Path:
        return DATA_DIR / ".cache"

    @property
    def secret_values(self) -> list[str]:
        """Secrets that must be redacted from logs."""
        key = self.groq_api_key.get_secret_value()
        return [key] if key else []


def _describe(error: ValidationError) -> str:
    """Summarize validation errors by variable name only, never by value."""
    lines = []
    for item in error.errors(include_input=False, include_url=False):
        location = item.get("loc") or ()
        name = str(location[0]).upper() if location else "settings"
        message = str(item.get("msg", "invalid value")).removeprefix("Value error, ")
        lines.append(f"{name}: {message}")
    return "Invalid configuration. " + "; ".join(lines)


def load_settings(_env_file: Path | None = ENV_FILE, **overrides: Any) -> Settings:
    """Build settings, converting validation errors into a value-free ConfigError."""
    try:
        return Settings(_env_file=_env_file, **overrides)  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigError(_describe(exc)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Call ``get_settings.cache_clear()`` in tests."""
    return load_settings()
