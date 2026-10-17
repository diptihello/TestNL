"""Persistent request/token budget guard and bounded result cache."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from zomato_reco.config import Settings
from zomato_reco.domain.models import Candidate, Preferences
from zomato_reco.llm.prompts import PROMPT_VERSION
from zomato_reco.llm.schema import LLMOutput

logger = logging.getLogger(__name__)
_PROCESS_LOCK = threading.RLock()
_WINDOW_MINUTE = 60.0
_WINDOW_DAY = 24 * 60 * 60.0


class UsageStoreError(OSError):
    """Usage counters could not be safely persisted."""


class UsageStore:
    """JSON-backed rolling usage events; each event represents one request."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self.events: list[dict[str, float | int | str]] = []
        self.reload()

    def reload(self) -> None:
        with _PROCESS_LOCK:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    raise ValueError("usage root must be a list")
                events: list[dict[str, float | int | str]] = []
                now = self._clock()
                for event in raw:
                    if not isinstance(event, dict):
                        raise ValueError("invalid usage event")
                    timestamp = float(event["timestamp"])
                    tokens = int(event["tokens"])
                    if tokens < 0:
                        raise ValueError("negative token count")
                    events.append(
                        {
                            "id": str(event.get("id") or uuid.uuid4().hex),
                            "timestamp": min(timestamp, now),
                            "tokens": tokens,
                        }
                    )
                self.events = events
            except FileNotFoundError:
                self.events = []
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._quarantine()
                logger.warning("usage_file_corrupt", extra={"error_class": type(exc).__name__})
                self.events = []
            self.prune()
            self.save()

    def _quarantine(self) -> None:
        if not self.path.exists():
            return
        bad_path = self.path.with_name(self.path.name + ".bad")
        try:
            os.replace(self.path, bad_path)
        except OSError as exc:
            raise UsageStoreError("Corrupt usage data could not be quarantined.") from exc

    def prune(self) -> None:
        now = self._clock()
        self.events = [
            event
            for event in self.events
            if now - float(event["timestamp"]) < _WINDOW_DAY
        ]

    def save(self) -> None:
        with _PROCESS_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            try:
                temporary.write_text(
                    json.dumps(self.events, separators=(",", ":")),
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)
            except OSError as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise UsageStoreError("Usage data could not be saved.") from exc


class BudgetExceeded(Exception):
    """A request would exceed a configured rolling usage limit."""

    def __init__(self, wait_seconds: int, limit_name: str) -> None:
        self.wait_seconds = max(1, int(wait_seconds))
        self.limit_name = limit_name
        super().__init__(f"{limit_name} reached; retry in {self.wait_seconds} seconds.")


def _wait_for_capacity(
    events: list[dict[str, float | int | str]],
    now: float,
    window: float,
    limit: int,
    required: int,
    *,
    token_weight: int | None = None,
) -> int:
    if token_weight is None:
        ordered = sorted(float(event["timestamp"]) for event in events)
        excess = len(ordered) + required - limit
        expiries = ordered[: max(0, excess)]
        return max(1, math.ceil(expiries[-1] + window - now)) if expiries else 1

    ordered_tokens = sorted(
        events, key=lambda event: (float(event["timestamp"]), int(event["tokens"]))
    )
    total = sum(int(event["tokens"]) for event in ordered_tokens)
    excess = total + token_weight - limit
    if excess <= 0:
        return 1
    removed = 0
    for event in ordered_tokens:
        removed += int(event["tokens"])
        if removed >= excess:
            return max(1, math.ceil(float(event["timestamp"]) + window - now))
    return max(1, int(window))


class BudgetGuard:
    """Atomic reservation and accounting for minute/day request and token caps."""

    def __init__(
        self,
        store: UsageStore,
        settings: Settings,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.settings = settings
        self._clock = clock
        self._local = threading.local()

    def _events(self, window: float) -> list[dict[str, float | int | str]]:
        now = self._clock()
        return [
            event
            for event in self.store.events
            if now - float(event["timestamp"]) < window
        ]

    def _check_locked(
        self, estimated_prompt_tokens: int, max_output_tokens: int
    ) -> BudgetExceeded | None:
        now = self._clock()
        self.store.prune()
        minute = self._events(_WINDOW_MINUTE)
        day = self._events(_WINDOW_DAY)
        worst_case = max(0, int(estimated_prompt_tokens)) + max(
            0, int(max_output_tokens)
        )
        checks = (
            (
                "requests per minute",
                minute,
                self.settings.rpm_limit,
                1,
                _WINDOW_MINUTE,
                None,
            ),
            (
                "requests per day",
                day,
                self.settings.rpd_limit,
                1,
                _WINDOW_DAY,
                None,
            ),
            (
                "tokens per minute",
                minute,
                self.settings.tpm_limit,
                worst_case,
                _WINDOW_MINUTE,
                worst_case,
            ),
            (
                "tokens per day",
                day,
                self.settings.tpd_limit,
                worst_case,
                _WINDOW_DAY,
                worst_case,
            ),
        )
        for name, events, limit, required, window, token_weight in checks:
            current = (
                sum(int(event["tokens"]) for event in events)
                if token_weight is not None
                else len(events)
            )
            addition = token_weight if token_weight is not None else required
            if current + addition > limit:
                wait = _wait_for_capacity(
                    events,
                    now,
                    window,
                    limit,
                    required,
                    token_weight=token_weight,
                )
                return BudgetExceeded(wait, name)
        return None

    def check(
        self, estimated_prompt_tokens: int, max_output_tokens: int = 0
    ) -> BudgetExceeded | None:
        with _PROCESS_LOCK:
            self.store.reload()
            exceeded = self._check_locked(estimated_prompt_tokens, max_output_tokens)
            self.store.save()
            return exceeded

    def check_and_reserve(
        self, estimated_prompt_tokens: int, max_output_tokens: int
    ) -> BudgetExceeded | None:
        """Atomically check all windows and reserve one request plus worst-case tokens."""
        with _PROCESS_LOCK:
            self.store.reload()
            exceeded = self._check_locked(estimated_prompt_tokens, max_output_tokens)
            if exceeded:
                self.store.save()
                return exceeded
            reservation_id = uuid.uuid4().hex
            self.store.events.append(
                {
                    "id": reservation_id,
                    "timestamp": self._clock(),
                    "tokens": max(0, int(estimated_prompt_tokens))
                    + max(0, int(max_output_tokens)),
                }
            )
            self.store.save()
            pending = getattr(self._local, "reservations", [])
            pending.append(reservation_id)
            self._local.reservations = pending
            return None

    def _record(self, tokens: int) -> None:
        with _PROCESS_LOCK:
            self.store.reload()
            pending = getattr(self._local, "reservations", [])
            reservation_id = pending.pop(0) if pending else None
            if reservation_id is not None:
                for event in self.store.events:
                    if event["id"] == reservation_id:
                        event["tokens"] = max(0, int(tokens))
                        break
            else:
                self.store.events.append(
                    {
                        "id": uuid.uuid4().hex,
                        "timestamp": self._clock(),
                        "tokens": max(0, int(tokens)),
                    }
                )
            self._local.reservations = pending
            self.store.save()

    def record(self, usage: int | dict[str, Any] | Any) -> None:
        if isinstance(usage, int):
            total = usage
        elif isinstance(usage, dict):
            total = int(usage.get("total_tokens", 0))
        else:
            total = int(getattr(usage, "total_tokens", 0))
        self._record(total)

    def record_failure(self, estimated_prompt_tokens: int) -> None:
        pending = getattr(self._local, "reservations", [])
        if pending:
            with _PROCESS_LOCK:
                self.store.reload()
                reservation_id = pending.pop(0)
                for event in self.store.events:
                    if event["id"] == reservation_id:
                        event["tokens"] = max(0, int(estimated_prompt_tokens))
                        break
                self._local.reservations = pending
                self.store.save()
            return
        self._record(estimated_prompt_tokens)

    def snapshot(self) -> dict[str, int]:
        with _PROCESS_LOCK:
            self.store.reload()
            minute = self._events(_WINDOW_MINUTE)
            day = self._events(_WINDOW_DAY)
            self.store.save()
            return {
                "requests_minute": len(minute),
                "requests_minute_limit": int(self.settings.rpm_limit),
                "requests_day": len(day),
                "requests_day_limit": int(self.settings.rpd_limit),
                "tokens_minute": sum(int(event["tokens"]) for event in minute),
                "tokens_minute_limit": int(self.settings.tpm_limit),
                "tokens_day": sum(int(event["tokens"]) for event in day),
                "tokens_day_limit": int(self.settings.tpd_limit),
            }


class ResultCache:
    """Small JSON cache for parsed LLM output; disabled after write failures."""

    def __init__(
        self,
        directory: Path,
        *,
        clock: Callable[[], float] = time.time,
        ttl_seconds: float = _WINDOW_DAY,
        max_entries: int = 200,
    ) -> None:
        self.directory = Path(directory)
        self._clock = clock
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._disabled = False
        self._warned = False

    @staticmethod
    def key(
        prefs: Preferences,
        candidates: list[Candidate],
        model: str,
        prompt_version: str = PROMPT_VERSION,
    ) -> str:
        normalized = {
            "location": " ".join(prefs.location.casefold().split()),
            "budget": prefs.budget.value,
            "min_rating": prefs.min_rating,
            "cuisine": " ".join((prefs.cuisine or "").casefold().split()),
            "extra_preferences": " ".join(
                (prefs.extra_preferences or "").casefold().split()
            ),
            "candidate_ids": sorted(candidate.id for candidate in candidates),
            "model": model,
            "prompt_version": prompt_version,
        }
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _disable(self, exc: OSError) -> None:
        self._disabled = True
        if not self._warned:
            logger.warning("result_cache_disabled", extra={"error_class": type(exc).__name__})
            self._warned = True

    def get_with_age(self, key: str) -> tuple[LLMOutput, float] | None:
        if self._disabled:
            return None
        path = self.directory / f"{key}.json"
        with _PROCESS_LOCK:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                saved_at = float(raw["saved_at"])
                age = max(0.0, self._clock() - saved_at)
                if age > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                    return None
                output = LLMOutput.model_validate(raw["output"])
                return output, age
            except FileNotFoundError:
                return None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    self._disable(exc)
                return None
            except OSError as exc:
                self._disable(exc)
                return None

    def get(self, key: str) -> LLMOutput | None:
        found = self.get_with_age(key)
        return found[0] if found else None

    def put(self, key: str, output: LLMOutput) -> None:
        if self._disabled:
            return
        with _PROCESS_LOCK:
            temporary: Path | None = None
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.directory / f".{key}.{uuid.uuid4().hex}.tmp"
                target = self.directory / f"{key}.json"
                temporary.write_text(
                    json.dumps(
                        {
                            "saved_at": self._clock(),
                            "output": output.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, target)
                self._prune_locked()
            except OSError as exc:
                self._disable(exc)
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass

    def _prune_locked(self) -> None:
        files = list(self.directory.glob("*.json"))
        now = self._clock()
        valid: list[tuple[float, Path]] = []
        for path in files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                saved_at = float(raw["saved_at"])
                if now - saved_at > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                else:
                    valid.append((saved_at, path))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    raise
        valid.sort(reverse=True)
        for _, path in valid[self.max_entries :]:
            path.unlink(missing_ok=True)
