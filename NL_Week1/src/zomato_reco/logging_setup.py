"""Single-line JSON logging with secret redaction.

The API key is removed from every message, exception traceback and extra field,
wherever it appears as a substring. Anything shaped like a Groq key is removed
as well, even if it is not the configured one.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import TextIO

from zomato_reco.config import Settings

REDACTED = "[REDACTED]"
_KEY_SHAPE = re.compile(r"gsk_[A-Za-z0-9]{8,}")
_MIN_SECRET_LENGTH = 4
_HANDLER_MARK = "_zomato_reco_handler"
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def make_redactor(secrets: Iterable[str]) -> Callable[[str], str]:
    """Return a function that removes the given secrets and key-shaped strings."""
    values = sorted(
        {s for s in secrets if s and len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True
    )

    def redact(text: str) -> str:
        for value in values:
            text = text.replace(value, REDACTED)
        return _KEY_SHAPE.sub(REDACTED, text)

    return redact


class RedactingFilter(logging.Filter):
    """Redact secrets in the formatted message and in exception text."""

    def __init__(self, redact: Callable[[str], str]) -> None:
        super().__init__()
        self._redact = redact

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = self._redact(record.getMessage())
            record.args = None
        except Exception:  # a broken log call must never break the app
            record.msg = "<unformattable log message>"
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = "".join(traceback.format_exception(*record.exc_info))
        if record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        if isinstance(record.stack_info, str):
            record.stack_info = self._redact(record.stack_info)
        return True


class JsonLineFormatter(logging.Formatter):
    """Format each record as one JSON object on one line."""

    def __init__(self, redact: Callable[[str], str] = lambda text: text) -> None:
        super().__init__()
        self._redact = redact

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name, value in record.__dict__.items():
            if name not in _STANDARD_ATTRS and not name.startswith("_"):
                payload[name] = value
        if record.exc_text:
            payload["exc"] = record.exc_text
        elif record.exc_info:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))
        line = json.dumps(payload, default=str, ensure_ascii=False)
        return self._redact(line)


def configure_logging(settings: Settings, stream: TextIO | None = None) -> logging.Handler:
    """Install one redacting JSON handler on the root logger. Safe to call repeatedly."""
    redact = make_redactor(settings.secret_values)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLineFormatter(redact))
    handler.addFilter(RedactingFilter(redact))
    setattr(handler, _HANDLER_MARK, True)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_MARK, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    return handler
