"""Structured logging with request tracing and redaction (US-5.1).

Extends the project's Loguru usage with the three things the acceptance
criteria require beyond plain logging:

    * **Structured & searchable** — JSON lines with stable fields.
    * **End-to-end traceability** — a request/correlation id, propagated via a
      ``contextvars`` context so every log line for one request shares an id.
    * **No sensitive data** — a redaction patcher masks PII in messages and
      masks the *values* of sensitive keys (password, token, …) in ``extra``.
    * **Errors distinct** — the ``level`` field separates error logs, and an
      optional dedicated error sink can capture ``ERROR`` and above.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger

# ── Request-id context ────────────────────────────────────────────────────────

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

REDACTION_TOKEN = "[REDACTED]"
MASK_TOKEN = "***"

DEFAULT_SENSITIVE_KEYS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "client_secret",
)

# PII patterns mirror the data-engineering cleaner's redaction intent.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),  # credit card
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IPv4
    re.compile(r"\+?\d[\d\s().-]{8,}\d"),  # phone
)


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id for the duration of the ``with`` block.

    All logs emitted inside the block carry ``request_id`` in ``extra``, so a
    single request can be traced end-to-end. A new id is generated if none is
    provided.
    """
    rid = request_id or uuid.uuid4().hex
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the current request id, or ``None`` outside a request context."""
    return _request_id.get()


# ── Redaction ─────────────────────────────────────────────────────────────────


def redact_text(text: str) -> str:
    """Replace PII occurrences in a string with the redaction token."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub(REDACTION_TOKEN, text)
    return text


def _redact_value(value: Any, sensitive_keys: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return _redact_mapping(value, sensitive_keys)
    if isinstance(value, (list, tuple)):
        return [_redact_value(v, sensitive_keys) for v in value]
    return value


def _redact_mapping(mapping: dict[str, Any], sensitive_keys: tuple[str, ...]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in mapping.items():
        if any(sk in key.lower() for sk in sensitive_keys):
            redacted[key] = MASK_TOKEN
        else:
            redacted[key] = _redact_value(value, sensitive_keys)
    return redacted


def _make_patcher(redact: bool, sensitive_keys: tuple[str, ...]):
    """Build a Loguru patcher that injects the request id and redacts."""

    def _patch(record: dict[str, Any]) -> None:
        rid = _request_id.get()
        if rid is not None:
            record["extra"].setdefault("request_id", rid)
        if redact:
            record["message"] = redact_text(record["message"])
            record["extra"] = _redact_mapping(record["extra"], sensitive_keys)

    return _patch


# ── Serialization ─────────────────────────────────────────────────────────────


def serialize_record(record: dict[str, Any]) -> str:
    """Render a (already patched/redacted) record as a JSON line."""
    exc = record["exception"]
    exception_data = None
    if exc is not None:
        exception_data = {"type": str(exc.type), "value": str(exc.value)}

    entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "request_id": record["extra"].get("request_id"),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "extra": {k: v for k, v in record["extra"].items() if k != "request_id"},
        "exception": exception_data,
    }
    return json.dumps(entry, ensure_ascii=False, default=str) + "\n"


def _json_sink_format(record: dict[str, Any]) -> str:
    # Loguru calls this to build the template; we stash the serialized line in
    # extra and emit it verbatim via "{extra[serialized]}".
    record["extra"]["serialized"] = serialize_record(record)
    return "{extra[serialized]}"


# ── Configuration ─────────────────────────────────────────────────────────────


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    redact: bool = True,
    sensitive_keys: tuple[str, ...] | None = None,
    sink: Any = sys.stderr,
    error_log_file: str | Path | None = None,
) -> None:
    """Configure structured logging with request tracing and redaction.

    Parameters
    ----------
    level:
        Minimum level for the primary sink.
    json_logs:
        Emit JSON lines (searchable) when ``True``; human format otherwise.
    redact:
        Mask PII in messages and sensitive-key values in ``extra``.
    sensitive_keys:
        Substrings that mark a key's value as sensitive. Defaults to
        :data:`DEFAULT_SENSITIVE_KEYS`.
    sink:
        Primary sink (defaults to stderr).
    error_log_file:
        If set, ``ERROR`` and above are *also* written here, keeping error logs
        distinct from the normal stream.
    """
    keys = sensitive_keys or DEFAULT_SENSITIVE_KEYS
    logger.remove()
    logger.configure(patcher=_make_patcher(redact, keys))

    if json_logs:
        logger.add(sink, level=level, format=_json_sink_format)
    else:
        logger.add(
            sink,
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "req={extra[request_id]} | {name}:{function}:{line} - {message}"
            ),
        )

    if error_log_file is not None:
        path = Path(error_log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(path), level="ERROR", format=_json_sink_format, enqueue=True)
