"""Unit tests for structured logging (US-5.1)."""

from __future__ import annotations

import json

from loguru import logger

from email_observability.logging import (
    configure_logging,
    get_request_id,
    redact_text,
    request_context,
)


class TestRedaction:
    def test_redacts_email(self) -> None:
        assert "john@example.com" not in redact_text("mail john@example.com now")
        assert "[REDACTED]" in redact_text("mail john@example.com now")

    def test_redacts_ssn_and_ip(self) -> None:
        assert "[REDACTED]" in redact_text("ssn 123-45-6789")
        assert "[REDACTED]" in redact_text("ip 192.168.0.1")

    def test_leaves_plain_text_untouched(self) -> None:
        assert redact_text("hello world") == "hello world"


class TestRequestContext:
    def test_sets_and_resets(self) -> None:
        assert get_request_id() is None
        with request_context("req-abc") as rid:
            assert rid == "req-abc"
            assert get_request_id() == "req-abc"
        assert get_request_id() is None

    def test_generates_id_when_absent(self) -> None:
        with request_context() as rid:
            assert rid and len(rid) > 0


class TestConfigureLogging:
    def _capture(self, **kwargs) -> list[str]:
        captured: list[str] = []
        configure_logging(sink=captured.append, **kwargs)
        return captured

    def test_json_log_has_request_id_and_redacts(self) -> None:
        captured = self._capture(json_logs=True)
        with request_context("req-123"):
            logger.bind(password="hunter2", user="a@b.com").info("contact john@example.com")
        data = json.loads(captured[-1])

        assert data["request_id"] == "req-123"
        assert data["level"] == "INFO"
        # PII redacted in message
        assert "john@example.com" not in data["message"]
        assert "[REDACTED]" in data["message"]
        # sensitive key value masked, PII in extra value redacted
        assert data["extra"]["password"] == "***"
        assert "a@b.com" not in json.dumps(data["extra"])

    def test_redaction_can_be_disabled(self) -> None:
        captured = self._capture(json_logs=True, redact=False)
        logger.bind(password="hunter2").info("john@example.com")
        data = json.loads(captured[-1])
        assert "john@example.com" in data["message"]
        assert data["extra"]["password"] == "hunter2"

    def test_error_level_recorded(self) -> None:
        captured = self._capture(json_logs=True)
        logger.error("boom")
        data = json.loads(captured[-1])
        assert data["level"] == "ERROR"
