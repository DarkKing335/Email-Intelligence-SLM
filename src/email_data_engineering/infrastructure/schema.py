"""Schema validation for raw email records.

Validates raw dicts produced by loaders against the EmailRecord domain model
before any pipeline stage processes them.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import ValidationError

from email_data_engineering.domain.models import EmailRecord


# ── Result type ───────────────────────────────────────────────────────────────

class ValidationReport:
    """Accumulates schema validation outcomes."""

    def __init__(self) -> None:
        self.valid: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.rejection_reasons: dict[str, int] = {}

    def add_valid(self, record: dict[str, Any]) -> None:
        self.valid.append(record)

    def add_rejected(self, record: dict[str, Any], reason: str) -> None:
        self.rejected.append({"record": record, "reason": reason})
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.rejected)

    @property
    def acceptance_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return len(self.valid) / self.total

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": len(self.valid),
            "rejected": len(self.rejected),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "rejection_reasons": self.rejection_reasons,
        }


# ── Validator ─────────────────────────────────────────────────────────────────

class SchemaValidator:
    """Validates raw record dicts against the EmailRecord schema.

    Uses Pydantic for structural validation and applies additional
    business-rule checks (required fields, length bounds).

    Parameters
    ----------
    required_fields:
        Fields that must be present and non-empty.
    max_body_chars:
        Maximum allowed body_text length.
    min_body_chars:
        Minimum allowed body_text length.
    max_subject_chars:
        Maximum allowed subject length.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        max_body_chars: int = 8_000,
        min_body_chars: int = 10,
        max_subject_chars: int = 500,
    ) -> None:
        self._required_fields = required_fields or ["subject", "body", "label"]
        self._max_body_chars = max_body_chars
        self._min_body_chars = min_body_chars
        self._max_subject_chars = max_subject_chars

    def validate_batch(
        self, raw_records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], ValidationReport]:
        """Validate a list of raw records.

        Parameters
        ----------
        raw_records:
            Raw dicts from a loader.

        Returns
        -------
        tuple[list[dict], ValidationReport]
            A tuple of (valid_records, report).  Valid records are the
            raw dicts that passed all checks (not EmailRecord objects —
            conversion happens in the Import stage).
        """
        report = ValidationReport()

        for raw in raw_records:
            rejection = self._check(raw)
            if rejection:
                report.add_rejected(raw, rejection)
            else:
                report.add_valid(raw)

        logger.info(
            "Schema validation complete",
            total=report.total,
            valid=len(report.valid),
            rejected=len(report.rejected),
            acceptance_rate=f"{report.acceptance_rate:.1%}",
        )
        return report.valid, report

    def validate_one(self, raw: dict[str, Any]) -> str | None:
        """Validate a single record.  Returns rejection reason or None."""
        return self._check(raw)

    # ── Private ───────────────────────────────────────────────────────────────

    def _check(self, raw: dict[str, Any]) -> str | None:
        """Run all checks.  Return the first failure reason or None."""
        # Required fields
        for field in self._required_fields:
            # Accept both 'body' (raw loader key) and 'body_text' (domain key)
            key = field if field != "body_text" else "body"
            alt_key = "body_text" if key == "body" else key
            val = raw.get(key) or raw.get(alt_key)
            if not val or not str(val).strip():
                return f"missing_required:{field}"

        # Body length
        body = str(raw.get("body") or raw.get("body_text") or "").strip()
        if len(body) < self._min_body_chars:
            return f"body_too_short:<{self._min_body_chars}_chars"
        # We clip long bodies rather than reject them (see cleaner.py)
        # but we still flag suspiciously empty subjects
        subject = str(raw.get("subject") or "").strip()
        if len(subject) > self._max_subject_chars:
            return f"subject_too_long:>{self._max_subject_chars}_chars"

        # Label presence
        label = str(raw.get("label") or raw.get("category") or "").strip()
        if not label:
            return "missing_label"

        return None
