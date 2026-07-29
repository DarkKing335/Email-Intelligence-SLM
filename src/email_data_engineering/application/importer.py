"""Import stage — US-1.1.

Loads raw email data from a local CSV or JSONL file, validates records against
the schema, and produces an :class:`ImportResult` with provenance metadata.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from email_data_engineering.domain.models import EmailRecord, ImportResult
from email_data_engineering.infrastructure.loaders import (
    CSVEmailLoader,
    JSONLEmailLoader,
)
from email_data_engineering.infrastructure.schema import SchemaValidator


class DataImporter:
    """Import stage: load raw data and validate records.

    Implements US-1.1 acceptance criteria:
    - Can ingest a dataset from a documented source path.
    - Imported records are validated against a defined schema.
    - Invalid records are reported separately.
    - The import produces a traceable dataset artifact.

    Parameters
    ----------
    column_mapping:
        Maps source CSV columns to internal field names.
    chunk_size:
        Rows per chunk for CSV loading (memory efficiency).
    encoding:
        Source file encoding.
    required_fields:
        Fields that must be present and non-empty.
    max_body_chars:
        Maximum body character length before clipping (handled in cleaner).
    min_body_chars:
        Minimum body character length; shorter records are rejected.
    max_subject_chars:
        Maximum subject character length.
    report_dir:
        Directory where ``import_report.json`` and ``rejection_report.json``
        are written.
    """

    def __init__(
        self,
        column_mapping: dict[str, str | None] | None = None,
        chunk_size: int = 10_000,
        encoding: str = "utf-8",
        required_fields: list[str] | None = None,
        max_body_chars: int = 8_000,
        min_body_chars: int = 10,
        max_subject_chars: int = 500,
        report_dir: str | Path = "reports",
    ) -> None:
        self._column_mapping = column_mapping
        self._chunk_size = chunk_size
        self._encoding = encoding
        self._validator = SchemaValidator(
            required_fields=required_fields,
            max_body_chars=max_body_chars,
            min_body_chars=min_body_chars,
            max_subject_chars=max_subject_chars,
        )
        self._report_dir = Path(report_dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, source_path: str | Path) -> ImportResult:
        """Run the import stage on *source_path*.

        Parameters
        ----------
        source_path:
            Path to a ``.csv`` or ``.jsonl`` file.

        Returns
        -------
        ImportResult
            Contains validated records and rejection statistics.
        """
        source = Path(source_path)
        logger.info(f"[Import] Starting import from: {source}")

        # 1. Load raw records
        raw_records = self._load(source)
        total_read = len(raw_records)
        logger.info(f"[Import] Loaded {total_read} raw records from {source.name}")

        # 2. Validate
        valid_raws, report = self._validator.validate_batch(raw_records)
        logger.info(
            f"[Import] Validation complete: {len(valid_raws)} valid, "
            f"{len(report.rejected)} rejected"
        )

        # 3. Convert to EmailRecord domain objects
        records = [self._to_email_record(r, str(source)) for r in valid_raws]

        # 4. Build result
        result = ImportResult(
            total_rows_read=total_read,
            valid_records=len(records),
            rejected_records=len(report.rejected),
            source_path=str(source),
            rejection_reasons=report.rejection_reasons,
            records=records,
        )

        # 5. Write reports
        self._write_reports(result, report.rejected)

        logger.info(
            f"[Import] Done — {result.valid_records} records accepted "
            f"({result.acceptance_rate:.1%} acceptance rate)"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, source: Path) -> list[dict[str, Any]]:
        suffix = source.suffix.lower()
        if suffix == ".csv":
            loader = CSVEmailLoader(
                source_path=source,
                column_mapping=self._column_mapping,
                chunk_size=self._chunk_size,
                encoding=self._encoding,
            )
        elif suffix == ".jsonl":
            loader = JSONLEmailLoader(source_path=source)
        else:
            raise ValueError(
                f"Unsupported source format '{suffix}'. "
                "Expected .csv or .jsonl"
            )
        return list(loader.iter_records())

    @staticmethod
    def _to_email_record(raw: dict[str, Any], source: str) -> EmailRecord:
        """Convert a validated raw dict to an EmailRecord."""
        body = str(raw.get("body") or raw.get("body_text") or "").strip()
        label = str(raw.get("label") or raw.get("category") or "").strip()
        subject = str(raw.get("subject") or "").strip()
        sender_domain = str(raw.get("sender_domain") or "").strip()

        return EmailRecord(
            email_id=str(raw.get("email_id") or uuid.uuid4()),
            thread_id=str(raw.get("thread_id") or ""),
            sender_domain=sender_domain,
            subject=subject,
            body_text=body,
            # At import time label is still raw; normalization happens next
            label=label,
            original_label=label,
            # Temporary defaults until normalization stage assigns real values
            priority="medium",
            response_required=False,
            source=source,
        )

    def _write_reports(
        self,
        result: ImportResult,
        rejected: list[dict[str, Any]],
    ) -> None:
        """Write import and rejection reports to the reports directory."""
        self._report_dir.mkdir(parents=True, exist_ok=True)

        # Import summary report
        import_report = {
            "stage": "import",
            "source_path": result.source_path,
            "total_rows_read": result.total_rows_read,
            "valid_records": result.valid_records,
            "rejected_records": result.rejected_records,
            "acceptance_rate": round(result.acceptance_rate, 4),
        }
        self._write_json(import_report, self._report_dir / "import_report.json")

        # Rejection detail report
        rejection_report = {
            "stage": "import",
            "total_rejected": result.rejected_records,
            "rejection_reasons": result.rejection_reasons,
            "rejected_records": rejected[:500],  # cap at 500 for readability
        }
        self._write_json(rejection_report, self._report_dir / "rejection_report.json")

    @staticmethod
    def _write_json(data: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Report written: {path}")
