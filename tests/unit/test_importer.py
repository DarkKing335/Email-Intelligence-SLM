"""Unit tests for the DataImporter stage (US-1.1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from email_data_engineering.application.importer import DataImporter


def test_import_csv_valid_records(tmp_dir: Path) -> None:
    # 1. Create a dummy valid CSV
    csv_path = tmp_dir / "valid_emails.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "category"])
        writer.writerow(["alice@example.com", "Urgent request", "This is a body that is long enough.", "work"])
        writer.writerow(["bob@example.com", "Fyi update", "Another long body text for testing.", "social"])

    # 2. Run importer
    importer = DataImporter(
        column_mapping={
            "email_id": None,
            "thread_id": None,
            "sender": "sender",
            "subject": "subject",
            "body": "body",
            "label": "category",
        },
        min_body_chars=5,
        report_dir=tmp_dir / "reports",
    )
    result = importer.run(csv_path)

    # 3. Assertions
    assert result.total_rows_read == 2
    assert result.valid_records == 2
    assert result.rejected_records == 0
    assert len(result.records) == 2
    assert result.records[0].subject == "Urgent request"
    assert result.records[0].sender_domain == "example.com"
    assert result.records[0].original_label == "work"
    assert result.records[1].subject == "Fyi update"
    
    # Check reports are written
    assert (tmp_dir / "reports" / "import_report.json").exists()
    assert (tmp_dir / "reports" / "rejection_report.json").exists()


def test_import_rejects_malformed_records(tmp_dir: Path) -> None:
    # 1. Create a CSV with some malformed rows (body too short, missing subject)
    csv_path = tmp_dir / "bad_emails.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "category"])
        # Valid row
        writer.writerow(["alice@example.com", "Valid Subject", "Valid long email body text", "work"])
        # Body too short
        writer.writerow(["bob@example.com", "Valid Subject", "Short", "work"])
        # Missing subject
        writer.writerow(["charlie@example.com", "", "Valid long email body text", "work"])
        # Missing category
        writer.writerow(["dave@example.com", "Valid Subject", "Valid long email body text", ""])

    importer = DataImporter(
        column_mapping={
            "email_id": None,
            "thread_id": None,
            "sender": "sender",
            "subject": "subject",
            "body": "body",
            "label": "category",
        },
        min_body_chars=10,
        report_dir=tmp_dir / "reports",
    )
    result = importer.run(csv_path)

    assert result.total_rows_read == 4
    assert result.valid_records == 1
    assert result.rejected_records == 3
    assert len(result.records) == 1
    assert result.records[0].subject == "Valid Subject"
    
    # Check rejection reasons
    rejection_report_path = tmp_dir / "reports" / "rejection_report.json"
    assert rejection_report_path.exists()
    with rejection_report_path.open("r", encoding="utf-8") as f:
        rep = json.load(f)
        assert rep["total_rejected"] == 3
        assert "body_too_short:<10_chars" in rep["rejection_reasons"]
        assert "missing_required:subject" in rep["rejection_reasons"]
        assert "missing_required:label" in rep["rejection_reasons"]
