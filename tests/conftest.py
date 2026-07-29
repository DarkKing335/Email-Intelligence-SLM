"""Pytest fixtures for Email Data Engineering testing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml

from email_data_engineering.domain.models import EmailRecord


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory Path object that is cleaned up after testing."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_config(tmp_dir: Path) -> dict:
    """Return a dictionary matching configs/data_engineering.yaml structure."""
    return {
        "directories": {
            "raw": str(tmp_dir / "raw"),
            "processed": str(tmp_dir / "processed"),
            "samples": str(tmp_dir / "samples"),
            "reports": str(tmp_dir / "reports"),
            "logs": str(tmp_dir / "logs"),
        },
        "source": {
            "primary_csv": str(tmp_dir / "raw" / "emails.csv"),
            "sample_csv": str(tmp_dir / "samples" / "emails_sample.csv"),
            "encoding": "utf-8",
            "chunk_size": 100,
            "column_mapping": {
                "email_id": None,
                "thread_id": None,
                "sender": "sender",
                "subject": "subject",
                "body": "body",
                "label": "category",
            },
        },
        "validation": {
            "required_fields": ["subject", "body", "label"],
            "max_body_chars": 1000,
            "min_body_chars": 5,
            "max_subject_chars": 100,
            "rejection_report": str(tmp_dir / "reports" / "rejection_report.json"),
        },
        "normalization": {
            "taxonomy_file": None,  # Will fallback to standard default
            "unknown_label_strategy": "reject",
            "default_label": "fyi",
            "report_file": str(tmp_dir / "reports" / "normalization_report.json"),
        },
        "cleaning": {
            "deduplication": {
                "remove_exact": True,
                "remove_near_duplicates": True,
                "near_duplicate_threshold": 0.95,
                "dedup_fields": ["subject", "body_text"],
            },
            "text_normalization": {
                "collapse_whitespace": True,
                "normalize_unicode_punctuation": True,
                "strip_html": True,
                "strip_signatures": True,
                "strip_quoted_reply": True,
            },
            "pii_redaction": {
                "enabled": True,
                "replacement_token": "[REDACTED]",
                "patterns": {
                    "email_address": True,
                    "phone_number": True,
                    "credit_card": True,
                    "ssn": True,
                    "ip_address": True,
                },
            },
            "length": {
                "clip_long_bodies": True,
            },
            "report_file": str(tmp_dir / "reports" / "cleaning_report.json"),
        },
        "augmentation": {
            "random_seed": 42,
            "augmentation_ratio": 0.5,
            "strategies": {
                "synonym_substitution": {"enabled": True, "substitution_rate": 0.15},
                "formatting_variation": {"enabled": True, "vary_greeting": True, "vary_closing": True},
                "case_variation": {"enabled": True, "subject_cases": ["title_case", "sentence_case"]},
                "whitespace_variation": {"enabled": True, "vary_paragraph_spacing": True},
            },
            "skip_labels": [],
            "report_file": str(tmp_dir / "reports" / "augmentation_report.json"),
        },
        "splitting": {
            "random_seed": 42,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "stratify": True,
            "report_file": str(tmp_dir / "reports" / "split_report.json"),
        },
        "versioning": {
            "initial_version": "0.1.0",
            "auto_increment": "patch",
            "capture_source_hash": True,
            "capture_config_snapshot": True,
            "report_file": str(tmp_dir / "reports" / "version_report.json"),
        },
        "export": {
            "format": "jsonl",
            "train_filename": "train.jsonl",
            "val_filename": "val.jsonl",
            "test_filename": "test.jsonl",
            "version_manifest_filename": "version.json",
            "atomic_write": True,
        },
        "logging": {
            "level": "DEBUG",
            "log_file": None,
            "rotation": "10 MB",
            "retention": "3 days",
            "json_logs": False,
        },
    }


@pytest.fixture
def mock_config_file(tmp_dir: Path, mock_config: dict) -> Path:
    """Save the mock config dict to a temp YAML file and return its path."""
    config_path = tmp_dir / "data_engineering.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)
    return config_path


@pytest.fixture
def sample_records() -> list[EmailRecord]:
    """Provide a list of diverse pre-normalized email records for stage testing."""
    return [
        EmailRecord(
            email_id="rec-1",
            thread_id="thread-1",
            sender_domain="company.com",
            subject="Urgent action required on project status",
            body_text="Hi Team, please send the verification code to proceed immediately. Thanks, Alice.",
            label="security",
            original_label="verify_code",
            priority="high",
            response_required=True,
            source="test_source",
        ),
        EmailRecord(
            email_id="rec-2",
            thread_id="thread-2",
            sender_domain="shopco.com",
            subject="Upgrade your membership today!",
            body_text="Dear Customer, check out our summer catalog and buy the best items now. Regards, Marketing.",
            label="unsubscribe",
            original_label="promotions",
            priority="low",
            response_required=False,
            source="test_source",
        ),
        EmailRecord(
            email_id="rec-3",
            thread_id="thread-3",
            sender_domain="gmail.com",
            subject="Invoice for July consulting services",
            body_text="Hi, please find your INV-2026-1002 attached. The amount is $2,500.00. Kind regards.",
            label="billing",
            original_label="billing",
            priority="high",
            response_required=True,
            source="test_source",
        ),
        EmailRecord(
            email_id="rec-4",
            thread_id="thread-4",
            sender_domain="support.io",
            subject="Support ticket reset password confirmation",
            body_text="Hello, resetting password for user account charlie@yahoo.com. View details here.",
            label="support",
            original_label="support",
            priority="medium",
            response_required=True,
            source="test_source",
        ),
        EmailRecord(
            email_id="rec-5",
            thread_id="thread-5",
            sender_domain="social.net",
            subject="Jane Doe commented on your status",
            body_text="You have a new comment from Jane. Check it out on your network feed.",
            label="fyi",
            original_label="social_media",
            priority="medium",
            response_required=False,
            source="test_source",
        ),
    ]


@pytest.fixture(autouse=True)
def clean_loguru_handlers():
    """Ensure Loguru handlers are removed after each test so file handles are released on Windows."""
    from loguru import logger
    yield
    logger.remove()

