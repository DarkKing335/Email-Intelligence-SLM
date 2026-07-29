"""Unit tests for the LabelNormalizer stage (US-1.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from email_data_engineering.domain.models import EmailRecord
from email_data_engineering.domain.taxonomy import LabelTaxonomy
from email_data_engineering.application.normalizer import LabelNormalizer


def test_normalize_known_labels(tmp_dir: Path) -> None:
    # 1. Create a list of records with raw category labels
    records = [
        EmailRecord(
            email_id="1",
            subject="Verification",
            body_text="Your code is 1234",
            label="verify_code",  # raw label mapped to security
            original_label="verify_code",
            priority="medium",
            response_required=False,
        ),
        EmailRecord(
            email_id="2",
            subject="Weekly newsletter",
            body_text="Check out these deals!",
            label="promotions",  # raw label mapped to unsubscribe
            original_label="promotions",
            priority="medium",
            response_required=False,
        ),
    ]

    normalizer = LabelNormalizer(
        taxonomy=LabelTaxonomy(),
        unknown_label_strategy="reject",
        report_dir=tmp_dir / "reports",
    )
    result = normalizer.run(records)

    assert result.total_records == 2
    assert result.normalized_records == 2
    assert result.rejected_unknown_label == 0
    assert len(result.records) == 2

    # Verify security mapping
    assert result.records[0].label == "security"
    assert result.records[0].original_label == "verify_code"
    assert result.records[0].priority == "high"
    assert result.records[0].response_required is True

    # Verify unsubscribe mapping
    assert result.records[1].label == "unsubscribe"
    assert result.records[1].original_label == "promotions"
    assert result.records[1].priority == "low"
    assert result.records[1].response_required is False


def test_normalize_rejects_unknown_labels(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Spam email",
            body_text="Buy replica watches now!",
            label="spam",  # mapped to spam
            original_label="spam",
            priority="medium",
            response_required=False,
        ),
        EmailRecord(
            email_id="2",
            subject="Weird category",
            body_text="Some random email text.",
            label="weird_xyz_cat",  # unknown label
            original_label="weird_xyz_cat",
            priority="medium",
            response_required=False,
        ),
    ]

    normalizer = LabelNormalizer(
        taxonomy=LabelTaxonomy(),
        unknown_label_strategy="reject",
        report_dir=tmp_dir / "reports",
    )
    result = normalizer.run(records)

    assert result.total_records == 2
    assert result.normalized_records == 1
    assert result.rejected_unknown_label == 1
    assert "weird_xyz_cat" in result.unmapped_labels


def test_normalize_default_strategy(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Unknown label email",
            body_text="Will map to default category.",
            label="unknown_label",
            original_label="unknown_label",
            priority="medium",
            response_required=False,
        )
    ]

    normalizer = LabelNormalizer(
        taxonomy=LabelTaxonomy(),
        unknown_label_strategy="default",
        default_label="fyi",
        report_dir=tmp_dir / "reports",
    )
    result = normalizer.run(records)

    assert result.total_records == 1
    assert result.normalized_records == 1
    assert result.rejected_unknown_label == 0
    assert result.records[0].label == "fyi"
    assert result.records[0].priority == "medium"
    assert result.records[0].response_required is False
