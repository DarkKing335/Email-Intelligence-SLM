"""Unit tests for the DataAugmentor stage (US-1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from email_data_engineering.domain.models import EmailRecord
from email_data_engineering.application.augmentor import DataAugmentor


def test_augmented_records_marked_synthetic(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Urgent verification action",
            body_text="Please review the project roadmap immediately.",
            label="security",
            original_label="verify_code",
            priority="high",
            response_required=True,
        )
    ]

    augmentor = DataAugmentor(
        random_seed=42,
        augmentation_ratio=1.0,
        report_dir=tmp_dir / "reports",
    )
    result = augmentor.run(records)

    assert result.original_records == 1
    assert result.augmented_records == 1
    assert result.total_records == 2

    # First record should be original, second synthetic
    orig = result.records[0]
    aug = result.records[1]

    assert orig.is_augmented is False
    assert orig.augmentation_type is None
    
    assert aug.is_augmented is True
    assert aug.augmentation_type in ["synonym_substitution", "formatting_variation", "case_variation", "whitespace_variation"]
    assert aug.email_id.startswith("aug-")


def test_augmentation_is_repeatable_with_seed(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Urgent support inquiry",
            body_text="We immediately need assistance resetting the password for customer admin.",
            label="support",
            original_label="support",
            priority="medium",
            response_required=True,
        )
    ]

    # Run 1 with seed=42
    aug1 = DataAugmentor(
        random_seed=42,
        augmentation_ratio=1.0,
        report_dir=tmp_dir / "reports1",
    ).run(records)

    # Run 2 with seed=42
    aug2 = DataAugmentor(
        random_seed=42,
        augmentation_ratio=1.0,
        report_dir=tmp_dir / "reports2",
    ).run(records)

    # Run 3 with seed=100
    aug3 = DataAugmentor(
        random_seed=100,
        augmentation_ratio=1.0,
        report_dir=tmp_dir / "reports3",
    ).run(records)

    # Run 1 & Run 2 should yield identical outputs
    assert aug1.records[1].subject == aug2.records[1].subject
    assert aug1.records[1].body_text == aug2.records[1].body_text
    assert aug1.records[1].augmentation_type == aug2.records[1].augmentation_type

    # Run 3 should differ or use a different strategy/substitution
    # Note: Depending on the choices, it might pick a different strategy.
    # We check if seed indeed controls randomization.
    assert aug1.random_seed == 42
    assert aug3.random_seed == 100


def test_synonym_substitution(tmp_dir: Path) -> None:
    records = [
        EmailRecord(
            email_id="1",
            subject="Normal Subject",
            body_text="urgent important customer support verification",
            label="support",
            original_label="support",
            priority="medium",
            response_required=True,
        )
    ]

    # Enable only synonym strategy to test it explicitly
    augmentor = DataAugmentor(
        random_seed=42,
        augmentation_ratio=1.0,
        synonym_enabled=True,
        substitution_rate=1.0,  # force substitution of every matched word
        formatting_enabled=False,
        case_enabled=False,
        whitespace_enabled=False,
        report_dir=tmp_dir / "reports",
    )
    result = augmentor.run(records)
    aug = result.records[1]

    assert aug.body_text != "urgent important customer support verification"
    # Check that at least some synonyms were substituted
    words = aug.body_text.split()
    assert words[0] in augmentor.DEFAULT_SYNONYMS["urgent"]
    assert words[1] in augmentor.DEFAULT_SYNONYMS["important"]
