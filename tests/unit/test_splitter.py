"""Unit tests for the DataSplitter stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from email_data_engineering.domain.models import EmailRecord
from email_data_engineering.application.splitter import DataSplitter


def test_split_ratios_correct(tmp_dir: Path) -> None:
    # 1. Generate 20 dummy records
    records = []
    for i in range(20):
        records.append(
            EmailRecord(
                email_id=str(i),
                subject=f"Subject {i}",
                body_text=f"This is a long enough body text for index {i}.",
                label="fyi" if i % 2 == 0 else "security",
                original_label="updates",
                priority="medium",
                response_required=False,
            )
        )

    # 2. Run splitter with 80/10/10 ratios
    splitter = DataSplitter(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        stratify=True,
        report_dir=tmp_dir / "reports",
    )
    result = splitter.run(records)

    # 3. Assertions
    assert result.total_records == 20
    assert result.split_counts.train == 16
    assert result.split_counts.val == 2
    assert result.split_counts.test == 2
    assert len(result.train_records) == 16
    assert len(result.val_records) == 2
    assert len(result.test_records) == 2

    # Check that splits have correct split tag assigned
    assert all(r.split == "train" for r in result.train_records)
    assert all(r.split == "val" for r in result.val_records)
    assert all(r.split == "test" for r in result.test_records)


def test_split_is_stratified_by_label(tmp_dir: Path) -> None:
    # Build class-imbalanced records: 80% class A, 20% class B
    records = []
    for i in range(50):
        label = "fyi" if i < 40 else "security"
        records.append(
            EmailRecord(
                email_id=str(i),
                subject=f"Subject {i}",
                body_text=f"Body text for record index {i}.",
                label=label,
                original_label="updates",
                priority="medium",
                response_required=False,
            )
        )

    splitter = DataSplitter(
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        random_seed=42,
        stratify=True,
        report_dir=tmp_dir / "reports",
    )
    result = splitter.run(records)

    # Class distribution check:
    # 40 FYI, 10 Security.
    # Train should have 40 * 0.8 = 32 FYI, 10 * 0.8 = 8 Security.
    train_dist = result.label_distribution["train"]
    assert train_dist["fyi"] == 32
    assert train_dist["security"] == 8

    val_dist = result.label_distribution["val"]
    assert val_dist["fyi"] == 4
    assert val_dist["security"] == 1

    test_dist = result.label_distribution["test"]
    assert test_dist["fyi"] == 4
    assert test_dist["security"] == 1


def test_split_is_deterministic_with_seed(tmp_dir: Path) -> None:
    records = []
    for i in range(30):
        records.append(
            EmailRecord(
                email_id=str(i),
                subject=f"Subject {i}",
                body_text=f"This is a long enough body text for index {i}.",
                label="fyi" if i % 2 == 0 else "security",
                original_label="updates",
                priority="medium",
                response_required=False,
            )
        )

    res1 = DataSplitter(random_seed=42, report_dir=tmp_dir / "rep1").run(records)
    res2 = DataSplitter(random_seed=42, report_dir=tmp_dir / "rep2").run(records)
    res3 = DataSplitter(random_seed=100, report_dir=tmp_dir / "rep3").run(records)

    # Identical seeds -> identical split membership
    set1 = {r.email_id for r in res1.train_records}
    set2 = {r.email_id for r in res2.train_records}
    set3 = {r.email_id for r in res3.train_records}

    assert set1 == set2
    assert set1 != set3  # Different seeds -> different members
