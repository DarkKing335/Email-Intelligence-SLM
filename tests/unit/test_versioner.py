"""Unit tests for the DataVersioner stage (US-1.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from email_data_engineering.domain.models import EmailRecord
from email_data_engineering.application.versioner import DataVersioner
from email_data_engineering.infrastructure.storage import DatasetStorage


def test_version_increments_correctly(tmp_dir: Path) -> None:
    storage_dir = tmp_dir / "processed"
    storage_dir.mkdir()

    # Define patch increment versioner
    versioner_patch = DataVersioner(storage_dir=storage_dir, auto_increment="patch")
    
    # 1. No prior versions -> uses initial version
    v1 = versioner_patch.determine_version()
    assert v1 == "v0.1.0"

    # Create dummy mock version folder
    storage = DatasetStorage(storage_dir)
    # Write a mock manifest to register v0.1.0
    (storage_dir / "v0.1.0").mkdir()
    with (storage_dir / "v0.1.0" / "version.json").open("w") as f:
        json.dump({"version": "v0.1.0"}, f)

    # 2. Patch increment from v0.1.0 -> v0.1.1
    v2 = versioner_patch.determine_version()
    assert v2 == "v0.1.1"

    # Write mock v0.1.1
    (storage_dir / "v0.1.1").mkdir()
    with (storage_dir / "v0.1.1" / "version.json").open("w") as f:
        json.dump({"version": "v0.1.1"}, f)

    # 3. Minor increment from v0.1.1 -> v0.2.0
    versioner_minor = DataVersioner(storage_dir=storage_dir, auto_increment="minor")
    v3 = versioner_minor.determine_version()
    assert v3 == "v0.2.0"

    # Write mock v0.2.0
    (storage_dir / "v0.2.0").mkdir()
    with (storage_dir / "v0.2.0" / "version.json").open("w") as f:
        json.dump({"version": "v0.2.0"}, f)

    # 4. Major increment from v0.2.0 -> v1.0.0
    versioner_major = DataVersioner(storage_dir=storage_dir, auto_increment="major")
    v4 = versioner_major.determine_version()
    assert v4 == "v1.0.0"


def test_version_metadata_captured(tmp_dir: Path) -> None:
    source_file = tmp_dir / "emails.csv"
    source_file.write_text("dummy csv file content", encoding="utf-8")

    versioner = DataVersioner(storage_dir=tmp_dir / "processed", report_dir=tmp_dir / "reports")
    
    metadata = versioner.build_metadata(
        version="v0.1.0",
        source_path=source_file,
        train_count=100,
        val_count=10,
        test_count=10,
        label_dist={"train": {"fyi": 100}, "val": {"fyi": 10}, "test": {"fyi": 10}},
        preprocessing_steps=["import", "validate", "clean", "split", "version"],
        random_seed=42,
        config_snapshot={"some_config_key": "some_value"},
    )

    # Assertions on fields
    assert metadata.version == "v0.1.0"
    assert metadata.source_path == str(source_file)
    assert metadata.source_hash != ""  # SHA-256 hash computed
    assert metadata.split_counts.train == 100
    assert metadata.split_counts.val == 10
    assert metadata.split_counts.test == 10
    assert metadata.random_seed == 42
    assert metadata.preprocessing_steps == ["import", "validate", "clean", "split", "version"]
    assert metadata.config_snapshot == {"some_config_key": "some_value"}
    
    # Check that a version report was written
    assert (tmp_dir / "reports" / "version_report.json").exists()
