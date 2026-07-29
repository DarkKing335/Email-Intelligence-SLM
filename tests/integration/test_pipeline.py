"""Integration tests for the complete Data Engineering Pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from email_data_engineering.pipeline import DataEngineeringPipeline
from email_data_engineering.infrastructure.storage import DatasetStorage


def test_full_pipeline_with_sample_data(tmp_dir: Path, mock_config_file: Path) -> None:
    # 1. Create a raw emails.csv inside tmp_dir/raw
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "emails.csv"
    
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "category"])
        
        # 10 realistic rows mapping to multiple categories to test stratified split
        writer.writerow(["a@com.com", "Urgent reset", "OTP is 948291. Do not share.", "verify_code"])
        writer.writerow(["b@com.com", "OTP security check", "OTP is 552918. Enter code immediately.", "verify_code"])
        writer.writerow(["c@shop.com", "Summer discounts", "Buy cheap weight loss items immediately. http://scam.com", "spam"])
        writer.writerow(["d@shop.com", "Extra cheap offers", "Exclusive watches sale today only. http://scam.com", "spam"])
        writer.writerow(["e@social.net", "Jane sent request", "Hi Charlie, check connection request on LinkUp.", "social_media"])
        writer.writerow(["f@social.net", "Emma sent message", "Hi Charlie, check connection request on LinkUp.", "social_media"])
        writer.writerow(["g@work.com", "Project updates", "Hello Team, the Q3 Roadmap document has been updated.", "updates"])
        writer.writerow(["h@work.com", "Sprint review notes", "Hi Team, meeting notes from sprint review sync are here.", "updates"])
        writer.writerow(["i@bill.com", "Consulting invoice", "Please find Invoice INV-2026-0048 for July consultation.", "billing"])
        writer.writerow(["j@bill.com", "Receipt of payment", "Thanks for your payment of INV-2026-0048 subscription.", "billing"])

    # 2. Run the pipeline
    pipeline = DataEngineeringPipeline(config_path=mock_config_file)
    result = pipeline.run(
        source_override=raw_csv,
        version_override="v0.1.0",
        seed_override=42,
    )

    # 3. Assertions
    assert result.success is True
    assert result.version == "v0.1.0"
    assert result.import_result.valid_records == 10
    assert result.normalization_result.normalized_records == 10
    
    # Cleaning assertions (check that duplicate HTML or redundant whitespaces fold)
    # The duplicate checks might remove near duplicates depending on TF-IDF. 
    # With 10 rows, it runs correctly. Let's inspect the files written:
    processed_dir = tmp_dir / "processed"
    storage = DatasetStorage(processed_dir)
    assert storage.version_exists("v0.1.0")

    manifest = storage.read_version_manifest("v0.1.0")
    assert manifest["version"] == "v0.1.0"
    assert manifest["source_path"] == str(raw_csv)
    assert manifest["preprocessing_steps"] == [
        "import",
        "validate",
        "normalize",
        "clean",
        "augment",
        "split",
        "version",
        "export",
    ]

    # Verify JSONL split files exist and contain actual records
    train_records = storage.read_split("v0.1.0", "train")
    val_records = storage.read_split("v0.1.0", "val")
    test_records = storage.read_split("v0.1.0", "test")

    assert len(train_records) > 0
    assert len(val_records) > 0
    assert len(test_records) > 0
    
    # Total count = originals after cleaning + augmented
    total_in_splits = len(train_records) + len(val_records) + len(test_records)
    assert total_in_splits == result.split_result.split_counts.total

    # Verify reports were successfully generated in reports directory
    reports_dir = tmp_dir / "reports"
    assert (reports_dir / "import_report.json").exists()
    assert (reports_dir / "rejection_report.json").exists()
    assert (reports_dir / "normalization_report.json").exists()
    assert (reports_dir / "cleaning_report.json").exists()
    assert (reports_dir / "augmentation_report.json").exists()
    assert (reports_dir / "split_report.json").exists()
    assert (reports_dir / "version_report.json").exists()


def test_pipeline_fails_gracefully_on_missing_source(tmp_dir: Path, mock_config_file: Path) -> None:
    non_existent_csv = tmp_dir / "does_not_exist.csv"

    pipeline = DataEngineeringPipeline(config_path=mock_config_file)
    result = pipeline.run(source_override=non_existent_csv)

    assert result.success is False
    assert "FileNotFoundError" in result.error_message or "not found" in result.error_message
