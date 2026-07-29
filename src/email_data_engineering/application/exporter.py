"""Exporter stage.

Writes the final split records to versioned JSONL files and writes the
associated version manifest using :class:`DatasetStorage`.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

from email_data_engineering.domain.models import DatasetVersion, EmailRecord
from email_data_engineering.infrastructure.storage import DatasetStorage


class DataExporter:
    """Exporter stage: write finalized splits and metadata to versioned paths.

    Delegates filesystem operations to the infrastructure storage service.
    """

    def __init__(self, storage_dir: str | Path = "data/processed") -> None:
        self._storage = DatasetStorage(storage_dir)

    def run(
        self,
        version: str,
        train_records: list[EmailRecord],
        val_records: list[EmailRecord],
        test_records: list[EmailRecord],
        metadata: DatasetVersion,
    ) -> dict[str, str]:
        """Export all splits and version metadata to disk.

        Returns
        -------
        dict[str, str]
            A map of split/manifest name to the absolute file path written.
        """
        logger.info(f"[Export] Starting export for version: {version}")

        # Update records with their final assigned version
        updated_train = [r.with_updates(version=version) for r in train_records]
        updated_val = [r.with_updates(version=version) for r in val_records]
        updated_test = [r.with_updates(version=version) for r in test_records]

        # Write splits
        train_path = self._storage.write_split(updated_train, version, "train")
        val_path = self._storage.write_split(updated_val, version, "val")
        test_path = self._storage.write_split(updated_test, version, "test")

        # Write version manifest metadata
        manifest_path = self._storage.write_version_manifest(metadata)

        logger.info(f"[Export] Export completed successfully for {version}")
        
        return {
            "train": str(train_path.absolute()),
            "val": str(val_path.absolute()),
            "test": str(test_path.absolute()),
            "manifest": str(manifest_path.absolute()),
        }
