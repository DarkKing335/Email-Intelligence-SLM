"""Versioner stage — US-1.5.

Manages dataset versions, increments versions using semantic versioning,
generates version manifest data, and handles version comparisons.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from loguru import logger

from email_data_engineering.domain.models import DatasetVersion, SplitManifest
from email_data_engineering.infrastructure.storage import DatasetStorage


class DataVersioner:
    """Versioner stage: handles semantic versioning and metadata extraction.

    Implements US-1.5:
    - Automatically manages versions (major, minor, patch increments).
    - Captures file hash, config snapshot, and split details.
    - Persists audit metadata.
    """

    def __init__(
        self,
        storage_dir: str | Path = "data/processed",
        initial_version: str = "0.1.0",
        auto_increment: str = "patch",
        capture_source_hash: bool = True,
        capture_config_snapshot: bool = True,
        report_dir: str | Path = "reports",
    ) -> None:
        self._storage = DatasetStorage(storage_dir)
        self._initial_version = initial_version.lstrip("v")
        self._auto_increment = auto_increment.lower()
        self._capture_hash = capture_source_hash
        self._capture_config = capture_config_snapshot
        self._report_dir = Path(report_dir)

    def determine_version(self, requested_version: str | None = None) -> str:
        """Decide the version string to use.

        If requested_version is provided, uses that.
        Otherwise, auto-increments the latest version found in storage.
        If no versions exist, uses the initial_version.
        """
        if requested_version:
            # Normalize to clean string
            v = requested_version.lstrip("v")
            logger.info(f"[Version] Using user-requested version: v{v}")
            return f"v{v}"

        latest = self._storage.get_latest_version()
        if not latest:
            logger.info(f"[Version] No prior versions found. Using initial: v{self._initial_version}")
            return f"v{self._initial_version}"

        # Clean latest version
        clean_latest = latest.lstrip("v")
        parts = clean_latest.split(".")
        if len(parts) != 3:
            logger.warning(
                f"[Version] Latest version '{latest}' is not semver. "
                f"Defaulting to initial version v{self._initial_version}"
            )
            return f"v{self._initial_version}"

        try:
            major, minor, patch = map(int, parts)
            if self._auto_increment == "major":
                major += 1
                minor = 0
                patch = 0
            elif self._auto_increment == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1

            new_v = f"v{major}.{minor}.{patch}"
            logger.info(f"[Version] Auto-incremented version from {latest} to {new_v}")
            return new_v
        except ValueError:
            logger.warning(
                f"[Version] Parsing latest version '{latest}' failed. "
                f"Defaulting to initial version v{self._initial_version}"
            )
            return f"v{self._initial_version}"

    def build_metadata(
        self,
        version: str,
        source_path: str | Path,
        train_count: int,
        val_count: int,
        test_count: int,
        label_dist: dict[str, dict[str, int]],
        preprocessing_steps: list[str],
        random_seed: int,
        config_snapshot: dict[str, Any],
    ) -> DatasetVersion:
        """Create a DatasetVersion domain object populated with full metadata."""
        src_path = Path(source_path)
        src_hash = ""
        if self._capture_hash and src_path.exists() and src_path.is_file():
            src_hash = self._hash_file(src_path)
            logger.debug(f"[Version] Computed source hash for {src_path.name}: {src_hash}")

        split_counts = SplitManifest(
            train=train_count,
            val=val_count,
            test=test_count,
        )

        dataset_version = DatasetVersion(
            version=version,
            source_path=str(source_path),
            source_hash=src_hash,
            split_counts=split_counts,
            label_distribution=label_dist,
            preprocessing_steps=preprocessing_steps,
            random_seed=random_seed,
            config_snapshot=config_snapshot if self._capture_config else {},
        )

        self._write_report(dataset_version)
        return dataset_version

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = sha256()
        with path.open("rb") as f:
            # Read in 64kb chunks
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_report(self, ver: DatasetVersion) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "stage": "version",
            "version": ver.version,
            "created_at": ver.created_at.isoformat(),
            "source_path": ver.source_path,
            "source_hash": ver.source_hash,
            "split_counts": ver.split_counts.as_dict(),
            "preprocessing_steps": ver.preprocessing_steps,
            "random_seed": ver.random_seed,
        }
        path = self._report_dir / "version_report.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.debug(f"Version report written: {path}")
