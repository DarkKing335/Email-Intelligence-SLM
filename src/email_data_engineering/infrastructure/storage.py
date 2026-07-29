"""Versioned dataset storage for the Email Data Engineering pipeline.

Manages reading and writing versioned JSONL split files and associated
``version.json`` metadata files under ``data/processed/``.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from email_data_engineering.domain.models import DatasetVersion, EmailRecord


class DatasetStorage:
    """Read/write versioned dataset artifacts to the local filesystem.

    Directory layout::

        {base_dir}/
          v0.1.0/
            train.jsonl
            val.jsonl
            test.jsonl
            version.json
          v0.2.0/
            ...

    Parameters
    ----------
    base_dir:
        Root directory for processed dataset versions (``data/processed``).
    """

    SPLIT_FILES = {
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
    }
    VERSION_MANIFEST = "version.json"

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Writing ───────────────────────────────────────────────────────────────

    def write_split(
        self,
        records: list[EmailRecord],
        version: str,
        split: str,
    ) -> Path:
        """Write *records* for *split* to the versioned directory.

        Writes are atomic: data is written to a ``.tmp`` file first and
        then renamed to the final filename.

        Returns
        -------
        Path
            The final path of the written JSONL file.
        """
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        filename = self.SPLIT_FILES.get(split, f"{split}.jsonl")
        final_path = version_dir / filename
        tmp_path = version_dir / f".{uuid.uuid4().hex}.tmp"

        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                for record in records:
                    line = json.dumps(record.to_dict(), ensure_ascii=False)
                    fh.write(line + "\n")
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Written split file",
            version=version,
            split=split,
            records=len(records),
            path=str(final_path),
        )
        return final_path

    def write_version_manifest(self, dataset_version: DatasetVersion) -> Path:
        """Write the ``version.json`` metadata file for a dataset version."""
        version_dir = self._version_dir(dataset_version.version)
        version_dir.mkdir(parents=True, exist_ok=True)

        final_path = version_dir / self.VERSION_MANIFEST
        tmp_path = version_dir / f".{uuid.uuid4().hex}.tmp"

        try:
            data = dataset_version.to_dict()
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_path, final_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "Written version manifest",
            version=dataset_version.version,
            path=str(final_path),
        )
        return final_path

    # ── Reading ───────────────────────────────────────────────────────────────

    def read_split(self, version: str, split: str) -> list[dict[str, Any]]:
        """Read records for a given *version* and *split*.

        Returns
        -------
        list[dict]
            Raw dicts (not EmailRecord objects) for flexibility.
        """
        version_dir = self._version_dir(version)
        filename = self.SPLIT_FILES.get(split, f"{split}.jsonl")
        path = version_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                f"Available versions: {self.list_versions()}"
            )

        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.debug(f"Read {len(records)} records from {path}")
        return records

    def read_version_manifest(self, version: str) -> dict[str, Any]:
        """Read and return the ``version.json`` metadata for *version*."""
        path = self._version_dir(version) / self.VERSION_MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"Version manifest not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ── Discovery ─────────────────────────────────────────────────────────────

    def list_versions(self) -> list[str]:
        """Return a sorted list of all available version strings."""
        versions = []
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                manifest = child / self.VERSION_MANIFEST
                if manifest.exists():
                    versions.append(child.name)
        return sorted(versions)

    def get_latest_version(self) -> str | None:
        """Return the lexicographically latest version string, or None."""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def version_exists(self, version: str) -> bool:
        return (self._version_dir(version) / self.VERSION_MANIFEST).exists()

    # ── Comparison ───────────────────────────────────────────────────────────

    def compare_versions(
        self, version_a: str, version_b: str
    ) -> dict[str, Any]:
        """Compare two version manifests and return a diff dict."""
        manifest_a = self.read_version_manifest(version_a)
        manifest_b = self.read_version_manifest(version_b)

        def _diff(a: Any, b: Any) -> Any:
            if isinstance(a, dict) and isinstance(b, dict):
                keys = set(a) | set(b)
                return {k: _diff(a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
            return {"before": a, "after": b}

        return {
            "version_a": version_a,
            "version_b": version_b,
            "diff": _diff(manifest_a, manifest_b),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _version_dir(self, version: str) -> Path:
        """Return the directory path for a version string.

        Normalises version strings so that both ``0.1.0`` and ``v0.1.0``
        resolve to the same ``v0.1.0`` directory.
        """
        clean = version.lstrip("v")
        return self.base_dir / f"v{clean}"
