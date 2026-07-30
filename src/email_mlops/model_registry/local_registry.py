"""Local JSON-file model registry.

Offline fallback that persists the registry to structured JSON files — one
file per registered model — under a root directory.  Useful for development,
CI, and air-gapped environments where an MLflow server is unavailable.

Layout::

    {root_dir}/
      reply-style-professional.json   ← all versions of this model
      reply-style-friendly.json
      ...

Each file::

    {
        "name": "reply-style-professional",
        "versions": [ {<ModelVersion>}, {<ModelVersion>}, ... ]
    }
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from email_mlops.model_registry.base import (
    BaseModelRegistry,
    ModelVersion,
    RegistryStage,
)

# Model names become filenames, so restrict to a filesystem-safe charset.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class LocalModelRegistry(BaseModelRegistry):
    """Model registry backed by local JSON files.

    Parameters
    ----------
    root_dir:
        Directory where per-model JSON files are written.
    pretty_print:
        If ``True``, JSON files are indented for human readability.
    """

    def __init__(
        self,
        root_dir: str | Path = "models/registry",
        pretty_print: bool = True,
    ) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._pretty_print = pretty_print
        logger.info(f"[LocalModelRegistry] Initialised — root dir: {self._root}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not _SAFE_NAME.match(name):
            raise ValueError(
                f"Invalid model name {name!r}. Use only letters, digits, '.', '_', and '-'."
            )

    def _path_for(self, name: str) -> Path:
        return self._root / f"{name}.json"

    def _read(self, name: str) -> list[ModelVersion]:
        path = self._path_for(name)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [ModelVersion(**v) for v in data.get("versions", [])]

    def _write(self, name: str, versions: list[ModelVersion]) -> None:
        path = self._path_for(name)
        payload = {
            "name": name,
            "versions": [v.model_dump(mode="json") for v in versions],
        }
        indent = 2 if self._pretty_print else None
        # Atomic-ish write via temp file + replace to avoid partial files.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False)
            fh.write("\n")
        tmp.replace(path)

    # ── Registry API ────────────────────────────────────────────────────────────

    def register_model(
        self,
        name: str,
        adapter_path: str,
        *,
        source_run_id: str | None = None,
        base_model: str | None = None,
        dataset_version: str | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> ModelVersion:
        self._validate_name(name)
        versions = self._read(name)
        next_version = (max((v.version for v in versions), default=0)) + 1

        entry = ModelVersion(
            name=name,
            version=next_version,
            stage=RegistryStage.NONE,
            adapter_path=adapter_path,
            source_run_id=source_run_id,
            base_model=base_model,
            dataset_version=dataset_version,
            metrics=dict(metrics) if metrics else {},
            tags=dict(tags) if tags else {},
            description=description,
        )
        versions.append(entry)
        self._write(name, versions)

        logger.info(
            f"[LocalModelRegistry] Registered '{name}' version {next_version} "
            f"(base_model={base_model}, run={source_run_id})"
        )
        return entry

    def list_models(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.json"))

    def list_versions(self, name: str) -> list[ModelVersion]:
        return sorted(self._read(name), key=lambda v: v.version)

    def get_version(self, name: str, version: int) -> ModelVersion:
        for v in self._read(name):
            if v.version == version:
                return v
        raise KeyError(f"Model '{name}' has no version {version}.")

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: RegistryStage,
    ) -> ModelVersion:
        stage = RegistryStage(stage)
        versions = self._read(name)
        if not versions:
            raise KeyError(f"Model '{name}' is not registered.")

        target: ModelVersion | None = None
        now = datetime.now(UTC).isoformat()
        updated: list[ModelVersion] = []
        for v in versions:
            if v.version == version:
                target = v.model_copy(update={"stage": stage, "updated_at": now})
                updated.append(target)
            elif stage == RegistryStage.PRODUCTION and v.stage == RegistryStage.PRODUCTION:
                # Enforce single production version: archive the incumbent.
                updated.append(
                    v.model_copy(update={"stage": RegistryStage.ARCHIVED, "updated_at": now})
                )
                logger.info(
                    f"[LocalModelRegistry] Archived previously-production "
                    f"'{name}' version {v.version}."
                )
            else:
                updated.append(v)

        if target is None:
            raise KeyError(f"Model '{name}' has no version {version}.")

        self._write(name, updated)
        logger.info(
            f"[LocalModelRegistry] Transitioned '{name}' version {version} → {stage.value}."
        )
        return target
