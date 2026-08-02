"""MLflow-backed model registry.

Production backend that maps this project's :class:`ModelVersion` abstraction
onto the MLflow Model Registry via :class:`mlflow.tracking.MlflowClient`.

Our lineage/metadata fields are stored as MLflow model-version *tags* (MLflow's
native schema only has ``name``/``version``/``current_stage``/``source``), so
they survive a round-trip through :meth:`get_version` / :meth:`list_versions`.
"""

from __future__ import annotations

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from email_mlops.model_registry.base import (
    BaseModelRegistry,
    ModelVersion,
    RegistryStage,
)

# Map our stages to MLflow's stage vocabulary.
_STAGE_TO_MLFLOW = {
    RegistryStage.NONE: "None",
    RegistryStage.STAGING: "Staging",
    RegistryStage.PRODUCTION: "Production",
    RegistryStage.ARCHIVED: "Archived",
}
_MLFLOW_TO_STAGE = {v: k for k, v in _STAGE_TO_MLFLOW.items()}

# Lineage/metadata tag keys used to round-trip our fields through MLflow tags.
_TAG_BASE_MODEL = "lineage.base_model"
_TAG_DATASET_VERSION = "lineage.dataset_version"
_TAG_DESCRIPTION = "description"


class MlflowModelRegistry(BaseModelRegistry):
    """Model registry backed by the MLflow Model Registry.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI (the registry lives alongside it).
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        self._tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        logger.info(f"[MlflowModelRegistry] Initialised — tracking URI: {tracking_uri}")

    # ── Mapping helpers ─────────────────────────────────────────────────────────

    def _to_model_version(self, mv) -> ModelVersion:
        tags = dict(mv.tags) if mv.tags else {}
        metrics: dict[str, float] = {}
        clean_tags: dict[str, str] = {}
        for key, value in tags.items():
            if key.startswith("metric."):
                try:
                    metrics[key[len("metric.") :]] = float(value)
                except (TypeError, ValueError):
                    clean_tags[key] = value
            elif key not in (_TAG_BASE_MODEL, _TAG_DATASET_VERSION, _TAG_DESCRIPTION):
                clean_tags[key] = value

        return ModelVersion(
            name=mv.name,
            version=int(mv.version),
            stage=_MLFLOW_TO_STAGE.get(mv.current_stage, RegistryStage.NONE),
            adapter_path=mv.source,
            source_run_id=mv.run_id or None,
            base_model=tags.get(_TAG_BASE_MODEL),
            dataset_version=tags.get(_TAG_DATASET_VERSION),
            description=tags.get(_TAG_DESCRIPTION) or (mv.description or None),
            metrics=metrics,
            tags=clean_tags,
        )

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
        # Ensure the registered model container exists.
        try:
            self._client.create_registered_model(name)
            logger.info(f"[MlflowModelRegistry] Created registered model '{name}'.")
        except mlflow.exceptions.MlflowException:
            # Already exists — that's fine, we're just adding a version.
            pass

        all_tags: dict[str, str] = {k: str(v) for k, v in (tags or {}).items()}
        if base_model:
            all_tags[_TAG_BASE_MODEL] = base_model
        if dataset_version:
            all_tags[_TAG_DATASET_VERSION] = dataset_version
        if description:
            all_tags[_TAG_DESCRIPTION] = description
        for mkey, mval in (metrics or {}).items():
            all_tags[f"metric.{mkey}"] = str(mval)

        mv = self._client.create_model_version(
            name=name,
            source=adapter_path,
            run_id=source_run_id,
            tags=all_tags,
        )
        logger.info(f"[MlflowModelRegistry] Registered '{name}' version {mv.version}.")
        return self._to_model_version(mv)

    def list_models(self) -> list[str]:
        return sorted(m.name for m in self._client.search_registered_models())

    def list_versions(self, name: str) -> list[ModelVersion]:
        mvs = self._client.search_model_versions(f"name='{name}'")
        return sorted(
            (self._to_model_version(mv) for mv in mvs),
            key=lambda v: v.version,
        )

    def get_version(self, name: str, version: int) -> ModelVersion:
        try:
            mv = self._client.get_model_version(name=name, version=str(version))
        except mlflow.exceptions.MlflowException as exc:
            raise KeyError(f"Model '{name}' has no version {version}.") from exc
        return self._to_model_version(mv)

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: RegistryStage,
    ) -> ModelVersion:
        stage = RegistryStage(stage)
        mv = self._client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=_STAGE_TO_MLFLOW[stage],
            # MLflow natively archives the incumbent production version for us.
            archive_existing_versions=(stage == RegistryStage.PRODUCTION),
        )
        logger.info(
            f"[MlflowModelRegistry] Transitioned '{name}' version {version} → {stage.value}."
        )
        return self._to_model_version(mv)
