"""Abstract Base Class and data models for the Model Registry (US-6.4).

The Model Registry stores *approved* model artifacts — in this project, the
fine-tuned LoRA reply-style adapters (``professional`` / ``friendly`` /
``concise``) and any base checkpoints — together with their metadata and
lineage.  Training scripts and deployment tooling interact only with the
:class:`BaseModelRegistry` interface, making the backend (local JSON or an
MLflow Model Registry) fully swappable via the factory.

Acceptance criteria coverage (US-6.4):
    * Register approved model artifacts        → :meth:`register_model`
    * Store metadata + lineage                 → :class:`ModelVersion`
    * Hold multiple versions simultaneously    → :meth:`list_versions`
    * Select a version to deploy               → :meth:`transition_stage` /
                                                 :meth:`get_deployment_target`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class RegistryStage(StrEnum):
    """Lifecycle stage of a registered model version.

    Only one version of a given model may sit in ``PRODUCTION`` at a time;
    promoting a new version archives the previous production version.  This is
    what lets deployment tooling unambiguously *select a version to deploy*.
    """

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelVersion(BaseModel):
    """A single registered model version with metadata and lineage.

    Lineage fields (``source_run_id``, ``base_model``, ``dataset_version``)
    tie a deployable artifact back to the experiment run, base checkpoint, and
    dataset build it originated from — satisfying the "store lineage" criterion.
    """

    name: str = Field(description="Registered model name, e.g. 'reply-style-professional'.")
    version: int = Field(description="Monotonically increasing version number (starts at 1).")
    stage: RegistryStage = Field(
        default=RegistryStage.NONE,
        description="Deployment lifecycle stage.",
    )

    # ── Artifact location ──
    adapter_path: str | None = Field(
        default=None,
        description="Filesystem path or URI of the adapter/checkpoint artifact.",
    )

    # ── Lineage ──
    source_run_id: str | None = Field(
        default=None,
        description="Experiment-tracking run id that produced this artifact.",
    )
    base_model: str | None = Field(
        default=None,
        description="Base model the adapter was trained on, e.g. 'Qwen3-4B'.",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Dataset build version used for training, e.g. 'v0.1.0'.",
    )

    # ── Free-form metadata ──
    description: str | None = Field(default=None, description="Human-readable notes.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Snapshot of evaluation metrics at registration time.",
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Arbitrary key-value tags.")

    # ── Timestamps ──
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)


class BaseModelRegistry(ABC):
    """Contract for all model-registry backends.

    Implementations must be safe to call without an active network connection
    for read paths where possible, and must not raise on a simply-empty
    registry (return empty lists / ``None`` instead).
    """

    @abstractmethod
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
        """Register a new version of an approved model artifact.

        A brand-new model name creates version ``1``; subsequent calls append
        incrementing versions.  Returns the created :class:`ModelVersion`.
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return the names of all registered models."""

    @abstractmethod
    def list_versions(self, name: str) -> list[ModelVersion]:
        """Return every registered version of ``name``, ascending by version."""

    @abstractmethod
    def get_version(self, name: str, version: int) -> ModelVersion:
        """Return a specific version, or raise :class:`KeyError` if absent."""

    @abstractmethod
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: RegistryStage,
    ) -> ModelVersion:
        """Move a version to a new lifecycle stage.

        Promoting a version to :attr:`RegistryStage.PRODUCTION` must archive any
        version currently in production for the same model, so that at most one
        production version exists at a time.
        """

    def get_latest_version(
        self,
        name: str,
        stage: RegistryStage | None = None,
    ) -> ModelVersion | None:
        """Return the highest-numbered version, optionally filtered by ``stage``.

        Default implementation is expressed in terms of :meth:`list_versions`;
        backends may override for efficiency.
        """
        versions = self.list_versions(name)
        if stage is not None:
            versions = [v for v in versions if v.stage == stage]
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)

    def get_deployment_target(self, name: str) -> ModelVersion | None:
        """Return the version that should be deployed for ``name``.

        Resolution order: the production version if one exists, otherwise the
        latest staging version, otherwise ``None``.  This is the single entry
        point deployment tooling (US-6.5) calls to pick what to ship.
        """
        production = self.get_latest_version(name, stage=RegistryStage.PRODUCTION)
        if production is not None:
            return production
        return self.get_latest_version(name, stage=RegistryStage.STAGING)
