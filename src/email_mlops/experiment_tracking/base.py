"""Abstract Base Class for experiment tracking.

Defines the universal contract that every tracking backend must implement.
Pipeline stages and training scripts interact only with this interface,
making the tracking backend fully swappable via the factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExperimentTracker(ABC):
    """Contract for all experiment tracking backends.

    Lifecycle:
        1. ``start_run(run_name)`` — Open a new tracking run.
        2. ``log_parameter`` / ``log_parameters`` — Record hyperparameters.
        3. ``log_metric`` / ``log_metrics`` — Record scalar metrics.
        4. ``set_tags`` — Attach key-value metadata tags.
        5. ``log_artifact`` — Upload a local file as an artifact.
        6. ``end_run()`` — Close and finalise the run.

    Implementations should be safe for repeated calls (idempotent where
    possible) and should not raise on missing optional fields.
    """

    @abstractmethod
    def start_run(
        self,
        run_name: str,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Begin a new experiment run.

        Parameters
        ----------
        run_name:
            Human-readable name for this run (e.g. ``"train-v0.2.0-lora-r16"``).
        experiment_name:
            Logical experiment group.  If ``None``, uses the default
            experiment configured in ``configs/mlops.yaml``.
        tags:
            Optional key-value tags to attach at run creation.

        Returns
        -------
        str
            A unique run identifier (backend-specific).
        """

    @abstractmethod
    def log_parameter(self, key: str, value: Any) -> None:
        """Log a single hyperparameter.

        Parameters
        ----------
        key:
            Parameter name (e.g. ``"learning_rate"``).
        value:
            Parameter value.  Will be serialised to string if needed.
        """

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Log multiple parameters at once.

        Default implementation iterates over ``log_parameter``; backends
        may override for batch efficiency.
        """
        for key, value in params.items():
            self.log_parameter(key, value)

    @abstractmethod
    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a single scalar metric.

        Parameters
        ----------
        key:
            Metric name (e.g. ``"val_accuracy"``).
        value:
            Metric value.
        step:
            Optional training step or epoch number.
        """

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Default implementation iterates over ``log_metric``; backends
        may override for batch efficiency.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    @abstractmethod
    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        """Upload a local file or directory as a run artifact.

        Parameters
        ----------
        local_path:
            Path to the file or directory on the local filesystem.
        artifact_path:
            Optional sub-directory within the artifact store.
        """

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """Attach key-value metadata tags to the active run.

        Parameters
        ----------
        tags:
            Dictionary of tag names to values.
        """

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """Finalise and close the active run.

        Parameters
        ----------
        status:
            Final run status (e.g. ``"FINISHED"``, ``"FAILED"``).
        """

    @abstractmethod
    def get_run_id(self) -> str | None:
        """Return the active run identifier, or ``None`` if no run is active."""
