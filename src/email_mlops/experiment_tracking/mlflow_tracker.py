"""MLflow experiment tracker implementation.

Production-grade backend that records parameters, metrics, tags, and
artifacts to an MLflow tracking server (local or remote).
"""

from __future__ import annotations

from typing import Any

import mlflow
from loguru import logger

from email_mlops.experiment_tracking.base import BaseExperimentTracker


class MlflowTracker(BaseExperimentTracker):
    """Experiment tracker backed by MLflow.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking server URI (e.g. ``"http://localhost:5000"``).
    default_experiment:
        Experiment name to use when ``start_run`` is called without
        an explicit experiment name.
    artifact_location:
        Default artifact root for the experiment.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        default_experiment: str = "email-triage-slm",
        artifact_location: str | None = None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._default_experiment = default_experiment
        self._artifact_location = artifact_location
        self._run_id: str | None = None

        # Configure the MLflow client
        mlflow.set_tracking_uri(self._tracking_uri)
        logger.info(
            f"[MlflowTracker] Initialised — tracking URI: {self._tracking_uri}, "
            f"default experiment: {self._default_experiment}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name: str,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        experiment = experiment_name or self._default_experiment

        # Get or create the experiment
        experiment_id = mlflow.get_experiment_by_name(experiment)
        if experiment_id is None:
            experiment_id = mlflow.create_experiment(
                experiment,
                artifact_location=self._artifact_location,
            )
            logger.info(f"[MlflowTracker] Created experiment: {experiment}")
        else:
            experiment_id = experiment_id.experiment_id

        # Start the run
        active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
            tags=tags,
        )
        self._run_id = active_run.info.run_id
        logger.info(
            f"[MlflowTracker] Started run '{run_name}' "
            f"(run_id={self._run_id}, experiment={experiment})"
        )
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        if self._run_id is None:
            logger.warning("[MlflowTracker] end_run called with no active run.")
            return

        mlflow.end_run(status=status)
        logger.info(
            f"[MlflowTracker] Ended run {self._run_id} with status={status}"
        )
        self._run_id = None

    def get_run_id(self) -> str | None:
        return self._run_id

    # ── Parameters ────────────────────────────────────────────────────────────

    def log_parameter(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_parameters(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    # ── Artifacts ─────────────────────────────────────────────────────────────

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(
            f"[MlflowTracker] Logged artifact: {local_path} -> {artifact_path}"
        )

    # ── Tags ──────────────────────────────────────────────────────────────────

    def set_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)
