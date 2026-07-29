"""Factory for experiment tracker instantiation.

Reads ``configs/mlops.yaml`` (or environment overrides) and returns the
correct :class:`BaseExperimentTracker` implementation.

Environment variable overrides:
    ``TRACKING_BACKEND``  — ``"mlflow"`` or ``"local"`` (overrides YAML).
    ``MLFLOW_TRACKING_URI`` — MLflow server URI (overrides YAML).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from email_mlops.experiment_tracking.base import BaseExperimentTracker
from email_mlops.experiment_tracking.mlflow_tracker import MlflowTracker
from email_mlops.experiment_tracking.local_tracker import LocalTracker


def _load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the MLOps config from YAML.

    Falls back to an empty dict if the file is not found, so that
    environment variable overrides can still drive the factory.
    """
    if config_path is None:
        # Resolve relative to the project root (4 levels up from this file)
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "configs"
            / "mlops.yaml"
        )
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            f"[TrackerFactory] MLOps config not found at {config_path}. "
            "Using environment variables and defaults."
        )
        return {}

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def create_tracker(
    config_path: Path | str | None = None,
    backend_override: str | None = None,
) -> BaseExperimentTracker:
    """Create and return an experiment tracker instance.

    Resolution order for backend selection:
        1. ``backend_override`` argument (highest priority).
        2. ``TRACKING_BACKEND`` environment variable.
        3. ``tracking.backend`` key in ``configs/mlops.yaml``.
        4. Default: ``"local"``.

    Parameters
    ----------
    config_path:
        Path to the MLOps YAML config file.  ``None`` uses the default
        ``configs/mlops.yaml``.
    backend_override:
        Explicitly select ``"mlflow"`` or ``"local"``.

    Returns
    -------
    BaseExperimentTracker
        A ready-to-use tracker instance.
    """
    config = _load_config(config_path)
    tracking_cfg = config.get("tracking", {})

    # Determine backend
    backend = (
        backend_override
        or os.environ.get("TRACKING_BACKEND")
        or tracking_cfg.get("backend", "local")
    ).lower().strip()

    logger.info(f"[TrackerFactory] Selected tracking backend: {backend}")

    if backend == "mlflow":
        mlflow_cfg = tracking_cfg.get("mlflow", {})
        tracking_uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or mlflow_cfg.get("tracking_uri", "http://localhost:5000")
        )
        default_experiment = mlflow_cfg.get(
            "default_experiment", "email-triage-slm"
        )
        artifact_location = mlflow_cfg.get("artifact_location")

        return MlflowTracker(
            tracking_uri=tracking_uri,
            default_experiment=default_experiment,
            artifact_location=artifact_location,
        )

    elif backend == "local":
        local_cfg = tracking_cfg.get("local", {})
        output_dir = local_cfg.get("output_dir", "reports/experiments")
        pretty_print = local_cfg.get("pretty_print", True)

        return LocalTracker(
            output_dir=output_dir,
            pretty_print=pretty_print,
        )

    else:
        raise ValueError(
            f"Unknown tracking backend '{backend}'. "
            "Supported values: 'mlflow', 'local'."
        )
