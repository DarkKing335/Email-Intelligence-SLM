"""Factory for model-registry instantiation.

Mirrors :func:`email_mlops.experiment_tracking.factory.create_tracker`: reads
``configs/mlops.yaml`` (or environment overrides) and returns the correct
:class:`BaseModelRegistry` implementation.

Environment variable overrides:
    ``MODEL_REGISTRY_BACKEND`` — ``"mlflow"`` or ``"local"`` (overrides YAML).
    ``MLFLOW_TRACKING_URI``    — MLflow server URI (overrides YAML).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from email_mlops.model_registry.base import BaseModelRegistry
from email_mlops.model_registry.local_registry import LocalModelRegistry


def _load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the MLOps config from YAML, tolerating a missing file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "mlops.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            f"[RegistryFactory] MLOps config not found at {config_path}. "
            "Using environment variables and defaults."
        )
        return {}

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def create_registry(
    config_path: Path | str | None = None,
    backend_override: str | None = None,
) -> BaseModelRegistry:
    """Create and return a model registry instance.

    Resolution order for backend selection:
        1. ``backend_override`` argument (highest priority).
        2. ``MODEL_REGISTRY_BACKEND`` environment variable.
        3. ``model_registry.backend`` key in ``configs/mlops.yaml``.
        4. Default: ``"local"``.
    """
    config = _load_config(config_path)
    registry_cfg = config.get("model_registry", {})

    backend = (
        (
            backend_override
            or os.environ.get("MODEL_REGISTRY_BACKEND")
            or registry_cfg.get("backend", "local")
        )
        .lower()
        .strip()
    )

    logger.info(f"[RegistryFactory] Selected model registry backend: {backend}")

    if backend == "local":
        local_cfg = registry_cfg.get("local", {})
        return LocalModelRegistry(
            root_dir=local_cfg.get("root_dir", "models/registry"),
            pretty_print=local_cfg.get("pretty_print", True),
        )

    if backend == "mlflow":
        # Import lazily so 'local' users don't pay the mlflow import cost.
        from email_mlops.model_registry.mlflow_registry import MlflowModelRegistry

        mlflow_cfg = registry_cfg.get("mlflow", {})
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
            "tracking_uri", "http://localhost:5000"
        )
        return MlflowModelRegistry(tracking_uri=tracking_uri)

    raise ValueError(
        f"Unknown model registry backend '{backend}'. Supported values: 'mlflow', 'local'."
    )
