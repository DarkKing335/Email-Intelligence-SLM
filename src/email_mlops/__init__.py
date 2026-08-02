"""Email MLOps — Experiment Tracking & Infrastructure Utilities.

Provides a clean, backend-agnostic interface for experiment tracking,
model registry, health checks, and deployment.
Supports MLflow for production and local fallbacks for offline development.
"""

from email_mlops.deployment import (
    Deployer,
    DeploymentResult,
    DeploymentStatus,
    load_environment_config,
)
from email_mlops.experiment_tracking.base import BaseExperimentTracker
from email_mlops.experiment_tracking.factory import create_tracker
from email_mlops.health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    Probe,
)
from email_mlops.model_registry import (
    BaseModelRegistry,
    ModelVersion,
    RegistryStage,
    create_registry,
)

__all__ = [
    # Experiment tracking (US-6.3)
    "create_tracker",
    "BaseExperimentTracker",
    # Model registry (US-6.4)
    "create_registry",
    "BaseModelRegistry",
    "ModelVersion",
    "RegistryStage",
    # Health checks (US-6.6)
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "Probe",
    # Deployment (US-6.5)
    "Deployer",
    "DeploymentResult",
    "DeploymentStatus",
    "load_environment_config",
]
