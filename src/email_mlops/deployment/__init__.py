"""Deployment (US-6.5).

Automated, repeatable, per-environment deployment that resolves what to ship
via the Model Registry (US-6.4) and gates on Health Checks (US-6.6).
"""

from email_mlops.deployment.config import (
    ComposeConfig,
    EnvironmentConfig,
    EnvironmentNotFoundError,
    HealthGateConfig,
    list_environments,
    load_environment_config,
)
from email_mlops.deployment.deployer import (
    Deployer,
    DeploymentResult,
    DeploymentStatus,
    StepResult,
)
from email_mlops.deployment.runner import (
    CommandOutcome,
    CommandRunner,
    SubprocessRunner,
)

__all__ = [
    "Deployer",
    "DeploymentResult",
    "DeploymentStatus",
    "StepResult",
    "EnvironmentConfig",
    "ComposeConfig",
    "HealthGateConfig",
    "EnvironmentNotFoundError",
    "load_environment_config",
    "list_environments",
    "CommandRunner",
    "CommandOutcome",
    "SubprocessRunner",
]
