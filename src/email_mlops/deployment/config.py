"""Per-environment deployment configuration (US-6.5).

Deployment configuration is *separated per environment*: one YAML file per
environment under ``configs/environments/`` (``dev.yaml``, ``staging.yaml``,
``production.yaml``). Each file is self-contained and describes how to bring
that environment up — which compose files, which env vars, which registered
models to deploy, and which dependencies to health-gate on.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_ENVIRONMENTS_DIR = Path("configs/environments")


class ComposeConfig(BaseModel):
    """How to invoke Docker Compose for this environment."""

    files: list[str] = Field(default_factory=lambda: ["docker-compose.yml"])
    project_name: str | None = None
    build: bool = True
    env: dict[str, str] = Field(default_factory=dict)


class HealthGateConfig(BaseModel):
    """Post-deploy readiness gate — the deploy only succeeds once these pass."""

    timeout_seconds: int = 120
    interval_seconds: int = 5
    dependencies: dict[str, dict] = Field(default_factory=dict)


class EnvironmentConfig(BaseModel):
    """A single environment's deployment definition."""

    environment: str
    compose: ComposeConfig = Field(default_factory=ComposeConfig)
    # Registered model names whose deployment target should be resolved.
    models: list[str] = Field(default_factory=list)
    # If True, a configured model with no deployable version fails the deploy.
    require_models: bool = False
    health: HealthGateConfig = Field(default_factory=HealthGateConfig)


class EnvironmentNotFoundError(FileNotFoundError):
    """Raised when a requested environment has no config file."""


def list_environments(
    environments_dir: str | Path = DEFAULT_ENVIRONMENTS_DIR,
) -> list[str]:
    """Return the names of all available environments (``*.yaml`` stems)."""
    directory = Path(environments_dir)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_environment_config(
    environment: str,
    environments_dir: str | Path = DEFAULT_ENVIRONMENTS_DIR,
) -> EnvironmentConfig:
    """Load and validate the config for ``environment``.

    Raises
    ------
    EnvironmentNotFoundError
        If no ``{environment}.yaml`` exists — with a clear message listing the
        environments that *are* available (surfaces the failure per US-6.5).
    """
    directory = Path(environments_dir)
    path = directory / f"{environment}.yaml"
    if not path.exists():
        available = list_environments(directory)
        raise EnvironmentNotFoundError(
            f"No deployment config for environment '{environment}' at {path}. "
            f"Available environments: {available or '(none found)'}."
        )

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("environment", environment)
    return EnvironmentConfig(**data)
