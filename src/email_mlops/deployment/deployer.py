"""Deployment orchestrator (US-6.5).

Runs an automated, repeatable deployment pipeline for a chosen environment:

    1. **load-config**    — resolve the per-environment deployment config.
    2. **resolve-models** — ask the Model Registry (US-6.4) which version of
                            each configured model to ship.
    3. **compose-up**     — bring the services up via ``docker compose``.
    4. **health-gate**    — poll readiness (US-6.6) until the system is
                            accessible, or fail on timeout.

Every step is captured in a structured :class:`DeploymentResult`, so a failure
at any stage is surfaced clearly (which step, and the captured error output).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from loguru import logger

from email_mlops.deployment.config import (
    EnvironmentConfig,
    EnvironmentNotFoundError,
    load_environment_config,
)
from email_mlops.deployment.runner import CommandRunner, SubprocessRunner
from email_mlops.health.checks import HealthChecker, HealthStatus
from email_mlops.health.factory import build_checker
from email_mlops.model_registry.base import BaseModelRegistry
from email_mlops.model_registry.factory import create_registry

# Truncate captured command output kept in the result to keep it readable.
_MAX_OUTPUT_CHARS = 4000


class DeploymentStatus(StrEnum):
    """Overall outcome of a deployment run."""

    SUCCESS = "success"
    FAILED = "failed"
    PLANNED = "planned"  # dry-run only


@dataclass
class StepResult:
    """Outcome of a single deployment step."""

    name: str
    ok: bool
    detail: str = ""
    output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeploymentResult:
    """Structured result of a deployment run."""

    environment: str
    status: DeploymentStatus
    message: str = ""
    steps: list[StepResult] = field(default_factory=list)
    # Registered model name -> resolved version being deployed.
    model_versions: dict[str, int] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != DeploymentStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "environment": self.environment,
            "status": self.status.value,
            "message": self.message,
            "steps": [s.to_dict() for s in self.steps],
            "model_versions": self.model_versions,
            "command": self.command,
        }


def _compose_command(cfg: EnvironmentConfig, build: bool) -> list[str]:
    """Build the ``docker compose ... up -d`` command for an environment."""
    cmd = ["docker", "compose"]
    if cfg.compose.project_name:
        cmd += ["-p", cfg.compose.project_name]
    for compose_file in cfg.compose.files:
        cmd += ["-f", compose_file]
    cmd += ["up", "-d"]
    if build:
        cmd += ["--build"]
    return cmd


class Deployer:
    """Automated, repeatable deployment driver.

    Parameters
    ----------
    config_path:
        Path to ``configs/mlops.yaml`` (used to build the model registry).
    environments_dir:
        Directory holding per-environment configs.
    runner:
        :class:`CommandRunner` used to invoke Docker Compose. Defaults to a
        real :class:`SubprocessRunner`; tests inject a fake.
    registry:
        Optional pre-built model registry (tests inject a local one).
    checker_builder:
        Builds the readiness :class:`HealthChecker` for an environment.
        Overridable for tests.
    sleep:
        Sleep function used between health-gate polls (injectable for tests).
    """

    def __init__(
        self,
        config_path: str | Path = "configs/mlops.yaml",
        environments_dir: str | Path = "configs/environments",
        runner: CommandRunner | None = None,
        registry: BaseModelRegistry | None = None,
        checker_builder: Callable[[EnvironmentConfig], HealthChecker] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config_path = config_path
        self._environments_dir = environments_dir
        self._runner = runner or SubprocessRunner()
        self._registry = registry
        self._checker_builder = checker_builder or self._default_checker_builder
        self._sleep = sleep

    @staticmethod
    def _default_checker_builder(cfg: EnvironmentConfig) -> HealthChecker:
        return build_checker(f"{cfg.environment}", cfg.health.dependencies)

    # ── Public API ────────────────────────────────────────────────────────────

    def deploy(
        self,
        environment: str,
        build: bool | None = None,
        dry_run: bool = False,
        skip_health: bool = False,
    ) -> DeploymentResult:
        """Deploy ``environment``. Returns a structured result (never raises)."""
        result = DeploymentResult(environment=environment, status=DeploymentStatus.SUCCESS)

        # ── Step 1: load per-environment config ──
        try:
            cfg = load_environment_config(environment, self._environments_dir)
        except EnvironmentNotFoundError as exc:
            return self._fail(result, "load-config", str(exc))
        result.steps.append(StepResult("load-config", True, f"loaded environment '{environment}'"))

        # ── Step 2: resolve which model versions to ship (US-6.4) ──
        registry = self._registry or create_registry(self._config_path)
        missing: list[str] = []
        for name in cfg.models:
            target = registry.get_deployment_target(name)
            if target is None:
                missing.append(name)
            else:
                result.model_versions[name] = target.version
        if missing and cfg.require_models:
            return self._fail(
                result,
                "resolve-models",
                f"No deployable version for required model(s): {missing}.",
            )
        detail = f"resolved {result.model_versions}"
        if missing:
            detail += f"; skipped (no deployable version): {missing}"
        result.steps.append(StepResult("resolve-models", True, detail))

        # ── Build the compose command ──
        do_build = cfg.compose.build if build is None else build
        result.command = _compose_command(cfg, do_build)

        if dry_run:
            result.status = DeploymentStatus.PLANNED
            result.message = f"[dry-run] Would deploy '{environment}' with: " + " ".join(
                result.command
            )
            result.steps.append(StepResult("plan", True, "dry-run — no changes made"))
            return result

        # ── Step 3: bring services up ──
        logger.info(f"[Deployer] {environment}: {' '.join(result.command)}")
        outcome = self._runner.run(result.command, env=cfg.compose.env)
        if not outcome.ok:
            return self._fail(
                result,
                "compose-up",
                f"docker compose exited {outcome.returncode}: "
                f"{outcome.stderr.strip() or '(no stderr)'}",
                output=outcome.stderr[:_MAX_OUTPUT_CHARS],
            )
        result.steps.append(
            StepResult(
                "compose-up",
                True,
                "services started",
                output=outcome.stdout[:_MAX_OUTPUT_CHARS],
            )
        )

        # ── Step 4: health gate (US-6.6) ──
        if not skip_health and cfg.health.dependencies:
            ok, detail = self._wait_for_health(cfg)
            result.steps.append(StepResult("health-gate", ok, detail))
            if not ok:
                return self._fail(result, "health-gate", detail, append_step=False)
        else:
            result.steps.append(StepResult("health-gate", True, "skipped (no gate configured)"))

        result.status = DeploymentStatus.SUCCESS
        result.message = f"Deployment to '{environment}' succeeded."
        logger.info(f"[Deployer] {result.message}")
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _wait_for_health(self, cfg: EnvironmentConfig) -> tuple[bool, str]:
        """Poll readiness until the system is serviceable or the timeout hits."""
        checker = self._checker_builder(cfg)
        interval = max(1, cfg.health.interval_seconds)
        attempts = max(1, cfg.health.timeout_seconds // interval)

        last_report = None
        for attempt in range(1, attempts + 1):
            report = checker.check_readiness()
            last_report = report
            # DEGRADED (only non-critical deps down) still counts as serviceable.
            if report.status != HealthStatus.UNHEALTHY:
                return True, (f"ready after {attempt} attempt(s): {report.status.value}")
            if attempt < attempts:
                self._sleep(interval)

        failing = (
            [c.name for c in last_report.checks if c.status != HealthStatus.HEALTHY]
            if last_report
            else []
        )
        return False, (
            f"not ready within {cfg.health.timeout_seconds}s; "
            f"failing critical dependencies: {failing}"
        )

    @staticmethod
    def _fail(
        result: DeploymentResult,
        step: str,
        message: str,
        output: str = "",
        append_step: bool = True,
    ) -> DeploymentResult:
        if append_step:
            result.steps.append(StepResult(step, False, message, output=output))
        result.status = DeploymentStatus.FAILED
        result.message = f"Deployment failed at '{step}': {message}"
        logger.error(f"[Deployer] {result.message}")
        return result
