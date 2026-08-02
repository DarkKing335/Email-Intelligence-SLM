"""Unit tests for the Deployment module (US-6.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from email_mlops.deployment import (
    Deployer,
    DeploymentStatus,
    EnvironmentNotFoundError,
    list_environments,
    load_environment_config,
)
from email_mlops.deployment.runner import CommandOutcome
from email_mlops.health import HealthChecker, Probe
from email_mlops.model_registry import LocalModelRegistry, RegistryStage

# ── Test doubles ──────────────────────────────────────────────────────────────


class FakeRunner:
    """A CommandRunner that returns a canned outcome and records the call."""

    def __init__(self, outcome: CommandOutcome) -> None:
        self._outcome = outcome
        self.calls: list[list[str]] = []

    def run(self, cmd, cwd=None, env=None) -> CommandOutcome:
        self.calls.append(cmd)
        return self._outcome


def _healthy_checker(_cfg) -> HealthChecker:
    return HealthChecker(service="test").add_probe(Probe("ok", lambda: (True, "up"), critical=True))


def _unhealthy_checker(_cfg) -> HealthChecker:
    return HealthChecker(service="test").add_probe(
        Probe("db", lambda: (False, "down"), critical=True)
    )


def _write_env(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(body, encoding="utf-8")


# ── Config loading ────────────────────────────────────────────────────────────


class TestEnvironmentConfig:
    def test_load_and_defaults(self, tmp_dir: Path) -> None:
        envs = tmp_dir / "environments"
        _write_env(
            envs,
            "staging",
            "environment: staging\n"
            "compose:\n"
            "  project_name: email-slm-staging\n"
            "  build: true\n"
            "models: [reply-style-professional]\n"
            "require_models: true\n",
        )
        cfg = load_environment_config("staging", envs)
        assert cfg.environment == "staging"
        assert cfg.compose.project_name == "email-slm-staging"
        assert cfg.compose.files == ["docker-compose.yml"]  # default
        assert cfg.models == ["reply-style-professional"]
        assert cfg.require_models is True

    def test_missing_environment_raises_with_available(self, tmp_dir: Path) -> None:
        envs = tmp_dir / "environments"
        _write_env(envs, "dev", "environment: dev\n")
        with pytest.raises(EnvironmentNotFoundError, match="Available environments"):
            load_environment_config("prod", envs)

    def test_list_environments(self, tmp_dir: Path) -> None:
        envs = tmp_dir / "environments"
        _write_env(envs, "dev", "environment: dev\n")
        _write_env(envs, "staging", "environment: staging\n")
        assert list_environments(envs) == ["dev", "staging"]


# ── Deployer ──────────────────────────────────────────────────────────────────


class TestDeployer:
    def _env(self, tmp_dir: Path, body: str, name: str = "staging") -> Path:
        envs = tmp_dir / "environments"
        _write_env(envs, name, body)
        return envs

    def _registry_with_prod_model(self, tmp_dir: Path) -> LocalModelRegistry:
        reg = LocalModelRegistry(root_dir=tmp_dir / "registry")
        reg.register_model("reply-style-professional", adapter_path="p1")
        reg.transition_stage("reply-style-professional", 1, RegistryStage.PRODUCTION)
        return reg

    def test_dry_run_plans_command_without_running(self, tmp_dir: Path) -> None:
        envs = self._env(
            tmp_dir,
            "environment: staging\n"
            "compose:\n  project_name: p\n  build: true\n"
            "models: [reply-style-professional]\n",
        )
        runner = FakeRunner(CommandOutcome(0))
        deployer = Deployer(
            environments_dir=envs,
            runner=runner,
            registry=self._registry_with_prod_model(tmp_dir),
        )
        result = deployer.deploy("staging", dry_run=True)

        assert result.status == DeploymentStatus.PLANNED
        assert runner.calls == []  # nothing executed
        assert result.command == [
            "docker",
            "compose",
            "-p",
            "p",
            "-f",
            "docker-compose.yml",
            "up",
            "-d",
            "--build",
        ]
        assert result.model_versions == {"reply-style-professional": 1}

    def test_successful_deploy(self, tmp_dir: Path) -> None:
        envs = self._env(
            tmp_dir,
            "environment: staging\n"
            "models: [reply-style-professional]\n"
            "health:\n"
            "  timeout_seconds: 10\n"
            "  interval_seconds: 1\n"
            "  dependencies:\n"
            "    db: {type: tcp, host: localhost, port: 5432, critical: true}\n",
        )
        runner = FakeRunner(CommandOutcome(0, stdout="Started"))
        deployer = Deployer(
            environments_dir=envs,
            runner=runner,
            registry=self._registry_with_prod_model(tmp_dir),
            checker_builder=_healthy_checker,
            sleep=lambda _s: None,
        )
        result = deployer.deploy("staging")

        assert result.status == DeploymentStatus.SUCCESS
        assert result.ok is True
        assert len(runner.calls) == 1
        step_names = [s.name for s in result.steps]
        assert step_names == ["load-config", "resolve-models", "compose-up", "health-gate"]
        assert all(s.ok for s in result.steps)

    def test_compose_failure_is_surfaced(self, tmp_dir: Path) -> None:
        envs = self._env(tmp_dir, "environment: staging\n")
        runner = FakeRunner(CommandOutcome(1, stderr="Error: port 5432 already allocated"))
        deployer = Deployer(
            environments_dir=envs,
            runner=runner,
            registry=LocalModelRegistry(root_dir=tmp_dir / "registry"),
            checker_builder=_healthy_checker,
        )
        result = deployer.deploy("staging")

        assert result.status == DeploymentStatus.FAILED
        assert result.ok is False
        assert "port 5432 already allocated" in result.message
        failed = [s for s in result.steps if not s.ok]
        assert failed and failed[0].name == "compose-up"

    def test_health_gate_failure_times_out(self, tmp_dir: Path) -> None:
        envs = self._env(
            tmp_dir,
            "environment: staging\n"
            "health:\n"
            "  timeout_seconds: 3\n"
            "  interval_seconds: 1\n"
            "  dependencies:\n"
            "    db: {type: tcp, host: localhost, port: 5432, critical: true}\n",
        )
        slept: list[float] = []
        deployer = Deployer(
            environments_dir=envs,
            runner=FakeRunner(CommandOutcome(0)),
            registry=LocalModelRegistry(root_dir=tmp_dir / "registry"),
            checker_builder=_unhealthy_checker,
            sleep=lambda s: slept.append(s),
        )
        result = deployer.deploy("staging")

        assert result.status == DeploymentStatus.FAILED
        assert "not ready" in result.message
        assert "db" in result.message
        # Polled multiple times before giving up (3s / 1s interval).
        assert len(slept) >= 1

    def test_missing_env_fails_cleanly(self, tmp_dir: Path) -> None:
        envs = self._env(tmp_dir, "environment: staging\n")
        deployer = Deployer(
            environments_dir=envs,
            runner=FakeRunner(CommandOutcome(0)),
            registry=LocalModelRegistry(root_dir=tmp_dir / "registry"),
        )
        result = deployer.deploy("production")
        assert result.status == DeploymentStatus.FAILED
        assert "production" in result.message
        assert result.steps[0].name == "load-config" and not result.steps[0].ok

    def test_required_model_missing_fails(self, tmp_dir: Path) -> None:
        envs = self._env(
            tmp_dir,
            "environment: staging\nmodels: [reply-style-professional]\nrequire_models: true\n",
        )
        deployer = Deployer(
            environments_dir=envs,
            runner=FakeRunner(CommandOutcome(0)),
            registry=LocalModelRegistry(root_dir=tmp_dir / "registry"),  # empty
            checker_builder=_healthy_checker,
        )
        result = deployer.deploy("staging")
        assert result.status == DeploymentStatus.FAILED
        assert "required model" in result.message.lower()

    def test_no_build_override(self, tmp_dir: Path) -> None:
        envs = self._env(
            tmp_dir,
            "environment: staging\ncompose:\n  build: true\n",
        )
        runner = FakeRunner(CommandOutcome(0))
        deployer = Deployer(
            environments_dir=envs,
            runner=runner,
            registry=LocalModelRegistry(root_dir=tmp_dir / "registry"),
        )
        result = deployer.deploy("staging", build=False, skip_health=True)
        assert result.status == DeploymentStatus.SUCCESS
        assert "--build" not in runner.calls[0]
