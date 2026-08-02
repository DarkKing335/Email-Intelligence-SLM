"""Command execution abstraction for deployments (US-6.5).

Deployments shell out to ``docker compose``. Hiding that behind a small
:class:`CommandRunner` protocol keeps the :class:`~email_mlops.deployment.deployer.Deployer`
testable — unit tests inject a fake runner instead of invoking Docker.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass
class CommandOutcome:
    """Result of running an external command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    """Runs an external command and captures its output."""

    def run(
        self,
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome: ...


class SubprocessRunner:
    """Real :class:`CommandRunner` backed by :func:`subprocess.run`."""

    def run(
        self,
        cmd: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandOutcome:
        merged_env = {**os.environ, **(env or {})}
        try:
            proc = subprocess.run(  # noqa: S603 — cmd is built from trusted config
                cmd,
                cwd=cwd,
                env=merged_env,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            # e.g. docker not installed — surface it as a failed outcome
            return CommandOutcome(returncode=127, stderr=str(exc))
        return CommandOutcome(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
