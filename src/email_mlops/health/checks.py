"""Health-check primitives (US-6.6).

Provides reusable *liveness* and *readiness* checks that any service (the
FastAPI inference/API servers, the Streamlit dashboard, CLI tooling, or CI)
can import and expose behind a ``/health`` endpoint or run standalone.

Distinction:
    * **Liveness**  — is *this* process up and responsive?  Never depends on
      external services, so a liveness probe won't cascade-fail a healthy
      process just because a dependency is down.
    * **Readiness** — is the service ready to serve traffic, i.e. are its
      dependencies reachable?  Aggregates a set of :class:`Probe` results.

Aggregation rule (reflects "degraded state when dependencies unavailable"):
    * any **critical** probe failing        → ``UNHEALTHY``
    * only **non-critical** probes failing   → ``DEGRADED``
    * everything passing                      → ``HEALTHY``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from loguru import logger


class HealthStatus(StrEnum):
    """Overall or per-check health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# A probe body returns (ok, detail).  Raising is also treated as a failure.
ProbeFn = Callable[[], tuple[bool, str]]


@dataclass
class Probe:
    """A named dependency check.

    Parameters
    ----------
    name:
        Identifier for the dependency (e.g. ``"postgres"``, ``"mlflow"``).
    check:
        Zero-argument callable returning ``(ok, detail)``.  Factory helpers in
        :mod:`email_mlops.health.probes` build these.
    critical:
        If ``True``, this probe failing makes the whole service ``UNHEALTHY``;
        otherwise it only degrades it.
    """

    name: str
    check: ProbeFn
    critical: bool = True


@dataclass
class CheckResult:
    """Outcome of running a single probe."""

    name: str
    status: HealthStatus
    critical: bool
    detail: str = ""
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "critical": self.critical,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


@dataclass
class HealthReport:
    """Aggregated health report for a service."""

    status: HealthStatus
    service: str
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True unless the service is fully ``UNHEALTHY`` (degraded still serves)."""
        return self.status != HealthStatus.UNHEALTHY

    @property
    def http_status(self) -> int:
        """Suggested HTTP status code for a ``/health`` endpoint."""
        return 200 if self.status == HealthStatus.HEALTHY else 503

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "service": self.service,
            "checked_at": self.checked_at,
            "checks": [c.to_dict() for c in self.checks],
        }


class HealthChecker:
    """Collects probes and produces liveness / readiness reports.

    Parameters
    ----------
    service:
        Name of the service this checker represents (appears in reports).
    """

    def __init__(self, service: str = "service") -> None:
        self._service = service
        self._probes: list[Probe] = []

    def add_probe(self, probe: Probe) -> HealthChecker:
        """Register a dependency probe. Returns ``self`` for chaining."""
        self._probes.append(probe)
        return self

    def add_probes(self, probes: list[Probe]) -> HealthChecker:
        self._probes.extend(probes)
        return self

    # ── Liveness ──────────────────────────────────────────────────────────────

    def check_liveness(self) -> HealthReport:
        """Liveness never touches dependencies — reaching here means we're alive."""
        return HealthReport(
            status=HealthStatus.HEALTHY,
            service=self._service,
            checks=[
                CheckResult(
                    name="process",
                    status=HealthStatus.HEALTHY,
                    critical=True,
                    detail="process is responsive",
                )
            ],
        )

    # ── Readiness ─────────────────────────────────────────────────────────────

    def check_readiness(self) -> HealthReport:
        """Run all registered probes and aggregate into an overall status."""
        results: list[CheckResult] = [self._run_probe(p) for p in self._probes]

        status = HealthStatus.HEALTHY
        for result in results:
            if result.status == HealthStatus.HEALTHY:
                continue
            if result.critical:
                status = HealthStatus.UNHEALTHY
                break
            status = HealthStatus.DEGRADED

        report = HealthReport(status=status, service=self._service, checks=results)
        logger.info(
            f"[HealthChecker] Readiness for '{self._service}': {status.value} "
            f"({len(results)} probe(s))"
        )
        return report

    def _run_probe(self, probe: Probe) -> CheckResult:
        start = time.perf_counter()
        try:
            ok, detail = probe.check()
        except Exception as exc:  # noqa: BLE001 — any failure means "not ready"
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return CheckResult(
            name=probe.name,
            status=HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY,
            critical=probe.critical,
            detail=detail,
            latency_ms=latency_ms,
        )
