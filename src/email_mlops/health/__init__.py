"""Health Check (US-6.6).

Reusable liveness/readiness checks and dependency probes for every service
in the platform. Import :class:`HealthChecker` and compose probes, or use the
``email-mlops health`` CLI for manual and CI checks.
"""

from email_mlops.health.checks import (
    CheckResult,
    HealthChecker,
    HealthReport,
    HealthStatus,
    Probe,
)
from email_mlops.health.factory import build_checker
from email_mlops.health.probes import (
    http_probe,
    mlflow_probe,
    postgres_probe,
    tcp_probe,
)

__all__ = [
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "CheckResult",
    "Probe",
    "build_checker",
    "tcp_probe",
    "http_probe",
    "mlflow_probe",
    "postgres_probe",
]
