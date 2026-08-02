"""Build a :class:`HealthChecker` from a declarative dependency spec.

Shared by the ``email-mlops health`` CLI and the deployment health-gate
(US-6.5) so probe wiring lives in exactly one place.
"""

from __future__ import annotations

from typing import Any

from email_mlops.health.checks import HealthChecker
from email_mlops.health.probes import (
    http_probe,
    mlflow_probe,
    postgres_probe,
    tcp_probe,
)


def build_checker(
    service: str,
    dependencies: dict[str, dict[str, Any]] | None,
    only: set[str] | None = None,
) -> HealthChecker:
    """Construct a :class:`HealthChecker` from a ``{name: spec}`` mapping.

    Each ``spec`` has a ``type`` (``http`` | ``tcp`` | ``mlflow`` | ``postgres``)
    and a ``critical`` flag, plus type-specific fields. ``only`` optionally
    restricts wiring to a subset of dependency names.
    """
    checker = HealthChecker(service=service)
    for name, spec in (dependencies or {}).items():
        if only is not None and name not in only:
            continue
        kind = spec.get("type", "tcp")
        critical = bool(spec.get("critical", True))
        if kind == "http":
            checker.add_probe(http_probe(name, spec["url"], critical=critical))
        elif kind == "mlflow":
            checker.add_probe(mlflow_probe(spec["tracking_uri"], critical=critical))
        elif kind == "postgres":
            checker.add_probe(
                postgres_probe(spec["host"], spec.get("port", 5432), critical=critical)
            )
        else:  # tcp
            checker.add_probe(tcp_probe(name, spec["host"], spec["port"], critical=critical))
    return checker
