"""In-process metrics collection and alerting (US-5.2).

A dependency-free metrics registry for the platform's services. It records
counters, gauges, and timings; produces point-in-time **snapshots** and a
bounded **history** (trackable over time); folds in service **health** state
(US-6.6) so degradation is visible; and evaluates configurable **alert rules**.

Deliberately *not* a Prometheus/OpenTelemetry client — those belong in the
long-running services. A plain-text Prometheus exposition is provided via
:meth:`MetricsCollector.to_prometheus` so standard scrapers can consume it
without adding a dependency here.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock

# Map health status → numeric gauge so "degraded" is visible on a dashboard.
from email_mlops.health.checks import HealthReport, HealthStatus

_HEALTH_VALUE = {
    HealthStatus.HEALTHY: 2.0,
    HealthStatus.DEGRADED: 1.0,
    HealthStatus.UNHEALTHY: 0.0,
}

_Key = tuple[str, tuple[tuple[str, str], ...]]


def _key(name: str, labels: dict[str, str]) -> _Key:
    return name, tuple(sorted(labels.items()))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


class Comparison(StrEnum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="


@dataclass
class AlertRule:
    """A threshold rule evaluated against a metrics snapshot."""

    name: str
    metric: str
    comparison: Comparison
    threshold: float
    severity: str = "warning"
    message: str = ""

    def evaluate(self, snapshot: dict[str, float]) -> Alert | None:
        if self.metric not in snapshot:
            return None
        value = snapshot[self.metric]
        cmp = self.comparison
        fired = (
            (cmp == Comparison.GT and value > self.threshold)
            or (cmp == Comparison.GTE and value >= self.threshold)
            or (cmp == Comparison.LT and value < self.threshold)
            or (cmp == Comparison.LTE and value <= self.threshold)
            or (cmp == Comparison.EQ and value == self.threshold)
        )
        if not fired:
            return None
        return Alert(
            name=self.name,
            severity=self.severity,
            metric=self.metric,
            value=value,
            threshold=self.threshold,
            message=self.message
            or f"{self.metric}={value} {self.comparison.value} {self.threshold}",
        )


@dataclass
class Alert:
    """A fired alert."""

    name: str
    severity: str
    metric: str
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class _Sample:
    timestamp: str
    metrics: dict[str, float]


@dataclass
class MetricsCollector:
    """Thread-safe in-process metrics registry."""

    history_size: int = 288  # e.g. 24h at 5-min sampling
    _counters: dict[_Key, float] = field(default_factory=dict)
    _gauges: dict[_Key, float] = field(default_factory=dict)
    _timers: dict[_Key, list[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    _history: deque[_Sample] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_size)

    # ── Recording ─────────────────────────────────────────────────────────────

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] = self._counters.get(_key(name, labels), 0.0) + value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value

    def observe(self, name: str, value_ms: float, **labels: str) -> None:
        """Record a timing/observation (milliseconds by convention)."""
        with self._lock:
            self._timers.setdefault(_key(name, labels), []).append(value_ms)

    @contextmanager
    def timer(self, name: str, **labels: str) -> Iterator[None]:
        """Context manager that records elapsed wall-clock time in ms."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0, **labels)

    def record_health(self, report: HealthReport) -> None:
        """Fold a health report into gauges so degradation is measurable."""
        self.gauge("service_health", _HEALTH_VALUE[report.status], service=report.service)
        for check in report.checks:
            self.gauge(
                "dependency_up",
                1.0 if check.status == HealthStatus.HEALTHY else 0.0,
                service=report.service,
                dependency=check.name,
            )

    # ── Reading ───────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, float]:
        """Flat name→value view for alerting and quick inspection.

        Counters are summed per name; gauges take their last value per name;
        timers expand to ``name.count``/``.avg``/``.p95``/``.max``.
        """
        with self._lock:
            flat: dict[str, float] = {}
            for (name, _labels), value in self._counters.items():
                flat[name] = flat.get(name, 0.0) + value
            for (name, _labels), value in self._gauges.items():
                flat[name] = value
            for (name, _labels), values in self._timers.items():
                if not values:
                    continue
                flat[f"{name}.count"] = float(len(values))
                flat[f"{name}.avg"] = sum(values) / len(values)
                flat[f"{name}.p95"] = _percentile(values, 95)
                flat[f"{name}.max"] = max(values)
            return flat

    def record_sample(self) -> _Sample:
        """Append the current snapshot to the bounded history (over-time view)."""
        sample = _Sample(datetime.now(UTC).isoformat(), self.snapshot())
        self._history.append(sample)
        return sample

    def history(self) -> list[dict]:
        return [{"timestamp": s.timestamp, "metrics": s.metrics} for s in self._history]

    def to_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format (labelled)."""
        lines: list[str] = []

        def _fmt_labels(labels: tuple[tuple[str, str], ...]) -> str:
            if not labels:
                return ""
            inner = ",".join(f'{k}="{v}"' for k, v in labels)
            return "{" + inner + "}"

        with self._lock:
            for (name, labels), value in self._counters.items():
                lines.append(f"{name}{_fmt_labels(labels)} {value}")
            for (name, labels), value in self._gauges.items():
                lines.append(f"{name}{_fmt_labels(labels)} {value}")
            for (name, labels), values in self._timers.items():
                if not values:
                    continue
                base = f"{name}{_fmt_labels(labels)}"
                lines.append(f"{base}_count {len(values)}")
                lines.append(f"{base}_avg {sum(values) / len(values)}")
                lines.append(f"{base}_p95 {_percentile(values, 95)}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            self._history.clear()


def evaluate_alerts(rules: list[AlertRule], snapshot: dict[str, float]) -> list[Alert]:
    """Return all alerts that fire for the given snapshot."""
    fired: list[Alert] = []
    for rule in rules:
        alert = rule.evaluate(snapshot)
        if alert is not None:
            fired.append(alert)
    return fired


def alert_rules_from_config(specs: list[dict]) -> list[AlertRule]:
    """Build :class:`AlertRule` objects from config dicts."""
    rules: list[AlertRule] = []
    for spec in specs or []:
        rules.append(
            AlertRule(
                name=spec["name"],
                metric=spec["metric"],
                comparison=Comparison(spec.get("comparison", ">")),
                threshold=float(spec["threshold"]),
                severity=spec.get("severity", "warning"),
                message=spec.get("message", ""),
            )
        )
    return rules
