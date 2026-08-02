"""Unit tests for metrics & alerting (US-5.2)."""

from __future__ import annotations

from email_mlops.health import HealthChecker, Probe
from email_observability.metrics import (
    AlertRule,
    Comparison,
    MetricsCollector,
    alert_rules_from_config,
    evaluate_alerts,
)


class TestMetricsCollector:
    def test_counter_sums(self) -> None:
        m = MetricsCollector()
        m.incr("errors_total")
        m.incr("errors_total", 4)
        assert m.snapshot()["errors_total"] == 5.0

    def test_gauge_last_value(self) -> None:
        m = MetricsCollector()
        m.gauge("queue_depth", 3)
        m.gauge("queue_depth", 7)
        assert m.snapshot()["queue_depth"] == 7.0

    def test_timer_and_observe_derived_metrics(self) -> None:
        m = MetricsCollector()
        for v in (10, 20, 30, 40, 100):
            m.observe("inference_latency_ms", v)
        snap = m.snapshot()
        assert snap["inference_latency_ms.count"] == 5.0
        assert snap["inference_latency_ms.max"] == 100.0
        assert snap["inference_latency_ms.p95"] >= 40.0

    def test_timer_context(self) -> None:
        m = MetricsCollector()
        with m.timer("op_ms"):
            pass
        assert m.snapshot()["op_ms.count"] == 1.0

    def test_record_health_folds_status(self) -> None:
        m = MetricsCollector()
        # non-critical probe down → DEGRADED
        report = (
            HealthChecker("inference")
            .add_probe(Probe("cache", lambda: (False, "down"), critical=False))
            .check_readiness()
        )
        m.record_health(report)
        snap = m.snapshot()
        assert snap["service_health"] == 1.0  # degraded
        assert snap["dependency_up"] == 0.0

    def test_history_records_samples(self) -> None:
        m = MetricsCollector(history_size=2)
        m.incr("x")
        m.record_sample()
        m.incr("x")
        m.record_sample()
        m.record_sample()  # exceeds maxlen -> oldest dropped
        assert len(m.history()) == 2

    def test_prometheus_format_has_labels(self) -> None:
        m = MetricsCollector()
        m.incr("requests_total", 2, service="api")
        text = m.to_prometheus()
        assert 'requests_total{service="api"} 2' in text

    def test_reset(self) -> None:
        m = MetricsCollector()
        m.incr("x")
        m.reset()
        assert m.snapshot() == {}


class TestAlerts:
    def test_rule_fires_when_over_threshold(self) -> None:
        rule = AlertRule("high", "errors_total", Comparison.GT, 10)
        alert = rule.evaluate({"errors_total": 15})
        assert alert is not None
        assert alert.metric == "errors_total"
        assert alert.value == 15

    def test_rule_silent_when_under(self) -> None:
        rule = AlertRule("high", "errors_total", Comparison.GT, 10)
        assert rule.evaluate({"errors_total": 5}) is None

    def test_missing_metric_does_not_fire(self) -> None:
        rule = AlertRule("high", "errors_total", Comparison.GT, 10)
        assert rule.evaluate({}) is None

    def test_evaluate_alerts_collects_fired(self) -> None:
        rules = [
            AlertRule("a", "errors_total", Comparison.GT, 10),
            AlertRule("b", "queue", Comparison.LT, 100),
        ]
        fired = evaluate_alerts(rules, {"errors_total": 20, "queue": 5})
        assert {a.name for a in fired} == {"a", "b"}

    def test_alert_rules_from_config(self) -> None:
        rules = alert_rules_from_config(
            [
                {
                    "name": "svc_down",
                    "metric": "service_health",
                    "comparison": "<",
                    "threshold": 1,
                    "severity": "critical",
                }
            ]
        )
        assert rules[0].name == "svc_down"
        assert rules[0].comparison == Comparison.LT
        alert = rules[0].evaluate({"service_health": 0.0})
        assert alert is not None and alert.severity == "critical"
