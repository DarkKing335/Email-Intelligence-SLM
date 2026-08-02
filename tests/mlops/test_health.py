"""Unit tests for the Health Check module (US-6.6)."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from email_mlops.health import (
    HealthChecker,
    HealthStatus,
    Probe,
    http_probe,
    tcp_probe,
)


def _always(ok: bool, detail: str = ""):
    return lambda: (ok, detail)


# ── HealthChecker aggregation ─────────────────────────────────────────────────


class TestHealthChecker:
    def test_liveness_always_healthy(self) -> None:
        report = HealthChecker(service="svc").check_liveness()
        assert report.status == HealthStatus.HEALTHY
        assert report.ok is True
        assert report.http_status == 200
        assert report.checks[0].name == "process"

    def test_readiness_all_pass(self) -> None:
        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("a", _always(True), critical=True))
        checker.add_probe(Probe("b", _always(True), critical=False))
        report = checker.check_readiness()
        assert report.status == HealthStatus.HEALTHY
        assert report.http_status == 200

    def test_critical_failure_is_unhealthy(self) -> None:
        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("db", _always(False, "down"), critical=True))
        checker.add_probe(Probe("cache", _always(True), critical=False))
        report = checker.check_readiness()
        assert report.status == HealthStatus.UNHEALTHY
        assert report.ok is False
        assert report.http_status == 503

    def test_noncritical_failure_is_degraded(self) -> None:
        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("db", _always(True), critical=True))
        checker.add_probe(Probe("cache", _always(False, "slow"), critical=False))
        report = checker.check_readiness()
        assert report.status == HealthStatus.DEGRADED
        # Degraded still counts as serviceable
        assert report.ok is True

    def test_probe_exception_treated_as_failure(self) -> None:
        def _boom() -> tuple[bool, str]:
            raise RuntimeError("kaboom")

        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("x", _boom, critical=True))
        report = checker.check_readiness()
        assert report.status == HealthStatus.UNHEALTHY
        assert "kaboom" in report.checks[0].detail

    def test_latency_recorded(self) -> None:
        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("a", _always(True), critical=True))
        report = checker.check_readiness()
        assert report.checks[0].latency_ms is not None
        assert report.checks[0].latency_ms >= 0

    def test_report_to_dict(self) -> None:
        checker = HealthChecker(service="svc")
        checker.add_probe(Probe("a", _always(True), critical=True))
        d = checker.check_readiness().to_dict()
        assert d["status"] == "healthy"
        assert d["service"] == "svc"
        assert d["checks"][0]["name"] == "a"
        assert "checked_at" in d


# ── TCP probe ─────────────────────────────────────────────────────────────────


class TestTcpProbe:
    def test_open_port_is_healthy(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            probe = tcp_probe("svc", "127.0.0.1", port)
            ok, _ = probe.check()
            assert ok is True
        finally:
            server.close()

    def test_closed_port_is_unhealthy(self) -> None:
        # Bind then close to obtain a port that is (almost certainly) free.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        probe = tcp_probe("svc", "127.0.0.1", port, timeout=0.5)
        ok, detail = probe.check()
        assert ok is False
        assert "unreachable" in detail


# ── HTTP probe ────────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        code = 200 if self.path == "/health" else 500
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence test server logging
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


class TestHttpProbe:
    def test_healthy_endpoint(self, http_server: str) -> None:
        probe = http_probe("svc", f"{http_server}/health")
        ok, detail = probe.check()
        assert ok is True
        assert "200" in detail

    def test_unexpected_status_fails(self, http_server: str) -> None:
        probe = http_probe("svc", f"{http_server}/missing")
        ok, _ = probe.check()
        assert ok is False

    def test_connection_refused_fails(self) -> None:
        probe = http_probe("svc", "http://127.0.0.1:1/health", timeout=0.5)
        ok, _ = probe.check()
        assert ok is False
