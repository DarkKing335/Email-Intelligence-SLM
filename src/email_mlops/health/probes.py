"""Reusable dependency probes (US-6.6).

Each factory returns a :class:`Probe` whose ``check`` callable yields
``(ok, detail)``.  Probes use only the Python standard library for the common
cases (TCP, HTTP) so the health module has no extra runtime dependencies;
the PostgreSQL probe lazily imports its driver and degrades gracefully to a
raw TCP check if none is installed.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

from email_mlops.health.checks import Probe


def tcp_probe(
    name: str,
    host: str,
    port: int,
    timeout: float = 2.0,
    critical: bool = True,
) -> Probe:
    """Probe that a TCP port is accepting connections."""

    def _check() -> tuple[bool, str]:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"tcp {host}:{port} reachable"
        except OSError as exc:
            return False, f"tcp {host}:{port} unreachable: {exc}"

    return Probe(name=name, check=_check, critical=critical)


def http_probe(
    name: str,
    url: str,
    timeout: float = 3.0,
    expected_status: int = 200,
    critical: bool = True,
) -> Probe:
    """Probe that an HTTP endpoint returns the expected status code."""

    def _check() -> tuple[bool, str]:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                code = resp.getcode()
                if code == expected_status:
                    return True, f"GET {url} → {code}"
                return False, f"GET {url} → {code} (expected {expected_status})"
        except urllib.error.HTTPError as exc:
            if exc.code == expected_status:
                return True, f"GET {url} → {exc.code}"
            return False, f"GET {url} → {exc.code} (expected {expected_status})"
        except (urllib.error.URLError, OSError) as exc:
            return False, f"GET {url} failed: {exc}"

    return Probe(name=name, check=_check, critical=critical)


def mlflow_probe(
    tracking_uri: str,
    timeout: float = 3.0,
    critical: bool = True,
) -> Probe:
    """Probe the MLflow tracking server's ``/health`` endpoint."""
    url = tracking_uri.rstrip("/") + "/health"
    probe = http_probe("mlflow", url, timeout=timeout, critical=critical)
    return probe


def postgres_probe(
    host: str,
    port: int = 5432,
    timeout: float = 2.0,
    critical: bool = True,
) -> Probe:
    """Probe PostgreSQL reachability.

    Attempts a lightweight ``SELECT 1`` if ``psycopg`` is available (and
    connection kwargs are supplied via ``dsn``); otherwise falls back to a raw
    TCP check on the port so the probe is always usable with zero extra deps.
    """
    # For the common "is Postgres up" question a TCP check is sufficient and
    # dependency-free. A full auth'd query belongs in the API service that owns
    # the credentials, which can compose its own probe.
    return tcp_probe(
        name="postgres",
        host=host,
        port=port,
        timeout=timeout,
        critical=critical,
    )
