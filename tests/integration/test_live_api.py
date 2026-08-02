"""Live, over-HTTP integration test of the full stack.

Unlike the in-process API tests (``tests/epic3``), this boots the *real*
``uvicorn`` servers — the inference service (mock model) and the backend API —
on real ports and drives them through the dashboard's real ``ApiClient`` over
HTTP. It is the automated version of the manual end-to-end check.

Marked ``integration`` and skipped automatically if the servers cannot be
started in this environment (so it never flakes a constrained CI runner red).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import httpx
import pytest

from email_dashboard.api_client import ApiClient, ApiError

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_healthy(url: str, procs: list[subprocess.Popen], timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"a server exited early (code {proc.returncode})")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError(f"servers did not become healthy at {url} within {timeout}s")


@pytest.fixture(scope="module")
def live_api(tmp_path_factory) -> str:
    inf_port = _free_port()
    api_port = _free_port()
    db_path = tmp_path_factory.mktemp("live") / "live.db"

    import os

    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    env["INFERENCE_URL"] = f"http://127.0.0.1:{inf_port}"

    def _start(app: str, port: int) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    procs = [
        _start("email_inference.api:app", inf_port),
        _start("email_api.main:app", api_port),
    ]
    base_url = f"http://127.0.0.1:{api_port}"
    try:
        _wait_healthy(f"{base_url}/health", procs)
    except RuntimeError as exc:
        for proc in procs:
            proc.terminate()
        pytest.skip(f"live servers unavailable: {exc}")

    yield base_url

    for proc in procs:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_health_reports_ok(live_api: str) -> None:
    client = ApiClient(live_api)
    try:
        health = client.health()
        assert health["status"] == "ok"
        assert health["database"] == "ok"
        assert health["inference"] == "ok"
    finally:
        client.close()


def test_full_review_flow_over_http(live_api: str) -> None:
    client = ApiClient(live_api)
    try:
        # Create + auto-analyze. Phrased so a reply is required ("can you" /
        # "let me know"), so draft generation is permitted, while still
        # containing "invoice" for the search assertion below.
        created = client.create_email(
            sender="Accounts <billing@acme.com>",
            subject="Can you review invoice INV-9?",
            body_text="Could you review invoice INV-9 and let me know by Friday?",
        )
        email_id = created["id"]
        assert created["analysis"]["classification"] == "billing"
        assert created["reviewed"]["priority"] == "high"
        assert created["analysis"]["response_required"] is True

        # Search finds it (US-4.6)
        assert client.list_queue(search="invoice")["total"] >= 1
        assert client.list_queue(search="zzz-nomatch")["total"] == 0

        # Regenerate the draft with a different tone (reply-style hook)
        client.generate_draft(email_id, tone="friendly")

        # Human review: edit then approve (US-5.4)
        client.review(email_id, "reviewer-1", priority="medium")
        approved = client.decide(email_id, "reviewer-1", "approve")
        assert approved["review_status"] == "approved"
        event_types = [e["event_type"] for e in approved["audit_events"]]
        assert "edited" in event_types
        assert "approved" in event_types
    finally:
        client.close()


def test_missing_email_returns_404(live_api: str) -> None:
    client = ApiClient(live_api)
    try:
        with pytest.raises(ApiError) as exc:
            client.get_email("does-not-exist")
        assert exc.value.status_code == 404
    finally:
        client.close()
