"""Unit tests for the dashboard API client (US-4.5/4.6/5.4).

Uses ``httpx.MockTransport`` so no server is needed.
"""

from __future__ import annotations

import httpx
import pytest

from email_dashboard.api_client import ApiClient, ApiError


def _client(handler) -> ApiClient:
    transport = httpx.MockTransport(handler)
    return ApiClient(client=httpx.Client(base_url="http://test", transport=transport))


class TestListQueue:
    def test_builds_params_and_parses(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [], "total": 0, "limit": 50, "offset": 0})

        client = _client(handler)
        result = client.list_queue(search="invoice", priority="high", review_status="pending")

        assert seen["path"] == "/v1/review-queue"
        assert seen["params"]["search"] == "invoice"
        assert seen["params"]["priority"] == "high"
        assert seen["params"]["review_status"] == "pending"
        assert seen["params"]["limit"] == "50"
        assert result["total"] == 0

    def test_omits_empty_filters(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [], "total": 0, "limit": 50, "offset": 0})

        _client(handler).list_queue()
        assert "search" not in seen["params"]
        assert "priority" not in seen["params"]


class TestActions:
    def test_get_email(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/emails/abc"
            return httpx.Response(200, json={"id": "abc"})

        assert _client(handler).get_email("abc")["id"] == "abc"

    def test_create_email_posts_payload(self) -> None:
        import json as _json

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = _json.loads(request.content)
            return httpx.Response(201, json={"id": "new"})

        _client(handler).create_email(subject="Hi", body_text="Body", sender="a@b.com")
        assert seen["body"]["subject"] == "Hi"
        assert seen["body"]["sender"] == "a@b.com"

    def test_generate_draft_sends_tone(self) -> None:
        import json as _json

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = _json.loads(request.content)
            seen["path"] = request.url.path
            return httpx.Response(200, json={"id": "e1"})

        _client(handler).generate_draft("e1", tone="friendly")
        assert seen["path"] == "/v1/emails/e1/draft"
        assert seen["body"]["tone"] == "friendly"

    def test_review_sends_only_changed_fields(self) -> None:
        import json as _json

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = _json.loads(request.content)
            assert request.method == "PATCH"
            return httpx.Response(200, json={"id": "e1"})

        _client(handler).review("e1", "user-7", priority="medium")
        assert seen["body"] == {"reviewer_id": "user-7", "priority": "medium"}

    def test_decide_approve(self) -> None:
        import json as _json

        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = _json.loads(request.content)
            return httpx.Response(200, json={"review_status": "approved"})

        out = _client(handler).decide("e1", "user-7", "approve")
        assert seen["body"] == {"reviewer_id": "user-7", "decision": "approve"}
        assert out["review_status"] == "approved"


class TestErrors:
    def test_http_error_raises_api_error_with_detail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "email x was not found"})

        with pytest.raises(ApiError) as exc:
            _client(handler).get_email("x")
        assert exc.value.status_code == 404
        assert "not found" in str(exc.value.detail)

    def test_connection_error_raises_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(ApiError) as exc:
            _client(handler).health()
        assert exc.value.status_code == 0
