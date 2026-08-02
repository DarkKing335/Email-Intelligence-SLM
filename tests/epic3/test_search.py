"""Tests for review-queue keyword search (US-4.6)."""

from __future__ import annotations

import httpx
import pytest

from email_api.main import create_app
from email_api.settings import ApiSettings
from email_inference.providers.mock import MockModelProvider


class _Gateway:
    """In-process inference gateway backed by the deterministic mock provider."""

    def __init__(self) -> None:
        self._provider = MockModelProvider()

    async def analyze(self, request):
        return await self._provider.analyze(request)

    async def generate_draft(self, request):
        return await self._provider.generate_draft(request)

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


async def _client(tmp_dir):
    settings = ApiSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_dir / 'search.db'}",
        auto_create_schema=True,
    )
    app = create_app(settings=settings, inference=_Gateway())
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return client, lifespan


async def _seed(client) -> None:
    await client.post(
        "/v1/emails",
        json={
            "sender": "Accounts <billing@acme.com>",
            "subject": "Urgent invoice INV-42",
            "body_text": "Please pay $1,250 by August 5 and let me know when complete.",
        },
    )
    await client.post(
        "/v1/emails",
        json={
            "sender": "Jira <no-reply@atlassian.com>",
            "subject": "Weekly status digest",
            "body_text": "The sprint report is attached for your information.",
        },
    )


@pytest.mark.asyncio
async def test_search_matches_subject(tmp_dir) -> None:
    client, lifespan = await _client(tmp_dir)
    try:
        await _seed(client)
        res = await client.get("/v1/review-queue", params={"search": "invoice"})
        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert "invoice" in body["items"][0]["subject"].lower()
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_search_matches_sender_and_body(tmp_dir) -> None:
    client, lifespan = await _client(tmp_dir)
    try:
        await _seed(client)
        by_sender = await client.get("/v1/review-queue", params={"search": "atlassian"})
        assert by_sender.json()["total"] == 1
        by_body = await client.get("/v1/review-queue", params={"search": "sprint"})
        assert by_body.json()["total"] == 1
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_search_is_case_insensitive_and_combines_with_filters(tmp_dir) -> None:
    client, lifespan = await _client(tmp_dir)
    try:
        await _seed(client)
        res = await client.get(
            "/v1/review-queue",
            params={"search": "INVOICE", "priority": "high"},
        )
        assert res.json()["total"] == 1
        # search + a filter that excludes the match -> empty
        none = await client.get(
            "/v1/review-queue",
            params={"search": "invoice", "priority": "low"},
        )
        assert none.json()["total"] == 0
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_search_no_match_returns_empty(tmp_dir) -> None:
    client, lifespan = await _client(tmp_dir)
    try:
        await _seed(client)
        res = await client.get("/v1/review-queue", params={"search": "zzz-nomatch"})
        assert res.status_code == 200
        assert res.json()["total"] == 0
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)
