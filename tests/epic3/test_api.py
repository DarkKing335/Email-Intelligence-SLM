"""End-to-end tests for the API-only Epic 3 workflow."""

from __future__ import annotations

import httpx
import pytest

from email_api.inference_client import InferenceError
from email_api.main import create_app
from email_api.settings import ApiSettings
from email_inference.providers.mock import MockModelProvider


class FakeInferenceGateway:
    def __init__(self, fail_analysis: bool = False, fail_draft: bool = False) -> None:
        self.provider = MockModelProvider()
        self.fail_analysis = fail_analysis
        self.fail_draft = fail_draft
        self.draft_calls = 0

    async def analyze(self, request):
        if self.fail_analysis:
            raise InferenceError("inference request failed: timeout")
        return await self.provider.analyze(request)

    async def generate_draft(self, request):
        self.draft_calls += 1
        if self.fail_draft:
            raise InferenceError("inference request failed: timeout")
        return await self.provider.generate_draft(request)

    async def health(self) -> bool:
        return not self.fail_analysis

    async def close(self) -> None:
        return None


async def api_client(tmp_dir, gateway: FakeInferenceGateway):
    settings = ApiSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_dir / 'epic3.db'}",
        auto_create_schema=True,
    )
    app = create_app(settings=settings, inference=gateway)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
    return app, client, lifespan


@pytest.mark.asyncio
async def test_complete_review_and_approval_flow(tmp_dir) -> None:
    gateway = FakeInferenceGateway()
    _, client, lifespan = await api_client(tmp_dir, gateway)
    try:
        response = await client.post(
            "/v1/emails",
            json={
                "external_id": "mailbox-42",
                "sender": "Accounts <billing@example.com>",
                "subject": "Urgent invoice INV-42",
                "body_text": "Please pay $1,250 by August 5 and let me know when complete.",
            },
        )
        assert response.status_code == 201
        created = response.json()
        email_id = created["id"]
        assert created["analysis"]["classification"] == "billing"
        assert created["analysis"]["classification_confidence"] == 0.86
        assert created["reviewed"]["priority"] == "high"
        assert created["draft"]["source"] == "model"
        assert gateway.draft_calls == 1

        queue = await client.get("/v1/review-queue", params={"priority": "high"})
        assert queue.status_code == 200
        assert queue.json()["total"] == 1
        assert queue.json()["items"][0]["id"] == email_id

        edited = await client.patch(
            f"/v1/emails/{email_id}/review",
            json={
                "reviewer_id": "user-7",
                "priority": "medium",
                "draft_body": "Thanks. I will confirm payment shortly.",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["reviewed"]["priority"] == "medium"
        assert edited.json()["draft"]["source"] == "user"
        assert edited.json()["draft"]["version"] == 2

        approved = await client.post(
            f"/v1/emails/{email_id}/decision",
            json={"reviewer_id": "user-7", "decision": "approve"},
        )
        assert approved.status_code == 200
        result = approved.json()
        assert result["review_status"] == "approved"
        event_types = [event["event_type"] for event in result["audit_events"]]
        assert "edited" in event_types
        assert "approved" in event_types

        reanalyzed = await client.post(f"/v1/emails/{email_id}/reanalyze")
        assert reanalyzed.status_code == 200
        assert reanalyzed.json()["analysis_version"] == 2
        assert reanalyzed.json()["review_status"] == "pending"
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_non_response_email_does_not_generate_draft(tmp_dir) -> None:
    gateway = FakeInferenceGateway()
    _, client, lifespan = await api_client(tmp_dir, gateway)
    try:
        response = await client.post(
            "/v1/emails",
            json={
                "subject": "Weekly status",
                "body_text": "The project status report is attached for information.",
            },
        )
        assert response.status_code == 201
        assert response.json()["draft"] is None
        assert gateway.draft_calls == 0
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_analysis_failure_is_persisted_for_retry(tmp_dir) -> None:
    gateway = FakeInferenceGateway(fail_analysis=True)
    _, client, lifespan = await api_client(tmp_dir, gateway)
    try:
        response = await client.post(
            "/v1/emails",
            json={"subject": "Status", "body_text": "A meaningful project update."},
        )
        assert response.status_code == 502

        queue = await client.get(
            "/v1/review-queue",
            params={"processing_status": "analysis_failed"},
        )
        assert queue.status_code == 200
        assert queue.json()["total"] == 1
        email_id = queue.json()["items"][0]["id"]

        gateway.fail_analysis = False
        retried = await client.post(f"/v1/emails/{email_id}/reanalyze")
        assert retried.status_code == 200
        assert retried.json()["processing_status"] == "analyzed"
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_draft_failure_keeps_analysis(tmp_dir) -> None:
    gateway = FakeInferenceGateway(fail_draft=True)
    _, client, lifespan = await api_client(tmp_dir, gateway)
    try:
        response = await client.post(
            "/v1/emails",
            json={
                "subject": "Request",
                "body_text": "Can you review this and let me know today?",
            },
        )
        assert response.status_code == 201
        result = response.json()
        assert result["processing_status"] == "analyzed"
        assert result["analysis"] is not None
        assert result["draft"] is None
        assert result["draft_error"] is not None
    finally:
        await client.aclose()
        await lifespan.__aexit__(None, None, None)
