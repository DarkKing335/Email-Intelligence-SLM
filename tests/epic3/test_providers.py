"""Tests for built-in inference providers."""

from __future__ import annotations

import httpx
import pytest

from email_inference.api import create_app
from email_inference.providers.base import ProviderError
from email_inference.providers.http import HttpModelProvider
from email_inference.providers.mock import MockModelProvider
from email_intelligence.schemas import AnalysisRequest, EmailContent


@pytest.mark.asyncio
async def test_mock_provider_produces_structured_billing_result() -> None:
    provider = MockModelProvider()
    result = await provider.analyze(
        AnalysisRequest(
            email=EmailContent(
                email_id="email-1",
                subject="Urgent invoice INV-42",
                body_text="Please pay $1,250 by August 5. Let me know when complete.",
            )
        )
    )

    assert result.classification.value == "billing"
    assert result.priority.value == "high"
    assert result.response_required is True
    assert result.entities.monetary_amounts[0].amount == 1250
    assert result.entities.dates[0].text == "August 5"
    assert result.entities.references[0].text == "INV-42"
    assert result.recommended_actions[0].action.value == "reply"


@pytest.mark.asyncio
async def test_http_provider_validates_remote_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "classification": "fyi",
                "classification_confidence": 0.9,
                "priority": "medium",
                "priority_confidence": 0.8,
                "summary": "Project status update.",
                "entities": {},
                "response_required": False,
                "recommended_actions": [
                    {
                        "action": "flag_for_review",
                        "confidence": 0.7,
                        "reason": "Informational update.",
                    }
                ],
                "model": {
                    "provider": "fine-tuned",
                    "model_name": "email-slm",
                    "model_version": "3.0",
                    "schema_version": "1.0",
                },
            },
        )

    provider = HttpModelProvider(
        endpoint="https://model.test",
        api_token="secret",
        transport=httpx.MockTransport(handler),
    )
    result = await provider.analyze(
        AnalysisRequest(
            email=EmailContent(
                email_id="email-1",
                subject="Status",
                body_text="Project status update.",
            )
        )
    )
    await provider.close()

    assert result.model.model_name == "email-slm"
    assert result.entities.people == []


@pytest.mark.asyncio
async def test_http_provider_hides_invalid_upstream_details() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"classification": "not-approved"})
    )
    provider = HttpModelProvider(endpoint="https://model.test", transport=transport)
    with pytest.raises(ProviderError, match="model provider request failed"):
        await provider.analyze(
            AnalysisRequest(
                email=EmailContent(
                    email_id="email-1",
                    subject="Status",
                    body_text="Project status update.",
                )
            )
        )
    await provider.close()


@pytest.mark.asyncio
async def test_inference_api_exposes_standard_contract() -> None:
    app = create_app(provider=MockModelProvider())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/analyze",
                json={
                    "email": {
                        "email_id": "email-1",
                        "subject": "Security alert",
                        "body_text": "Use OTP 123456 to verify your account.",
                    }
                },
            )

    assert response.status_code == 200
    assert response.json()["classification"] == "security"
    assert response.json()["model"]["schema_version"] == "1.0"
