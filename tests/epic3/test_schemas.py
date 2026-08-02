"""Tests for the provider-neutral model contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from email_data_engineering.domain.taxonomy import CanonicalLabel, Priority
from email_intelligence.schemas import (
    ActionType,
    AnalysisResult,
    DraftRequest,
    EmailContent,
    ModelMetadata,
    RecommendedAction,
)


def analysis_result(response_required: bool = False) -> AnalysisResult:
    return AnalysisResult(
        classification=CanonicalLabel.FYI,
        classification_confidence=0.8,
        priority=Priority.MEDIUM,
        priority_confidence=0.7,
        summary="A concise update.",
        response_required=response_required,
        recommended_actions=[
            RecommendedAction(
                action=ActionType.FLAG_FOR_REVIEW,
                confidence=0.75,
                reason="Review the informational message.",
            )
        ],
        model=ModelMetadata(
            provider="test",
            model_name="test-model",
            model_version="1",
        ),
    )


def test_confidence_must_be_bounded() -> None:
    payload = analysis_result().model_dump()
    payload["classification_confidence"] = 1.1
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(payload)


def test_draft_rejected_when_reply_is_not_required() -> None:
    with pytest.raises(ValidationError):
        DraftRequest(
            email=EmailContent(
                email_id="email-1",
                subject="Update",
                body_text="The project has been updated.",
            ),
            analysis=analysis_result(),
        )
