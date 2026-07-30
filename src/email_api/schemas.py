"""Public API schemas for Epic 3."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from email_data_engineering.domain.taxonomy import CanonicalLabel, Priority
from email_intelligence.schemas import ActionType, AnalysisResult, UserContext


class EmailCreate(BaseModel):
    external_id: str | None = Field(default=None, max_length=255)
    thread_id: str | None = Field(default=None, max_length=255)
    sender: str = Field(default="", max_length=1_000)
    recipients: list[str] = Field(default_factory=list)
    subject: str = Field(min_length=1, max_length=500)
    body_text: str = Field(min_length=1, max_length=50_000)
    received_at: datetime | None = None
    user_context: UserContext | None = None


class ReviewChanges(BaseModel):
    classification: CanonicalLabel | None = None
    priority: Priority | None = None
    recommended_action: ActionType | None = None
    draft_body: str | None = Field(default=None, min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def _require_a_change(self) -> ReviewChanges:
        if all(
            value is None
            for value in (
                self.classification,
                self.priority,
                self.recommended_action,
                self.draft_body,
            )
        ):
            raise ValueError("at least one review field must be supplied")
        return self


class ReviewUpdate(ReviewChanges):
    reviewer_id: str = Field(min_length=1, max_length=255)


class DecisionRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=255)
    decision: Literal["approve", "reject"]
    changes: ReviewChanges | None = None


class DraftGenerateRequest(BaseModel):
    tone: str = Field(default="professional", min_length=1, max_length=50)
    user_context: UserContext | None = None


class DraftView(BaseModel):
    id: str
    version: int
    body: str
    tone: str
    source: str
    model_metadata: dict
    created_at: datetime


class AuditEventView(BaseModel):
    id: str
    event_type: str
    reviewer_id: str | None
    before_state: dict | None
    after_state: dict | None
    created_at: datetime


class ReviewedValues(BaseModel):
    classification: CanonicalLabel | None
    priority: Priority | None
    recommended_action: ActionType | None


class EmailDetail(BaseModel):
    id: str
    external_id: str | None
    thread_id: str | None
    sender: str
    recipients: list[str]
    subject: str
    body_text: str
    received_at: datetime | None
    processing_status: str
    review_status: str
    analysis_error: str | None
    draft_error: str | None
    reviewed: ReviewedValues
    analysis_version: int | None
    analysis: AnalysisResult | None
    draft: DraftView | None
    audit_events: list[AuditEventView] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ReviewQueueItem(BaseModel):
    id: str
    sender: str
    subject: str
    processing_status: str
    review_status: str
    classification: CanonicalLabel | None
    priority: Priority | None
    recommended_action: ActionType | None
    summary: str | None
    classification_confidence: float | None
    priority_confidence: float | None
    created_at: datetime


class ReviewQueuePage(BaseModel):
    items: list[ReviewQueueItem]
    total: int
    limit: int
    offset: int
