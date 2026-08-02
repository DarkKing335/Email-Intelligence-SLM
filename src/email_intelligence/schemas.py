"""Provider-neutral request and response schemas for email intelligence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, model_validator

from email_data_engineering.domain.taxonomy import CanonicalLabel, Priority


class ActionType(StrEnum):
    REPLY = "reply"
    ARCHIVE = "archive"
    IGNORE = "ignore"
    UNSUBSCRIBE = "unsubscribe"
    FLAG_FOR_REVIEW = "flag_for_review"


class EmailContent(BaseModel):
    email_id: str
    external_id: str | None = None
    thread_id: str | None = None
    sender: str = ""
    recipients: list[str] = Field(default_factory=list)
    subject: str = Field(min_length=1, max_length=500)
    body_text: str = Field(min_length=1, max_length=50_000)
    received_at: datetime | None = None


class UserContext(BaseModel):
    role: str | None = None
    preferred_tone: str = Field(default="professional", max_length=50)
    additional_context: str | None = Field(default=None, max_length=2_000)


class AnalysisRequest(BaseModel):
    email: EmailContent
    user_context: UserContext | None = None


class TextEntity(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    normalized: str | None = Field(default=None, max_length=500)


class LinkEntity(BaseModel):
    text: str = Field(min_length=1, max_length=2_048)
    url: HttpUrl


class MoneyEntity(BaseModel):
    text: str = Field(min_length=1, max_length=100)
    amount: float
    currency: str | None = Field(default=None, min_length=3, max_length=3)


class ExtractedEntities(BaseModel):
    people: list[TextEntity] = Field(default_factory=list)
    dates: list[TextEntity] = Field(default_factory=list)
    links: list[LinkEntity] = Field(default_factory=list)
    monetary_amounts: list[MoneyEntity] = Field(default_factory=list)
    references: list[TextEntity] = Field(default_factory=list)


class RecommendedAction(BaseModel):
    action: ActionType
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1_000)


class ModelMetadata(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=200)
    model_version: str = Field(min_length=1, max_length=100)
    schema_version: str = "1.0"


class AnalysisResult(BaseModel):
    classification: CanonicalLabel
    classification_confidence: float = Field(ge=0, le=1)
    priority: Priority
    priority_confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1, max_length=500)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    response_required: bool
    recommended_actions: list[RecommendedAction] = Field(min_length=1)
    model: ModelMetadata


class DraftRequest(BaseModel):
    email: EmailContent
    analysis: AnalysisResult
    tone: str = Field(default="professional", min_length=1, max_length=50)
    user_context: UserContext | None = None

    @model_validator(mode="after")
    def _ensure_reply_is_required(self) -> DraftRequest:
        reply_recommended = any(
            item.action is ActionType.REPLY for item in self.analysis.recommended_actions
        )
        if not self.analysis.response_required or not reply_recommended:
            raise ValueError("a draft is allowed only when a reply is required and recommended")
        return self


class DraftResult(BaseModel):
    body: str = Field(min_length=1, max_length=20_000)
    tone: str = Field(min_length=1, max_length=50)
    model: ModelMetadata
