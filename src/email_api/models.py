"""Persistence models for emails, analyses, drafts, and review audit events."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from email_api.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EmailRecordDB(Base):
    __tablename__ = "email_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sender: Mapped[str] = mapped_column(String(1_000), default="")
    recipients: Mapped[list[str]] = mapped_column(JSON, default=list)
    subject: Mapped[str] = mapped_column(String(500))
    body_text: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    processing_status: Mapped[str] = mapped_column(String(32), default="processing", index=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    analysis_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    draft_error: Mapped[str | None] = mapped_column(String(500), nullable=True)

    current_classification: Mapped[str | None] = mapped_column(String(32), index=True)
    current_priority: Mapped[str | None] = mapped_column(String(16), index=True)
    current_action: Mapped[str | None] = mapped_column(String(32), index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    analyses: Mapped[list[EmailAnalysisDB]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )
    drafts: Mapped[list[EmailDraftDB]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )
    audit_events: Mapped[list[ReviewAuditEventDB]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_email_review_queue", "review_status", "current_priority", "created_at"),
    )


class EmailAnalysisDB(Base):
    __tablename__ = "email_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email_id: Mapped[str] = mapped_column(
        ForeignKey("email_records.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(100))
    model_name: Mapped[str] = mapped_column(String(200))
    model_version: Mapped[str] = mapped_column(String(100))
    classification_confidence: Mapped[float] = mapped_column(Float)
    priority_confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    email: Mapped[EmailRecordDB] = relationship(back_populates="analyses")

    __table_args__ = (UniqueConstraint("email_id", "version", name="uq_analysis_version"),)


class EmailDraftDB(Base):
    __tablename__ = "email_drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email_id: Mapped[str] = mapped_column(
        ForeignKey("email_records.id", ondelete="CASCADE"), index=True
    )
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("email_analyses.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)
    tone: Mapped[str] = mapped_column(String(50))
    source: Mapped[str] = mapped_column(String(20), default="model")
    model_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    email: Mapped[EmailRecordDB] = relationship(back_populates="drafts")

    __table_args__ = (UniqueConstraint("email_id", "version", name="uq_draft_version"),)


class ReviewAuditEventDB(Base):
    __tablename__ = "review_audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email_id: Mapped[str] = mapped_column(
        ForeignKey("email_records.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    before_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    after_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    email: Mapped[EmailRecordDB] = relationship(back_populates="audit_events")
