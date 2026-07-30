"""Create Epic 3 workflow tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "email_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("thread_id", sa.String(length=255), nullable=True),
        sa.Column("sender", sa.String(length=1000), nullable=False),
        sa.Column("recipients", sa.JSON(), nullable=False),
        sa.Column("subject", sa.String(length=500), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_status", sa.String(length=32), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("analysis_error", sa.String(length=500), nullable=True),
        sa.Column("draft_error", sa.String(length=500), nullable=True),
        sa.Column("current_classification", sa.String(length=32), nullable=True),
        sa.Column("current_priority", sa.String(length=16), nullable=True),
        sa.Column("current_action", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_email_records_created_at", "email_records", ["created_at"])
    op.create_index(
        "ix_email_review_queue",
        "email_records",
        ["review_status", "current_priority", "created_at"],
    )
    for column in (
        "processing_status",
        "review_status",
        "current_classification",
        "current_priority",
        "current_action",
    ):
        op.create_index(f"ix_email_records_{column}", "email_records", [column])

    op.create_table(
        "email_analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("classification_confidence", sa.Float(), nullable=False),
        sa.Column("priority_confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["email_id"], ["email_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email_id", "version", name="uq_analysis_version"),
    )
    op.create_index("ix_email_analyses_email_id", "email_analyses", ["email_id"])

    op.create_table(
        "email_drafts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email_id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("tone", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("model_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["email_analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["email_id"], ["email_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email_id", "version", name="uq_draft_version"),
    )
    op.create_index("ix_email_drafts_analysis_id", "email_drafts", ["analysis_id"])
    op.create_index("ix_email_drafts_email_id", "email_drafts", ["email_id"])

    op.create_table(
        "review_audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("reviewer_id", sa.String(length=255), nullable=True),
        sa.Column("before_state", sa.JSON(), nullable=True),
        sa.Column("after_state", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["email_id"], ["email_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_review_audit_events_email_id", "review_audit_events", ["email_id"])
    op.create_index("ix_review_audit_events_event_type", "review_audit_events", ["event_type"])


def downgrade() -> None:
    op.drop_table("review_audit_events")
    op.drop_table("email_drafts")
    op.drop_table("email_analyses")
    op.drop_table("email_records")
