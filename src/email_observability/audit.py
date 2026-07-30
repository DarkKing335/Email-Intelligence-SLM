"""Append-only audit trail (US-5.3).

Records important model and user actions to a durable, append-only JSONL file
so that decisions can be reviewed later. Unlike application logs (US-5.1),
which *redact* identities, the audit trail intentionally records the actor and
resource so actions are filterable by user or by email — that is its purpose.

Each event captures a timestamp, the actor, the action, the resource acted on,
the outcome, and — when relevant — the model and dataset versions that were in
effect (lineage), tying back to the Model Registry (US-6.4).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field


class AuditAction(StrEnum):
    """Auditable action types."""

    CLASSIFY = "classify"
    SUMMARIZE = "summarize"
    DRAFT_CREATED = "draft_created"
    APPROVE = "approve"
    OVERRIDE = "override"
    REJECT = "reject"
    SEND = "send"
    MODEL_REGISTERED = "model_registered"
    MODEL_PROMOTED = "model_promoted"
    DEPLOY = "deploy"


class AuditEvent(BaseModel):
    """A single immutable audit record."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    actor: str = Field(description="Who performed the action (user id / email / 'system').")
    action: AuditAction
    resource: str | None = Field(
        default=None, description="What was acted on (e.g. an email id or thread id)."
    )
    outcome: str | None = Field(default=None, description="Result, e.g. 'approved'.")
    model_version: str | None = None
    dataset_version: str | None = None
    detail: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class AuditTrail:
    """Append-only JSONL audit trail with simple filtering.

    Parameters
    ----------
    path:
        JSONL file that events are appended to. Parent dirs are created.
    """

    def __init__(self, path: str | Path = "reports/audit/audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Writing ───────────────────────────────────────────────────────────────

    def record(
        self,
        action: AuditAction | str,
        actor: str,
        *,
        resource: str | None = None,
        outcome: str | None = None,
        model_version: str | None = None,
        dataset_version: str | None = None,
        detail: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AuditEvent:
        """Append an audit event and return it."""
        event = AuditEvent(
            action=AuditAction(action),
            actor=actor,
            resource=resource,
            outcome=outcome,
            model_version=model_version,
            dataset_version=dataset_version,
            detail=detail,
            metadata=metadata or {},
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        logger.info(
            "audit event recorded",
            audit_action=event.action.value,
            actor=actor,
            resource=resource,
        )
        return event

    # ── Reading ───────────────────────────────────────────────────────────────

    def all(self) -> list[AuditEvent]:
        """Return every recorded event in insertion order."""
        if not self._path.exists():
            return []
        events: list[AuditEvent] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(AuditEvent(**json.loads(line)))
        return events

    def query(
        self,
        actor: str | None = None,
        resource: str | None = None,
        action: AuditAction | str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Return events matching the given filters (all optional).

        ``actor`` and ``resource`` satisfy the "filter by user or email"
        criterion. ``since``/``until`` are ISO-8601 timestamp bounds.
        """
        action_val = AuditAction(action).value if action is not None else None
        results: list[AuditEvent] = []
        for event in self.all():
            if actor is not None and event.actor != actor:
                continue
            if resource is not None and event.resource != resource:
                continue
            if action_val is not None and event.action.value != action_val:
                continue
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue
            results.append(event)
        if limit is not None:
            results = results[-limit:]
        return results
