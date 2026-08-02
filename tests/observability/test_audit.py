"""Unit tests for the audit trail (US-5.3)."""

from __future__ import annotations

from pathlib import Path

from email_observability.audit import AuditAction, AuditEvent, AuditTrail


def _trail(tmp_dir: Path) -> AuditTrail:
    return AuditTrail(path=tmp_dir / "audit" / "audit.jsonl")


class TestAuditTrail:
    def test_record_and_read_back(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        event = trail.record(
            AuditAction.APPROVE,
            actor="alice@corp.com",
            resource="email-42",
            outcome="approved",
            model_version="2",
            dataset_version="v0.2.0",
        )
        assert isinstance(event, AuditEvent)

        events = trail.all()
        assert len(events) == 1
        assert events[0].actor == "alice@corp.com"
        assert events[0].action == AuditAction.APPROVE
        assert events[0].model_version == "2"
        assert events[0].dataset_version == "v0.2.0"

    def test_appends_in_order(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        trail.record("classify", actor="system", resource="e1")
        trail.record("draft_created", actor="system", resource="e1")
        trail.record("approve", actor="bob", resource="e1")
        actions = [e.action.value for e in trail.all()]
        assert actions == ["classify", "draft_created", "approve"]

    def test_filter_by_actor(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        trail.record("approve", actor="alice", resource="e1")
        trail.record("override", actor="bob", resource="e2")
        results = trail.query(actor="alice")
        assert len(results) == 1 and results[0].actor == "alice"

    def test_filter_by_resource(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        trail.record("classify", actor="system", resource="e1")
        trail.record("classify", actor="system", resource="e2")
        assert len(trail.query(resource="e2")) == 1

    def test_filter_by_action(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        trail.record("approve", actor="a", resource="e1")
        trail.record("reject", actor="a", resource="e2")
        results = trail.query(action=AuditAction.REJECT)
        assert len(results) == 1 and results[0].action == AuditAction.REJECT

    def test_limit_returns_most_recent(self, tmp_dir: Path) -> None:
        trail = _trail(tmp_dir)
        for i in range(5):
            trail.record("classify", actor="system", resource=f"e{i}")
        recent = trail.query(limit=2)
        assert len(recent) == 2
        assert [e.resource for e in recent] == ["e3", "e4"]

    def test_query_empty_trail(self, tmp_dir: Path) -> None:
        assert _trail(tmp_dir).query(actor="nobody") == []
