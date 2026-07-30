"""Streamlit dashboard for the Email Intelligence Assistant.

Delivers three member-4 stories against the Epic 3 API:

* **US-4.5 Dashboard** — a queue of analyzed emails with status/history and a
  drill-in detail view.
* **US-4.6 Search History** — free-text search over sender/subject/body.
* **US-5.4 Human Review** — review the model's suggestion (with confidence) and
  approve / edit / reject; nothing is auto-executed.

Run with:  ``streamlit run src/email_dashboard/app.py``
Backend URL comes from the ``API_URL`` env var (default ``http://localhost:8080``).
"""

from __future__ import annotations

import os

import streamlit as st
from loguru import logger

from email_dashboard.api_client import ApiClient, ApiError
from email_observability import (
    AuditAction,
    AuditTrail,
    MetricsCollector,
    configure_logging,
    request_context,
)

API_URL = os.environ.get("API_URL", "http://localhost:8080")
REVIEWER_DEFAULT = os.environ.get("DASHBOARD_REVIEWER", "dashboard-user")

CLASSIFICATIONS = [
    "respond",
    "fyi",
    "support",
    "billing",
    "security",
    "unsubscribe",
    "spam",
    "archive",
]
PRIORITIES = ["high", "medium", "low"]
ACTIONS = ["reply", "archive", "ignore", "unsubscribe", "flag_for_review"]
TONES = ["professional", "friendly", "concise"]


# ── Shared singletons (wired-in observability) ────────────────────────────────


@st.cache_resource
def _bootstrap() -> tuple[ApiClient, MetricsCollector, AuditTrail]:
    configure_logging(level="INFO", json_logs=True)
    client = ApiClient(API_URL)
    metrics = MetricsCollector()
    audit = AuditTrail(os.environ.get("AUDIT_LOG", "reports/audit/dashboard.jsonl"))
    logger.info("dashboard started", api_url=API_URL)
    return client, metrics, audit


client, metrics, audit = _bootstrap()


def _record(action: str, email_id: str, extra: dict | None = None) -> None:
    """Log + count + audit a user action (US-5.1 / 5.2 / 5.3)."""
    metrics.incr("dashboard_actions_total", action=action)
    with request_context():
        logger.info("dashboard action", action=action, email_id=email_id, **(extra or {}))


# ── Sidebar: connection, reviewer, filters, search, compose ───────────────────

st.set_page_config(page_title="Email Intelligence", page_icon="📧", layout="wide")
st.title("📧 Email Intelligence — Review Dashboard")

with st.sidebar:
    st.header("Connection")
    try:
        health = client.health()
        if health.get("status") == "ok":
            st.success(f"API: {health.get('status')}")
        else:
            st.warning(
                f"API: {health.get('status')} · db={health.get('database')} · "
                f"inference={health.get('inference')}"
            )
    except ApiError as exc:
        st.error(f"API unreachable at {API_URL}\n\n{exc.detail}")

    reviewer_id = st.text_input("Reviewer id", value=REVIEWER_DEFAULT)
    st.session_state["reviewer_id"] = reviewer_id

    st.header("Filters & search")  # US-4.6
    search = st.text_input("Search (sender / subject / body)")
    f_review = st.selectbox("Review status", ["", "pending", "approved", "rejected"])
    f_class = st.selectbox("Classification", [""] + CLASSIFICATIONS)
    f_priority = st.selectbox("Priority", [""] + PRIORITIES)

    with st.expander("➕ New email (create & analyze)"):
        with st.form("new_email"):
            ne_sender = st.text_input("Sender", value="someone@example.com")
            ne_subject = st.text_input("Subject", value="Please review the Q3 invoice")
            ne_body = st.text_area(
                "Body", value="Can you review the attached invoice and let me know by Friday?"
            )
            if st.form_submit_button("Create") and ne_subject and ne_body:
                try:
                    created = client.create_email(
                        subject=ne_subject, body_text=ne_body, sender=ne_sender
                    )
                    _record("create", created["id"])
                    st.session_state["selected"] = created["id"]
                    st.success(f"Created & analyzed: {created['id'][:8]}")
                except ApiError as exc:
                    st.error(f"Create failed: {exc.detail}")


# ── Queue (US-4.5) ────────────────────────────────────────────────────────────


def _load_queue() -> list[dict]:
    try:
        page = client.list_queue(
            search=search or None,
            review_status=f_review or None,
            classification=f_class or None,
            priority=f_priority or None,
            limit=100,
        )
    except ApiError as exc:
        st.error(f"Could not load queue: {exc.detail}")
        return []
    st.caption(f"{page['total']} email(s) match")
    return page["items"]


left, right = st.columns([2, 3])

with left:
    st.subheader("Queue")
    items = _load_queue()
    if not items:
        st.info("No emails. Use ‘New email’ in the sidebar to add one.")
    for item in items:
        label = (
            f"**{item['subject'] or '(no subject)'}**  \n"
            f"{item['sender'] or '—'}  \n"
            f"`{item.get('classification') or '?'}` · "
            f"`{item.get('priority') or '?'}` · {item['review_status']}"
        )
        if st.button(label, key=f"pick-{item['id']}", use_container_width=True):
            st.session_state["selected"] = item["id"]


# ── Detail + Human Review (US-5.4) ────────────────────────────────────────────

with right:
    selected = st.session_state.get("selected")
    if not selected:
        st.info("Select an email from the queue to review it.")
    else:
        try:
            email = client.get_email(selected)
        except ApiError as exc:
            st.error(f"Could not load email: {exc.detail}")
            email = None

        if email:
            st.subheader(email["subject"] or "(no subject)")
            st.caption(
                f"From {email['sender'] or '—'} · status: "
                f"{email['processing_status']} / {email['review_status']}"
            )

            analysis = email.get("analysis")
            if analysis:
                c1, c2 = st.columns(2)
                c1.metric(
                    "Classification",
                    analysis["classification"],
                    f"{analysis['classification_confidence']:.0%} conf.",
                )
                c2.metric(
                    "Priority",
                    analysis["priority"],
                    f"{analysis['priority_confidence']:.0%} conf.",
                )
                st.write("**Summary**")
                st.write(analysis["summary"])
                actions = ", ".join(a["action"] for a in analysis["recommended_actions"])
                st.caption(f"Recommended: {actions}")

            with st.expander("Original email body"):
                st.write(email["body_text"])

            # ── Draft (reply-style tone) ──
            st.divider()
            st.write("**Draft reply**")
            draft = email.get("draft")
            if draft:
                st.text_area("Current draft", value=draft["body"], key="draft_view", height=140)
                st.caption(
                    f"tone: {draft['tone']} · source: {draft['source']} · v{draft['version']}"
                )
            tone = st.selectbox("Tone", TONES, key="tone")
            if st.button("Generate / regenerate draft"):
                try:
                    client.generate_draft(selected, tone=tone)
                    _record("generate_draft", selected, {"tone": tone})
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Draft failed: {exc.detail}")

            # ── Human review: edit + approve/reject (no auto-execute) ──
            st.divider()
            st.write("**Human review**")
            reviewed = email["reviewed"]
            edit_class = st.selectbox(
                "Classification",
                CLASSIFICATIONS,
                index=CLASSIFICATIONS.index(reviewed["classification"])
                if reviewed["classification"] in CLASSIFICATIONS
                else 0,
            )
            edit_priority = st.selectbox(
                "Priority",
                PRIORITIES,
                index=PRIORITIES.index(reviewed["priority"])
                if reviewed["priority"] in PRIORITIES
                else 1,
            )
            edit_draft = st.text_area(
                "Edit draft body (optional)",
                value=draft["body"] if draft else "",
                height=120,
            )

            b1, b2, b3 = st.columns(3)
            reviewer = st.session_state.get("reviewer_id", REVIEWER_DEFAULT)

            if b1.button("💾 Save edits"):
                try:
                    client.review(
                        selected,
                        reviewer,
                        classification=edit_class,
                        priority=edit_priority,
                        draft_body=edit_draft or None,
                    )
                    _record("edit", selected)
                    st.success("Saved.")
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Save failed: {exc.detail}")

            if b2.button("✅ Approve", type="primary"):
                try:
                    client.decide(selected, reviewer, "approve")
                    _record("approve", selected)
                    audit.record(
                        AuditAction.APPROVE, actor=reviewer, resource=selected, outcome="approved"
                    )
                    st.success("Approved.")
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Approve failed: {exc.detail}")

            if b3.button("❌ Reject"):
                try:
                    client.decide(selected, reviewer, "reject")
                    _record("reject", selected)
                    audit.record(
                        AuditAction.REJECT, actor=reviewer, resource=selected, outcome="rejected"
                    )
                    st.warning("Rejected.")
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Reject failed: {exc.detail}")

            # ── Audit history (from the backend) ──
            if email.get("audit_events"):
                st.divider()
                st.write("**History**")
                for event in email["audit_events"]:
                    who = event["reviewer_id"] or "system"
                    st.caption(
                        f"{event['created_at'][:19].replace('T', ' ')} · "
                        f"{event['event_type']} · {who}"
                    )
