"""Synchronous client for the Email Intelligence API (US-4.5/4.6/5.4).

A thin, dependency-light wrapper the Streamlit dashboard uses to talk to the
Epic 3 backend. Kept free of any Streamlit imports so it is unit-testable with
``httpx.MockTransport`` (no running server required).
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    """Raised when the API returns an error or is unreachable."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


class ApiClient:
    """Small synchronous client for the backend REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # ── Health (US-6.6 surface of the API) ────────────────────────────────────

    def health(self) -> dict:
        return self._request("GET", "/health")

    # ── Dashboard + search (US-4.5 / US-4.6) ──────────────────────────────────

    def list_queue(
        self,
        *,
        search: str | None = None,
        review_status: str | None = None,
        processing_status: str | None = None,
        classification: str | None = None,
        priority: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        for key, value in (
            ("search", search),
            ("review_status", review_status),
            ("processing_status", processing_status),
            ("classification", classification),
            ("priority", priority),
        ):
            if value:
                params[key] = value
        return self._request("GET", "/v1/review-queue", params=params)

    def get_email(self, email_id: str) -> dict:
        return self._request("GET", f"/v1/emails/{email_id}")

    def create_email(
        self,
        *,
        subject: str,
        body_text: str,
        sender: str = "",
        recipients: list[str] | None = None,
        external_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "subject": subject,
            "body_text": body_text,
            "sender": sender,
            "recipients": recipients or [],
        }
        if external_id:
            payload["external_id"] = external_id
        if thread_id:
            payload["thread_id"] = thread_id
        return self._request("POST", "/v1/emails", json=payload)

    # ── Draft generation (reply-style tone hook, US-3.6/Epic 2) ───────────────

    def generate_draft(self, email_id: str, tone: str = "professional") -> dict:
        return self._request("POST", f"/v1/emails/{email_id}/draft", json={"tone": tone})

    def reanalyze(self, email_id: str) -> dict:
        return self._request("POST", f"/v1/emails/{email_id}/reanalyze")

    # ── Human review (US-5.4) ─────────────────────────────────────────────────

    def review(
        self,
        email_id: str,
        reviewer_id: str,
        *,
        classification: str | None = None,
        priority: str | None = None,
        recommended_action: str | None = None,
        draft_body: str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"reviewer_id": reviewer_id}
        for key, value in (
            ("classification", classification),
            ("priority", priority),
            ("recommended_action", recommended_action),
            ("draft_body", draft_body),
        ):
            if value is not None:
                payload[key] = value
        return self._request("PATCH", f"/v1/emails/{email_id}/review", json=payload)

    def decide(
        self,
        email_id: str,
        reviewer_id: str,
        decision: str,
        changes: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"reviewer_id": reviewer_id, "decision": decision}
        if changes:
            payload["changes"] = changes
        return self._request("POST", f"/v1/emails/{email_id}/decision", json=payload)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            raise ApiError(0, f"connection error: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))
        return response.json() if response.content else {}


def _detail(response: httpx.Response) -> Any:
    try:
        body = response.json()
    except ValueError:
        return response.text
    return body.get("detail", body) if isinstance(body, dict) else body
