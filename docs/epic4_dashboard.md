# Epic 4/5: Review Dashboard (US-4.5, US-4.6, US-5.4)

A Streamlit frontend over the Epic 3 API that delivers three member-4 stories:

| Story | Delivered by |
| :--- | :--- |
| **US-4.5 Dashboard** | A queue of analyzed emails (status, classification, priority) with a drill-in detail view. |
| **US-4.6 Search History** | Free-text search over sender / subject / body, combinable with filters. |
| **US-5.4 Human Review** | Review the model's suggestion + confidence and **approve / edit / reject**; nothing is auto-executed; every action is logged to the backend audit trail. |

Code: `src/email_dashboard/` (`app.py` UI, `api_client.py` client). Backend
search added in `email_api/main.py`.

---

## 1. Architecture

```
┌──────────────────┐   ApiClient (httpx)   ┌──────────────────┐   HttpInferenceGateway  ┌──────────────────┐
│  Streamlit app   │ ────────────────────> │   email_api      │ ──────────────────────> │  email_inference │
│  (email_dashboard│  /v1/review-queue     │  (FastAPI, Epic3)│   /v1/analyze           │  (mock provider) │
│   /app.py)       │  /v1/emails/{id}...   │  + SQLite        │   /v1/generate-draft    │                  │
└──────────────────┘                       └──────────────────┘                         └──────────────────┘
        │
        └── wires in Epic 5: configure_logging(), MetricsCollector, AuditTrail
```

The `ApiClient` carries **no Streamlit import**, so it is unit-tested with
`httpx.MockTransport` (no server needed). The whole stack runs offline: SQLite +
the deterministic `MockModelProvider`.

---

## 2. Backend addition — keyword search (US-4.6)

`GET /v1/review-queue` gained a `search` parameter that matches sender, subject,
or body (case-insensitive), combinable with the existing filters:

```
GET /v1/review-queue?search=invoice&priority=high
```

Implemented as an `OR ILIKE` over `sender`, `subject`, `body_text`.

> **Known limitation (needs US-4.1 Auth):** results are **not yet scoped to a
> logged-in user** — `EmailRecordDB` has no owner column. Per-user scoping
> depends on the auth/user model from US-4.1, which is not built. Once an
> `owner` field exists, add it to the same filter list.

---

## 3. The dashboard (US-4.5 / 5.4)

- **Sidebar:** live API health, reviewer id, filters (review status /
  classification / priority), the **search box**, and a “New email” form that
  creates + analyzes an email.
- **Queue (left):** every matching email as a button — subject, sender, and
  `classification · priority · review_status`.
- **Detail + review (right):** classification & priority **with confidence**,
  summary, recommended actions, the original body, the draft (with a
  **tone** selector — professional / friendly / concise), and the review
  controls: **💾 Save edits**, **✅ Approve**, **❌ Reject**, plus the backend
  audit history.

**Safety (US-5.4):** the UI never sends or executes anything on its own —
approve/reject only set the review status through the API, and the backend
records every decision. This matches the "no auto-execute by default"
acceptance criterion.

**Observability wiring (US-5.1/5.2/5.3):** on startup the app calls
`configure_logging()`, holds a `MetricsCollector` (counts `dashboard_actions_total`
by action), and mirrors approve/reject decisions into an `AuditTrail`.

---

## 4. Running it

```powershell
# 1. install (frontend extra brings Streamlit)
.venv\Scripts\pip install -e ".[dev,frontend]"

# 2. inference service (mock model) and API — two terminals
.venv\Scripts\python -m uvicorn email_inference.api:app --port 8000
.venv\Scripts\python -m uvicorn email_api.main:app --port 8080     # uses SQLite by default

# 3. dashboard
$env:API_URL = "http://localhost:8080"
.venv\Scripts\streamlit run src/email_dashboard/app.py
```

Or via Docker Compose (`docker/Dockerfile.frontend` already runs
`streamlit run src/email_dashboard/app.py`).

> If port 8080 is blocked locally, use another (e.g. `--port 8090`) and set
> `API_URL` to match.

---

## 5. Testing

```powershell
.venv\Scripts\python -m pytest tests/dashboard tests/epic3/test_search.py -v
```

| Test file | Covers |
| :--- | :--- |
| `tests/epic3/test_search.py` | Search matches subject / sender / body; case-insensitive; combines with filters; no-match returns empty. |
| `tests/dashboard/test_api_client.py` | Query-param building, payload shaping (create / draft-tone / review / decide), response parsing, and error mapping to `ApiError` — all via `httpx.MockTransport`. |

The full flow was also verified end-to-end against the live two-service stack
(create → analyze → queue → search → generate friendly draft → edit → approve →
audit trail).

---

## 6. Follow-ups

- **US-4.1 Auth + per-user scoping** for US-4.6 (owner column + filter).
- Optionally converge the backend's per-email audit (`ReviewAuditEventDB`) and
  the system-wide `email_observability.AuditTrail` behind one review surface.
- Expose the API's readiness via `email_mlops.health` / a `/metrics` route for
  the monitoring story.
