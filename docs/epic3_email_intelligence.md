# Epic 3: Email Intelligence Workflow

Epic 3 is an API-only workflow around a pluggable model gateway. It stores an
incoming email, requests structured intelligence, exposes a review queue, and
records edits and approval decisions. It never sends, archives, or changes a
mailbox.

## Services

- `email_inference` exposes a stable model contract. The default `mock` provider
  makes local development deterministic.
- `email_api` persists email analysis and exposes review operations for the
  Epic 4 user interface.
- PostgreSQL stores immutable model analyses, versioned drafts, and audit events.

## Plugging in the fine-tuned model

Deploy the model behind endpoints compatible with:

- `POST /v1/analyze`
- `POST /v1/generate-draft`

Set `MODEL_PROVIDER=http` and `MODEL_ENDPOINT` on the inference service. If the
model must be loaded in-process, implement `ModelProvider` and configure
`MODEL_PROVIDER=custom` with `MODEL_PROVIDER_CLASS=package.module:ClassName`.
No backend workflow or database changes are required.

The provider must return the schemas defined in `email_intelligence.schemas`.
Classification and priority confidence are required numbers between zero and
one. Unknown or malformed model output is rejected.

## Review API

Create and synchronously analyze an email with `POST /v1/emails`. Use
`GET /v1/review-queue` to list results, `PATCH /v1/emails/{id}/review` to record
edits, and `POST /v1/emails/{id}/decision` to approve or reject. Decisions are
audit-only in Epic 3.

Run both services locally with:

```bash
uvicorn email_inference.api:app --port 8000
uvicorn email_api.main:app --port 8080
```
