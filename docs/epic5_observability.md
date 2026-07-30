# Epic 5: Safety & Observability (US-5.1 – 5.3)

This document details the **Logging (US-5.1)**, **Monitoring (US-5.2)**, and
**Audit Trail (US-5.3)** modules. They live in a self-contained package,
`src/email_observability/`, with **no external observability dependencies**
(no Prometheus/OpenTelemetry client, no database) — everything runs and is
tested offline. Configuration lives in `configs/observability.yaml`.

> US-5.4 (Human Review) is UI-coupled and is delivered together with the
> dashboard (US-4.5); it is out of scope for this package.

---

## 1. Logging (US-5.1)

`email_observability.logging` — structured logging built on Loguru.

| Acceptance criterion | Implementation |
| :--- | :--- |
| Structured & searchable | JSON lines with stable fields (`timestamp`, `level`, `message`, `request_id`, `module`, …). |
| End-to-end traceability | `request_context()` binds a request id (via `contextvars`) that every log line in that scope carries. |
| No sensitive data | A redaction patcher masks PII in messages (email/SSN/credit-card/IP/phone) and masks the *values* of sensitive keys (`password`, `token`, …) in `extra`. |
| Errors distinct | The `level` field separates errors; an optional `error_log_file` also captures `ERROR`+ to its own sink. |

```python
from loguru import logger
from email_observability import configure_logging, request_context

configure_logging(level="INFO", json_logs=True, redact=True)

with request_context() as rid:        # or request_context("trace-abc")
    logger.bind(password="hunter2").info("emailing john@example.com")
    # -> {"level":"INFO","message":"emailing [REDACTED]",
    #     "request_id":"...","extra":{"password":"***"}, ...}
```

Config (`configs/observability.yaml`):

```yaml
logging:
  level: "INFO"
  json: true
  redact: true
  error_log_file: "logs/errors.log"
  sensitive_keys: [password, secret, token, api_key, authorization]
```

---

## 2. Monitoring (US-5.2)

`email_observability.metrics` — an in-process `MetricsCollector` plus alert rules.

| Acceptance criterion | Implementation |
| :--- | :--- |
| Health metrics for key services | `record_health(report)` folds a US-6.6 `HealthReport` into gauges (`service_health` = 2/1/0 for healthy/degraded/unhealthy; `dependency_up`). |
| Degraded state shown clearly | `service_health == 1` is the degraded value; a shipped alert rule fires on `service_health < 1`. |
| Trackable over time | `record_sample()` appends the current snapshot to a bounded history (`history_size`). |
| Configurable alerts | `AlertRule`s (metric, comparison, threshold, severity) loaded from config via `alert_rules_from_config`, evaluated with `evaluate_alerts`. |

```python
from email_observability import MetricsCollector, alert_rules_from_config, evaluate_alerts

m = MetricsCollector()
m.incr("errors_total")
with m.timer("inference_latency_ms"):
    run_inference()
m.record_health(checker.check_readiness())   # from US-6.6

snapshot = m.snapshot()          # {"errors_total":1, "inference_latency_ms.p95":..., "service_health":...}
prom_text = m.to_prometheus()    # standard scraper format, no dependency

rules = alert_rules_from_config(config["monitoring"]["alerts"])
for alert in evaluate_alerts(rules, snapshot):
    logger.warning("ALERT", alert=alert.to_dict())
```

Metric kinds: **counter** (`incr`), **gauge** (`gauge`), **timing** (`observe`
/ `timer`). Timings expand in the snapshot to `name.count` / `.avg` / `.p95` /
`.max`.

---

## 3. Audit Trail (US-5.3)

`email_observability.audit` — an append-only JSONL record of important actions.

> Unlike application logs, the audit trail **intentionally records identities**
> (actor, resource) — that is what makes it filterable by user or email. It is
> not redacted.

| Acceptance criterion | Implementation |
| :--- | :--- |
| Every important action timestamped | `AuditTrail.record()` appends an `AuditEvent` with a UTC timestamp and a unique id. |
| Model + dataset version recorded | `model_version` / `dataset_version` fields (lineage back to US-6.4). |
| Approve/override auditable | `AuditAction` includes `approve`, `override`, `reject`, `send`, plus model lifecycle actions. |
| Filterable by user or email | `query(actor=…, resource=…, action=…, since=…, until=…, limit=…)`. |

```python
from email_observability import AuditTrail, AuditAction

trail = AuditTrail("reports/audit/audit.jsonl")
trail.record(AuditAction.APPROVE, actor="alice@corp.com",
             resource="email-42", outcome="sent", model_version="2")

trail.query(actor="alice@corp.com")   # review one user's decisions
trail.query(resource="email-42")      # everything that happened to one email
```

---

## 4. CLI (`email-obs`)

Registered as `email-obs = "cli.observability:cli"`. Config via `-c/--config`
(defaults to `configs/observability.yaml`).

```powershell
# Record an action (US-5.3)
email-obs audit record --actor alice@corp.com --action approve `
    --resource email-42 --outcome sent --model-version 2

# Review the trail, filterable by user or email
email-obs audit query
email-obs audit query --actor alice@corp.com
email-obs audit query --resource email-42 --action approve --limit 20
email-obs audit query --json
```

> On Windows, set `PYTHONIOENCODING=utf-8` if piping output so the `✔` glyph renders.

---

## 5. Testing

```powershell
.venv\Scripts\python -m pytest tests/observability -v
```

| Test file | Covers |
| :--- | :--- |
| `test_logging.py` | PII redaction (email/SSN/IP); request-id set/reset & generation; JSON log carries `request_id`, redacts message + masks sensitive `extra` keys; redaction toggle; error level recorded. |
| `test_metrics.py` | Counter/gauge/timer/observe; derived p95/max; health folding (degraded=1); bounded history; Prometheus label format; alert fire/silent/missing-metric; `evaluate_alerts`; config loading. |
| `test_audit.py` | Record & read-back; insertion order; filter by actor / resource / action; `limit` returns most recent; empty trail. |

---

## 6. Notes & follow-ups

- These are **libraries** meant to be consumed by the runtime services (the
  FastAPI API/inference servers, the worker) once those exist: call
  `configure_logging()` at startup, hold one `MetricsCollector`, expose
  `to_prometheus()` on a `/metrics` route, and write `AuditTrail` events at each
  approve/override/send decision (US-5.4).
- The audit trail's local JSONL backend is the offline/default store; a
  database-backed store belongs to the API service that owns the DB
  connection and can implement the same `record`/`query` surface.
