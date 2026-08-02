# Epic 6: Model Registry & Health Checks

This document details the **Model Registry (US-6.4)** and **Health Check (US-6.6)**
modules of the AI Email Assistant project. Both extend the existing
`src/email_mlops/` package and reuse the backend-agnostic factory pattern
established by the Experiment Tracking module (US-6.3).

---

## 1. Overview

| Story | Module | Purpose |
| :--- | :--- | :--- |
| **US-6.4 Model Registry** | `src/email_mlops/model_registry/` | Register approved model artifacts (the LoRA reply-style adapters / checkpoints), store their metadata + lineage, hold multiple versions, and select one version to deploy. |
| **US-6.6 Health Check** | `src/email_mlops/health/` | Provide reusable liveness / readiness checks and dependency probes for every service, usable behind a `/health` endpoint, from the CLI, and in CI. |

Both ship a `local` backend that runs fully offline (dev / CI) and integrate
with the same `configs/mlops.yaml` used by the tracker. A single CLI,
`email-mlops`, drives both modules.

---

## 2. Model Registry (US-6.4)

### 2.1 Architecture

The registry mirrors the Experiment Tracking layout: an abstract contract, two
interchangeable backends, and a config-driven factory.

```
                       ┌─────────────────────────┐
                       │   create_registry()     │  (factory — reads mlops.yaml)
                       └────────────┬────────────┘
                                    │ returns
                    ┌───────────────┴────────────────┐
                    ▼                                 ▼
        ┌───────────────────────┐        ┌───────────────────────┐
        │  LocalModelRegistry   │        │  MlflowModelRegistry  │
        │  (JSON files on disk) │        │  (MLflow Model Reg.)  │
        └───────────┬───────────┘        └───────────┬───────────┘
                    └──────────────┬─────────────────┘
                                   ▼
                       ┌───────────────────────┐
                       │   BaseModelRegistry   │  (ABC — the contract)
                       │   + ModelVersion      │  (pydantic data model)
                       │   + RegistryStage     │  (lifecycle enum)
                       └───────────────────────┘
```

### 2.2 Data model — `ModelVersion`

Every registered version records its **artifact location**, **lineage**, and
**metadata**:

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `str` | Registered model name, e.g. `reply-style-professional`. |
| `version` | `int` | Monotonic version number (starts at 1). |
| `stage` | `RegistryStage` | `none` \| `staging` \| `production` \| `archived`. |
| `adapter_path` | `str` | Path / URI of the adapter or checkpoint artifact. |
| `source_run_id` | `str \| None` | **Lineage:** experiment-tracking run that produced it. |
| `base_model` | `str \| None` | **Lineage:** base model, e.g. `Qwen3-4B`. |
| `dataset_version` | `str \| None` | **Lineage:** dataset build used for training, e.g. `v0.1.0`. |
| `metrics` | `dict[str, float]` | Evaluation metrics snapshot at registration. |
| `tags` | `dict[str, str]` | Arbitrary key-value metadata. |
| `description` | `str \| None` | Human-readable notes. |
| `created_at` / `updated_at` | `str` | UTC ISO-8601 timestamps. |

### 2.3 Lifecycle & stages

```
register → version 1 (stage=none)
register → version 2 (stage=none)
promote v2 → production
      │
      └── any existing PRODUCTION version is automatically ARCHIVED
          (at most one production version per model at a time)

get_deployment_target(name)
      └── returns the PRODUCTION version, else the latest STAGING version, else None
```

`get_deployment_target()` is the single entry point deployment tooling (US-6.5)
calls to decide what to ship.

### 2.4 Backends

1. **`LocalModelRegistry`** — one JSON file per model under `root_dir`
   (default `models/registry/`). Atomic writes; works offline; ideal for dev
   and CI.
2. **`MlflowModelRegistry`** — wraps `mlflow.tracking.MlflowClient`. Our
   lineage/metric fields are stored as MLflow model-version *tags* so they
   round-trip faithfully. MLflow natively archives the incumbent production
   version on promotion.

### 2.5 Configuration (`configs/mlops.yaml`)

```yaml
model_registry:
  backend: "local"            # local | mlflow
  local:
    root_dir: "models/registry"
    pretty_print: true
  mlflow:
    tracking_uri: "http://localhost:5000"
```

Backend resolution order (highest priority first):
1. `backend_override` argument to `create_registry()`.
2. `MODEL_REGISTRY_BACKEND` environment variable.
3. `model_registry.backend` in `configs/mlops.yaml`.
4. Default: `"local"`.

### 2.6 Python usage

```python
from email_mlops import create_registry, RegistryStage

registry = create_registry()  # reads configs/mlops.yaml

# Register an approved adapter with full lineage
registry.register_model(
    "reply-style-professional",
    adapter_path="models/adapters/professional-r16",
    source_run_id="qwen3-4b-lora-run-007",
    base_model="Qwen3-4B",
    dataset_version="v0.2.0",
    metrics={"val_loss": 0.33},
)

# Promote it — the previous production version is archived automatically
registry.transition_stage("reply-style-professional", 1, RegistryStage.PRODUCTION)

# Ask what deployment (US-6.5) should ship
target = registry.get_deployment_target("reply-style-professional")
print(target.version, target.adapter_path)
```

---

## 3. Health Check (US-6.6)

### 3.1 Liveness vs. readiness

| Check | Question | Touches dependencies? | Use |
| :--- | :--- | :--- | :--- |
| **Liveness** | Is *this* process up and responsive? | No | Restart decisions — a healthy process isn't killed just because a dependency is slow. |
| **Readiness** | Can the service actually serve traffic right now? | Yes (runs probes) | Load-balancer / deploy gating. |

### 3.2 Aggregation rule

`check_readiness()` runs every registered `Probe` and aggregates:

| Condition | Overall status | HTTP | CLI exit |
| :--- | :--- | :--- | :--- |
| All probes pass | `HEALTHY` | 200 | 0 |
| Only **non-critical** probe(s) fail | `DEGRADED` | 503 | 0 |
| Any **critical** probe fails | `UNHEALTHY` | 503 | 1 |

`DEGRADED` still counts as serviceable (`HealthReport.ok is True`), so a
non-essential dependency being down won't fail a CI gate; a critical failure
returns a non-zero exit code that stops a deploy.

### 3.3 Components

- **`HealthChecker`** — collects probes; exposes `check_liveness()` and
  `check_readiness()`, returning a `HealthReport`.
- **`HealthReport`** — `status`, `service`, `checked_at`, per-check results;
  `.ok`, `.http_status`, and `.to_dict()` helpers.
- **`Probe`** — a named check (`name`, `check` callable → `(ok, detail)`,
  `critical`). A probe that raises is treated as a failure, never crashing the
  checker.
- **Probe factories** (`probes.py`, stdlib-only, no extra deps):
  `tcp_probe`, `http_probe`, `mlflow_probe`, `postgres_probe`.

### 3.4 Configuration (`configs/mlops.yaml`)

```yaml
health:
  service: "email-intelligence-slm"
  dependencies:
    mlflow:
      type: http
      url: "http://localhost:5000/health"
      critical: true
    postgres:
      type: tcp
      host: "localhost"
      port: 5432
      critical: true
    inference:
      type: http
      url: "http://localhost:8000/health"
      critical: false
```

Supported `type` values: `http`, `tcp`, `mlflow`, `postgres`.

### 3.5 Python usage (e.g. inside a FastAPI service)

```python
from email_mlops import HealthChecker
from email_mlops.health import http_probe, postgres_probe

checker = (
    HealthChecker(service="inference")
    .add_probe(postgres_probe("localhost", 5432, critical=True))
    .add_probe(http_probe("mlflow", "http://localhost:5000/health", critical=False))
)

# In a FastAPI route:
#   report = checker.check_readiness()
#   return JSONResponse(report.to_dict(), status_code=report.http_status)
```

The Dockerfiles for the `inference` and `api` services already declare
image-level `HEALTHCHECK` directives that curl their `/health` endpoints, and
`docker-compose.yml` gates dependent services on `condition: service_healthy`.

---

## 4. CLI Reference (`email-mlops`)

Installed via `[project.scripts]` (`email-mlops = "cli.mlops:cli"`). Pass a
custom config with `-c/--config` (defaults to `configs/mlops.yaml`).

> On Windows, set `PYTHONIOENCODING=utf-8` if you pipe output, so the `✔`/`✘`
> glyphs render.

### 4.1 Model Registry commands

```powershell
# Register an approved adapter (US-6.4)
email-mlops register-model reply-style-professional `
    --adapter-path models/adapters/professional-r16 `
    --run-id qwen3-4b-lora-run-007 `
    --base-model Qwen3-4B `
    --dataset-version v0.2.0 `
    --metric val_loss=0.33 --metric rouge_l=0.71 `
    --description "Professional-tone reply adapter"

# List registered models and their latest version
email-mlops list-models

# Inspect all versions of one model (stages, lineage)
email-mlops list-versions reply-style-professional

# Promote a version (archives the incumbent production version)
email-mlops promote-model reply-style-professional 2 --stage production

# Show which version deployment tooling would ship
email-mlops deployment-target reply-style-professional
```

### 4.2 Health commands

```powershell
# Liveness — process only
email-mlops health --live

# Readiness — probes all configured dependencies
email-mlops health

# Machine-readable output (as a /health endpoint would return)
email-mlops health --json

# Probe only a subset of configured dependencies
email-mlops health --deps mlflow,postgres
```

`email-mlops health` exits non-zero **only** when the service is `UNHEALTHY`,
so it can gate CI and deployment scripts while allowing `DEGRADED` to pass.

---

## 5. Testing

The modules are covered by 29 unit tests, all runnable offline (no MLflow
server or database required — the MLflow backend is exercised with mocks).

```powershell
# One-time environment setup
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Run the registry + health suites
.venv\Scripts\python -m pytest tests/mlops/test_model_registry.py tests/mlops/test_health.py -v

# Run the full MLOps suite with coverage (matches CI)
.venv\Scripts\python -m pytest tests/mlops --cov=src/email_mlops --cov-report=term-missing
```

| Test file | Covers |
| :--- | :--- |
| `tests/mlops/test_model_registry.py` | Version increment & persistence, stage transitions, production auto-archiving, `get_deployment_target` resolution, name validation, factory backend selection, MLflow lineage↔tag mapping (mocked). |
| `tests/mlops/test_health.py` | Liveness always healthy; readiness aggregation (healthy / degraded / unhealthy); probe exceptions handled; latency recorded; real TCP and HTTP probes (open/closed port, expected/unexpected status, connection refused). |

---

## 6. Notes & Follow-ups

- **Packaging:** `pyproject.toml`'s wheel `packages` now includes
  `src/email_mlops`, so `pip install -e .` (used by CI and all Dockerfiles)
  exposes the package. This was previously missing.
- Integrate `create_registry()` into the training pipeline (Epic 2) so approved
  adapters are registered automatically at the end of a run, and into the
  deployment step (US-6.5) so it deploys `get_deployment_target()`.
- Compose the shared `HealthChecker` into the `inference` and `api` FastAPI
  services' `/health` routes once those services exist.
