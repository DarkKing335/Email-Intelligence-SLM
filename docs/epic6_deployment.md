# Epic 6: Deployment (US-6.5)

This document details the **Deployment** module of the AI Email Assistant
project. It provides an automated, repeatable, per-environment deployment
pipeline that consumes the **Model Registry (US-6.4)** to decide *what* to ship
and the **Health Checks (US-6.6)** to verify the system is *accessible* after
shipping.

Module location: `src/email_mlops/deployment/`.

---

## 1. Acceptance criteria mapping

| Criterion (US-6.5) | How it is met |
| :--- | :--- |
| System can be re-deployed via docs or an automated process | `email-mlops deploy --env <env>` runs the full pipeline; `--dry-run` prints the exact command for manual use. |
| System is accessible & working after deploy | The **health-gate** step polls readiness (US-6.6) until serviceable or times out. |
| Configuration separated per environment | One self-contained YAML per environment under `configs/environments/`. |
| Deployment failures are clearly surfaced | Every step is captured in a `DeploymentResult`; failures record the step, message, and captured stderr, and the CLI exits non-zero. |

---

## 2. Pipeline

```
email-mlops deploy --env staging
        │
        ▼
1. load-config     : resolve configs/environments/staging.yaml
        │             (clear error + available list if missing)
        ▼
2. resolve-models  : ask the Model Registry (US-6.4) for each model's
        │             get_deployment_target(); record name → version
        │             (fails here if require_models and none is deployable)
        ▼
3. compose-up      : docker compose -p <project> -f <files> up -d [--build]
        │             (fails here with captured stderr on non-zero exit)
        ▼
4. health-gate     : poll readiness (US-6.6) until not-UNHEALTHY or timeout
        │             (DEGRADED still passes; UNHEALTHY fails the deploy)
        ▼
   SUCCESS
```

`--dry-run` stops after step 2 and prints the resolved command and model
versions without changing anything (status `PLANNED`).

---

## 3. Per-environment configuration

Each environment is a standalone file — `configs/environments/{dev,staging,production}.yaml`:

```yaml
environment: staging
compose:
  files: ["docker-compose.yml"]
  project_name: "email-slm-staging"
  build: true                  # build images from source (false = use prebuilt)
  env:                         # env vars injected into docker compose
    LOG_LEVEL: "INFO"
    POSTGRES_DB: "email_slm_staging"
models:                        # registered adapters to resolve a version for
  - "reply-style-professional"
  - "reply-style-friendly"
  - "reply-style-concise"
require_models: true           # fail the deploy if any has no deployable version
health:                        # post-deploy readiness gate (US-6.6)
  timeout_seconds: 180
  interval_seconds: 5
  dependencies:
    postgres: {type: tcp, host: localhost, port: 5432, critical: true}
    mlflow:   {type: http, url: "http://localhost:5000/health", critical: true}
    inference:{type: http, url: "http://localhost:8000/health", critical: true}
    api:      {type: http, url: "http://localhost:8080/health", critical: true}
```

Shipped environments:

| Env | Build images? | `require_models` | Health gate |
| :--- | :--- | :--- | :--- |
| `dev` | yes | no | postgres (critical), mlflow (non-critical) |
| `staging` | yes | yes | postgres, mlflow, inference, api (all critical) |
| `production` | no (uses GHCR images) | yes | + frontend (non-critical); longer timeout |

---

## 4. CLI reference

```powershell
# List configured environments
email-mlops environments

# Preview a deployment without running it (prints the exact compose command)
email-mlops deploy --env staging --dry-run

# Deploy for real
email-mlops deploy --env staging

# Override the environment's build setting / skip the readiness gate
email-mlops deploy --env production --no-build
email-mlops deploy --env dev --skip-health

# Machine-readable result (for CI)
email-mlops deploy --env staging --json
```

`email-mlops deploy` exits non-zero when the deployment fails, so it can gate a
CI/CD job. A `DEGRADED` health result (only non-critical dependencies down)
still counts as a successful deploy.

### Example — failure is surfaced clearly

```text
production: FAILED
  ✔ load-config: loaded environment 'production'
  ✘ resolve-models: No deployable version for required model(s): [...]
Deployment failed at 'resolve-models': No deployable version for required model(s): [...]
# exit code 1
```

---

## 5. Python usage

```python
from email_mlops import Deployer

deployer = Deployer()  # uses configs/mlops.yaml + configs/environments/
result = deployer.deploy("staging", dry_run=True)

print(result.status)          # DeploymentStatus.PLANNED
print(result.model_versions)  # {'reply-style-professional': 2, ...}
print(result.command)         # ['docker', 'compose', '-p', ...]
for step in result.steps:
    print(step.name, step.ok, step.detail)
```

`Deployer` is fully injectable for testing — a fake `CommandRunner`, a local
`registry`, a `checker_builder`, and a no-op `sleep` let the whole pipeline run
without Docker (see `tests/mlops/test_deployment.py`).

---

## 6. Testing

```powershell
.venv\Scripts\python -m pytest tests/mlops/test_deployment.py -v
```

| Area | Cases |
| :--- | :--- |
| Config | load + defaults; missing environment error lists available; `list_environments`. |
| Deploy | dry-run plans without executing; successful full pipeline; compose failure surfaced with stderr; health-gate timeout; missing environment; required-model missing; `--no-build` override. |

---

## 7. Notes & follow-ups

- **Rollback** is a separate story (US-5.6) and is intentionally out of scope
  here; a failed deploy surfaces the error but does not auto-revert.
- Real `docker compose up` requires the `inference`/`api`/`frontend` service
  images, which are owned by other epics; until those exist, `--dry-run`
  exercises the full planning path, and `--skip-health` deploys only the infra
  services (postgres, mlflow).
- The health gate reuses the shared `email_mlops.health.build_checker`, the
  same probe-wiring used by `email-mlops health`.
