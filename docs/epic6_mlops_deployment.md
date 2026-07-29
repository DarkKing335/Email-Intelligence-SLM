# Epic 6: MLOps & Deployment Infrastructure

This document details the infrastructure skeleton, containerization, CI/CD pipeline, and experiment tracking abstractions for **Epic 6 (MLOps & Deployment)** of the AI Email Assistant project.

---

## 1. Overview

The Epic 6 infrastructure provides a production-grade, modular foundation supporting:
- **Multi-container orchestration** using Docker & Docker Compose for training, inference, backend API, frontend dashboard, and MLflow tracking.
- **Automated CI/CD** via GitHub Actions for code linting, pytest execution, security auditing, and container image publishing to GitHub Container Registry (GHCR).
- **Backend-Agnostic Experiment Tracking** using Clean Architecture design patterns to allow seamless switching between MLflow (production/staging) and a local JSON-based tracker (offline/dev).

---

## 2. Infrastructure Architecture

```
                                  ┌─────────────────────────┐
                                  │   GitHub Actions CI/CD  │
                                  └────────────┬────────────┘
                                               │ Builds & Pushes Images
                                               ▼
                                  ┌─────────────────────────┐
                                  │ GitHub Container Reg.   │
                                  │   (ghcr.io/fpt-slm)     │
                                  └────────────┬────────────┘
                                               │ Pulls Images
                                               ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                                Docker Compose                                   │
 │                                                                                 │
 │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
 │  │    Frontend     │───>│   Backend API   │───>│      Inference Server       │  │
 │  │   (Streamlit)   │    │    (FastAPI)    │    │ (Transformers + PEFT Fast)  │  │
 │  └─────────────────┘    └────────┬────────┘    └──────────────┬──────────────┘  │
 │                                  │                            │                 │
 │                                  ▼                            ▼                 │
 │                         ┌─────────────────┐          ┌─────────────────┐        │
 │                         │   PostgreSQL    │          │  MLflow Server  │        │
 │                         │   (Database)    │          │ (Tracking / UI) │        │
 │                         └─────────────────┘          └─────────────────┘        │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Docker Infrastructure (`docker/` & `docker-compose.yml`)

### Generated Dockerfiles
- **[Dockerfile.training](file:///e:/FPT/Email%20Intelligence%20SLM/docker/Dockerfile.training)**: CUDA 12.1 + PyTorch environment pre-configured with `transformers`, `peft`, `accelerate`, and `tinker` SDK dependencies.
- **[Dockerfile.inference](file:///e:/FPT/Email%20Intelligence%20SLM/docker/Dockerfile.inference)**: Lightweight Python 3.12 slim container for running the low-latency FastAPI model serving engine. Pre-configures HuggingFace model cache directories.
- **[Dockerfile.api](file:///e:/FPT/Email%20Intelligence%20SLM/docker/Dockerfile.api)**: Backend application container with PostgreSQL client dependencies (`asyncpg`, `psycopg2`), `sqlalchemy`, and security libraries (`python-jose`, `passlib`).
- **[Dockerfile.frontend](file:///e:/FPT/Email%20Intelligence%20SLM/docker/Dockerfile.frontend)**: Streamlit container serving interactive dashboards for triage visualization, metrics review, and pipeline management.

### Docker Compose Services (`docker-compose.yml`)
The root `docker-compose.yml` ties together all five services:

| Service Name | Port | Description | Healthcheck Endpoint |
| :--- | :--- | :--- | :--- |
| `postgres` | `5432` | Relational storage for email history and audit logs | `pg_isready` |
| `mlflow` | `5000` | MLflow tracking server and model registry | `/health` |
| `inference` | `8000` | SLM inference API (FastAPI) | `/health` |
| `api` | `8080` | Core backend business logic & orchestration | `/health` |
| `frontend` | `8501` | User interface dashboard | `/_stcore/health` |

#### Commands
```powershell
# Start all services in detached mode
docker compose up --build -d

# Start only DB and MLflow tracking server
docker compose up postgres mlflow -d

# Check service status and health
docker compose ps

# View logs for a specific service
docker compose logs -f inference

# Stop and clean up containers/volumes
docker compose down -v
```

---

## 4. CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

The GitHub Actions workflow automates quality checks and deployment across 5 stages:

1. **Lint & Format**: Runs `ruff check` and `ruff format --check`.
2. **Tests**: Executes `pytest` with code coverage reports exported to XML artifacts.
3. **Security Scan**: Runs `pip-audit` to detect vulnerable Python dependencies.
4. **Docker Build & Push**: Uses Buildx matrix build for all 4 images (`training`, `inference`, `api`, `frontend`) and pushes to `ghcr.io`.
5. **Container Vulnerability Scan**: Uses Aquasecurity Trivy to scan images for CRITICAL/HIGH vulnerabilities.

---

## 5. Experiment Tracking Abstraction (`src/email_mlops/`)

The experiment tracking system follows Clean Architecture principles to keep tracking logic decoupled from specific SDK implementations.

### Abstract Base Contract (`BaseExperimentTracker`)
Defines standardized methods used by training scripts and data processing pipelines:
- `start_run(run_name, experiment_name, tags)`
- `log_parameter(key, value)` / `log_parameters(params)`
- `log_metric(key, value, step)` / `log_metrics(metrics, step)`
- `log_artifact(local_path, artifact_path)`
- `set_tags(tags)`
- `end_run(status)`

### Backends
1. **`MlflowTracker`** ([mlflow_tracker.py](file:///e:/FPT/Email%20Intelligence%20SLM/src/email_mlops/experiment_tracking/mlflow_tracker.py)): Connects to MLflow tracking server. Logs parameters, scalar metrics over steps, tags, and uploads model artifacts.
2. **`LocalTracker`** ([local_tracker.py](file:///e:/FPT/Email%20Intelligence%20SLM/src/email_mlops/experiment_tracking/local_tracker.py)): Fallback tracker that persists experiment metrics, parameters, tags, and artifact metadata to structured `.json` files in `reports/experiments/`. Ideal for offline development and local CI tests.

### Factory Selector (`create_tracker`)
Instantiates the appropriate tracker based on resolution order:
1. `backend_override` parameter passed to function.
2. `TRACKING_BACKEND` environment variable (`"mlflow"` or `"local"`).
3. `tracking.backend` setting in `configs/mlops.yaml`.
4. Default fallback to `"local"`.

#### Python Usage Example
```python
from email_mlops import create_tracker

# Instantiate tracker using configs/mlops.yaml or env vars
tracker = create_tracker()

# Start an experiment run
tracker.start_run(run_name="qwen-4b-lora-r16", experiment_name="slm-finetuning")

# Log parameters & metrics
tracker.log_parameters({"learning_rate": 2e-4, "lora_r": 16, "batch_size": 8})
for epoch in range(1, 5):
    tracker.log_metric("train_loss", 1.5 / epoch, step=epoch)

# Record tags and log model artifacts
tracker.set_tags({"framework": "peft", "base_model": "Qwen3-4B"})
tracker.log_artifact("models/adapter_config.json", artifact_path="lora_adapter")

# End run
tracker.end_run(status="FINISHED")
```

---

## 6. Extending the Infrastructure

When adding new modules (e.g. model training, inference handlers, or database schemas):
- **Training scripts**: Import `create_tracker` from `email_mlops` to log hyperparameters and loss metrics automatically.
- **Inference service**: Place FastAPI route handlers in `src/email_inference/` and update `docker/Dockerfile.inference` as new model dependencies are introduced.
- **Backend API**: Add ORM models in `src/email_api/` and manage migrations via Alembic.
