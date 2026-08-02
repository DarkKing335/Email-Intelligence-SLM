# AI Email Intelligence SLM

A production-grade platform that fine-tunes a Small Language Model (SLM) to intelligently triage, classify, and draft replies to emails — built on **Qwen2.5-7B-Instruct** with a **LoRA adapter**, served via a FastAPI inference gateway, reviewed through a Streamlit dashboard, and orchestrated with a full MLOps stack.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Email Intelligence SLM                             │
│                                                                             │
│  Epic 1              Epic 2               Epic 3           Epic 4 / 5       │
│  ─────────           ──────────           ───────────       ────────────     │
│  Data                SLM                 Email             Review           │
│  Engineering    →    Fine-Tuning    →    Intelligence  →   Dashboard        │
│  Pipeline            (QLoRA)             API               + Observability  │
│                                                                             │
│                       Epic 6: MLOps & Deployment                            │
│           (Experiment Tracking · Model Registry · Health Checks · CI/CD)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Service Map

| Service | Port | Description |
|:---|:---|:---|
| `inference` | 8000 | FastAPI — SLM gateway (mock / http / local provider) |
| `api` | 8080 | FastAPI — Email workflow, PostgreSQL, review queue, audit |
| `frontend` | 8501 | Streamlit — Review dashboard (Epic 4 profile) |
| `mlflow` | 5000 | MLflow tracking server |
| `postgres` | 5432 | PostgreSQL database |

---

## Quick Start (Docker Compose)

```powershell
# 1. Clone and configure
cp .env.example .env        # edit secrets if needed

# 2. Start all core services (API + Inference + MLflow + Postgres)
docker compose up --build -d

# 3. Start the dashboard (separate Epic 4 profile)
docker compose --profile epic4 up frontend -d

# 4. Verify all services are healthy
docker compose ps
```

Open:
- **Dashboard**: http://localhost:8501
- **Backend API docs**: http://localhost:8080/docs
- **Inference API docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## Quick Start (Local Development)

### Prerequisites

- **Python 3.12+**
- **pip** or **uv** (recommended)
- **Git LFS** (for data files)

### 1. Install Dependencies

```powershell
# Using uv (recommended)
uv sync

# Or pip
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,frontend]"
```

### 2. Configure Environment

```powershell
copy .env.example .env
# Edit .env to set DATABASE_URL, INFERENCE_URL, etc.
```

### 3. Run Services Locally (two terminals)

```powershell
# Terminal 1: Inference service (mock model by default)
python -m uvicorn email_inference.api:app --port 8000

# Terminal 2: Backend API (SQLite by default for local dev)
python -m uvicorn email_api.main:app --port 8080
```

### 4. Run Dashboard

```powershell
$env:API_URL = "http://localhost:8080"
streamlit run src/email_dashboard/app.py
```

---

## Project Structure

```
Email Intelligence SLM/
├── adapter/                    # Trained LoRA adapter (Qwen2.5-7B)
│   ├── adapter_config.json     # LoRA config (r=16, alpha=16)
│   ├── tokenizer.json          # Qwen2.5 tokenizer
│   ├── chat_template.jinja     # ChatML chat format template
│   ├── promt.py                # Prompt format reference
│   └── README.md               # Model card
│
├── src/
│   ├── email_data_engineering/ # Epic 1: 8-stage data pipeline
│   ├── email_training/         # Epic 2: LoRA fine-tuning pipeline
│   ├── email_intelligence/     # Epic 3: Shared schemas (AnalysisResult, etc.)
│   ├── email_inference/        # Epic 3: FastAPI inference gateway
│   │   └── providers/          #   mock | http | local (Unsloth)
│   ├── email_api/              # Epic 3: Email workflow + review API
│   ├── email_dashboard/        # Epic 4: Streamlit review dashboard
│   ├── email_observability/    # Epic 5: Logging, Metrics, Audit Trail
│   └── email_mlops/            # Epic 6: Experiment tracking, Model Registry,
│       ├── experiment_tracking/ #          Health checks, Deployer
│       ├── model_registry/
│       ├── health/
│       └── deployment/
│
├── cli/
│   ├── main.py                 # email-data CLI (Epic 1)
│   ├── training.py             # email-train CLI (Epic 2)
│   ├── mlops.py                # email-mlops CLI (Epic 6)
│   └── observability.py        # email-obs CLI (Epic 5)
│
├── configs/
│   ├── data_engineering.yaml   # Epic 1 pipeline config
│   ├── mlops.yaml              # Epic 6 MLOps config
│   ├── observability.yaml      # Epic 5 logging/metrics config
│   └── environments/           # Per-environment deploy configs
│
├── docker/
│   ├── Dockerfile.training     # CUDA 12.1 + PyTorch for training
│   ├── Dockerfile.inference    # FastAPI inference service
│   ├── Dockerfile.api          # FastAPI backend service
│   └── Dockerfile.frontend     # Streamlit dashboard
│
├── docker-compose.yml          # Full stack orchestration
├── .github/workflows/ci-cd.yml # GitHub Actions CI/CD
├── pyproject.toml              # Python package manifest
└── docs/                       # Epic documentation
    ├── epic1_data_engineering.md
    ├── epic2_training.md
    ├── epic3_email_intelligence.md
    ├── epic4_dashboard.md
    ├── epic5_observability.md
    ├── epic6_deployment.md
    ├── epic6_registry_health.md
    └── epic6_mlops_deployment.md
```

---

## CLI Reference

### Epic 1: Data Engineering (`email-data`)

```powershell
# Run the full 8-stage data pipeline
email-data pipeline --source data/samples/emails_sample.csv --seed 42

# Individual stages
email-data import --source data/samples/emails_sample.csv
email-data normalize --source data/samples/emails_sample.csv
email-data clean --source data/samples/emails_sample.csv
email-data augment --source data/samples/emails_sample.csv --seed 42

# Version management
email-data list-versions
email-data show-version v0.1.0
email-data compare-versions v0.1.0 v0.2.0
email-data stats --version v0.1.0
```

### Epic 2: Training & Benchmarking (`email-train` & `benchmark_slm.py`)

```powershell
# Prepare data from Epic 1 output
email-train prepare --version v0.1.0

# Run LoRA fine-tuning (requires GPU + pip install unsloth trl datasets)
email-train run --epochs 3 --lora-r 16

# Evaluate trained adapter
email-train evaluate --adapter models/checkpoints/email-intelligence-adapter

# Run SLM Project Suitability Benchmark suite (Mock, Local, or HTTP)
python scripts/benchmark_slm.py --provider mock
python scripts/benchmark_slm.py --provider local --model-dir ./adapter
python scripts/benchmark_slm.py --provider http --endpoint http://localhost:8000

# Register to model registry (Epic 6)
email-train export models/checkpoints/email-intelligence-adapter \
    --name email-intelligence-adapter --dataset-version v0.1.0 --promote
```

### Epic 5: Observability (`email-obs`)

```powershell
# Record an audit event
email-obs audit record --actor alice@corp.com --action approve --resource email-42

# Query audit trail
email-obs audit query --actor alice@corp.com
email-obs audit query --resource email-42 --limit 20 --json
```

### Epic 6: MLOps (`email-mlops`)

```powershell
# Model registry
email-mlops register-model email-intelligence-adapter --adapter-path models/adapters/v1
email-mlops list-models
email-mlops promote-model email-intelligence-adapter 1 --stage production

# Health checks
email-mlops health
email-mlops health --json

# Deployment
email-mlops environments
email-mlops deploy --env staging --dry-run
email-mlops deploy --env staging
```

---

## Testing

```powershell
# Run all tests (108 pass, 1 skipped — live integration requires running server)
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=term-missing

# Run specific epic tests
python -m pytest tests/unit/ -v                        # Epic 1
python -m pytest tests/epic3/ tests/dashboard/ -v      # Epic 3 + 4
python -m pytest tests/observability/ -v               # Epic 5
python -m pytest tests/mlops/ -v                       # Epic 6
```

---

## Configuration

| Variable | Default | Description |
|:---|:---|:---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./email_slm.db` | Backend API database |
| `INFERENCE_URL` | `http://localhost:8000` | Inference service URL |
| `MODEL_PROVIDER` | `mock` | `mock` / `http` / `local` |
| `MODEL_ENDPOINT` | – | HTTP inference endpoint (for `http` provider) |
| `MODEL_DIR` | `./adapter` | Path to LoRA adapter (for `local` provider) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `POSTGRES_USER` | `slm_user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `slm_secret` | PostgreSQL password |
| `POSTGRES_DB` | `email_slm` | PostgreSQL database name |

---

## Documentation

| Document | Description |
|:---|:---|
| [Epic 1: Data Engineering](docs/epic1_data_engineering.md) | 8-stage pipeline, schema, CLI reference |
| [Epic 2: Training Pipeline](docs/epic2_training.md) | LoRA fine-tuning, data prep, evaluation |
| [Epic 3: Email Intelligence](docs/epic3_email_intelligence.md) | Inference gateway, API workflow |
| [Epic 4/5: Dashboard & Observability](docs/epic4_dashboard.md) | Review UI, logging, metrics, audit |
| [Epic 5: Safety & Observability](docs/epic5_observability.md) | Logging, metrics, audit trail details |
| [Epic 6: Model Registry & Health](docs/epic6_registry_health.md) | Model versioning, health probes |
| [Epic 6: Deployment](docs/epic6_deployment.md) | Automated multi-environment deployment |
| [Epic 6: MLOps Infrastructure](docs/epic6_mlops_deployment.md) | Docker, CI/CD, experiment tracking |
| [Adapter Model Card](adapter/README.md) | LoRA adapter details, usage, training info |

---

## License

MIT — FPT Email Intelligence SLM Team
