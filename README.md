# AI Email Assistant — Dataset Engineering Module (Epic 1)

This project contains the complete, production-ready Data Engineering module for the AI Email Assistant SLM fine-tuning project, implementing User Stories **US-1.1 through US-1.5**.

The module uses Clean Architecture layers to clean, normalize, redact, augment, split, and version email datasets starting from a raw CSV (such as the 1.3 GB `emails.csv` corpus) to generate training-ready datasets.

---

## Project Structure

```
e:\FPT\Email Intelligence SLM/
├── cli/
│   ├── __init__.py
│   └── main.py                     # Click CLI wrapper
├── configs/
│   ├── data_engineering.yaml       # Central pipeline config YAML
│   └── label_taxonomy.yaml         # Taxonomy classes and mapping definitions
├── data/
│   ├── raw/                        # Raw email CSV directories (ignored)
│   ├── samples/                    # Test-ready CSV and JSONL sample datasets
│   │   ├── emails_sample.csv       # 100-row generated test email CSV
│   │   └── emails_sample.jsonl     # 100-row generated test email JSONL
│   └── processed/                  # Final export targets (v0.1.0/, etc.)
├── docs/
│   └── epic1_data_engineering.md   # Architectural details & schema documentation
├── logs/                           # Pipeline log outputs (JSON formatted)
├── reports/                        # Stage execution reports
├── src/
│   └── email_data_engineering/     # Source package
│       ├── __init__.py
│       ├── pipeline.py             # Orchestrates the 8-stage pipeline
│       ├── logging_config.py       # Loguru logger config
│       ├── domain/                 # Domain logic (Models, Taxonomy validation)
│       │   ├── models.py
│       │   └── taxonomy.py
│       ├── infrastructure/         # File loaders, validators, storage writers
│       │   ├── loaders.py
│       │   ├── schema.py
│       │   └── storage.py
│       └── application/            # Stage engines (cleaner, augmentor, splitter, etc.)
│           ├── importer.py
│           ├── normalizer.py
│           ├── cleaner.py
│           ├── augmentor.py
│           ├── splitter.py
│           ├── versioner.py
│           └── exporter.py
├── tests/                          # Integration and Unit tests
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── .env.example                    # Env template
├── .gitignore                      # Git exclusion rules
├── pyproject.toml                  # Python manifest
└── README.md                       # This file
```

---

## Prerequisites

- **Python 3.12+**
- **uv** (recommended package manager) or standard **pip / virtualenv**

---

## Setup & Installation

### 1. Install Dependencies

Using `uv` (recommended):
```powershell
# Sync/install all dependencies including developer tools
uv sync
```

Or using standard pip:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

### 2. Configure Environment

Copy the environment template and edit parameters if needed:
```powershell
copy .env.example .env
```

---

## CLI Reference

Run pipeline stages or inspect builds using the Click CLI:

### 1. Run the Entire Pipeline End-to-End
Loads settings from `configs/data_engineering.yaml`, imports the specified CSV, runs cleaning/augmentation, partitions datasets, and writes version metadata.
```powershell
uv run email-data pipeline --source data/samples/emails_sample.csv --seed 42
```
Options:
- `-s, --source PATH`: Override the default raw source dataset path.
- `-v, --version STR`: Force a specific dataset build version (e.g. `v0.2.0`).
- `--seed INT`: Override the random state seed for reproducible runs.

### 2. Run Individual Stages (Debugging/Tests)
```powershell
# US-1.1: Load and validate raw file schema
uv run email-data import --source data/samples/emails_sample.csv

# US-1.2: Normalise raw labels to project taxonomy
uv run email-data normalize --source data/samples/emails_sample.csv

# US-1.3: Clean, deduplicate, and redact PII
uv run email-data clean --source data/samples/emails_sample.csv

# US-1.4: Deterministic synonym/whitespace augmentations
uv run email-data augment --source data/samples/emails_sample.csv --seed 42
```

### 3. Version Inspection & Comparison
```powershell
# US-1.5: List all processed dataset versions
uv run email-data list-versions

# US-1.5: Output full JSON manifest metadata for a version
uv run email-data show-version v0.1.0

# US-1.5: Print a semantic diff between two builds
uv run email-data compare-versions v0.1.0 v0.2.0

# View label distributions across splits
uv run email-data stats --version v0.1.0
```

---

## Testing

Execute the complete pytest suite to check unit and integration coverage:

```powershell
# Run all tests
uv run pytest

# Run with stdout prints
uv run pytest -s -v

# Generate test coverage report
uv run pytest --cov=src/email_data_engineering --cov-report=term-missing
```

---

## Documentation

Full structural documentation can be found in [docs/epic1_data_engineering.md](file:///e:/FPT/Email%20Intelligence%20SLM/docs/epic1_data_engineering.md).
