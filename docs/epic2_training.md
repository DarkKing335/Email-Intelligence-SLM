# Epic 2: SLM Fine-Tuning Pipeline

This document details the **training pipeline** for fine-tuning the Qwen2.5-7B-Instruct base model using LoRA (QLoRA) on the email dataset produced by Epic 1.

Module location: `src/email_training/`

---

## 1. Overview

| Story | Description |
|:---|:---|
| **US-2.1 Data Preparation** | Convert Epic 1 JSONL output to ChatML SFT instruction format |
| **US-2.2 LoRA Fine-Tuning** | Fine-tune with Unsloth + PEFT + TRL on GPU |
| **US-2.3 Evaluation** | Measure JSON validity, classification accuracy, priority accuracy |
| **US-2.4 Registry Export** | Register trained adapter to Epic 6 MLOps model registry |

The trained adapter is stored in `adapter/` and used by the `email_inference` service (Epic 3) via the `LocalModelProvider` (`MODEL_PROVIDER=local`).

---

## 2. Architecture

```
Epic 1 Output (data/processed/<version>/)
        │
        ▼ email-train prepare
data/training/{train,val,test}.jsonl   ← ChatML instruction format
        │
        ▼ email-train run
models/checkpoints/<adapter-name>/    ← Saved LoRA adapter
        │
        ├── email-train evaluate → reports/evaluation_report.json
        │
        └── email-train export → MLOps Model Registry (Epic 6)
```

---

## 3. Model Architecture

| Parameter | Value |
|:---|:---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Adapter type | QLoRA (4-bit quantized base + BF16 LoRA adapters) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 (scaling ratio = 1.0) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all 7) |
| Max sequence length | 2,048 tokens |
| PEFT version | 0.19.1 |

---

## 4. Data Preparation (US-2.1)

### Input Format (Epic 1 EmailRecord JSONL)
```json
{"email_id": "abc-123", "subject": "CI pipeline broken", "body_text": "...", "label": "security", "priority": "high", "split": "train"}
```

### Output Format (ChatML instruction pair)
```json
{
  "conversations": [
    {"role": "system", "content": "You are a precise data extraction assistant..."},
    {"role": "user", "content": "Analyze the following email...\n\nEmail to analyze:\nSubject: CI pipeline broken\n\n..."},
    {"role": "assistant", "content": "{\"classification\": \"action_required\", \"priority\": \"high\", ...}"}
  ]
}
```

### Classification Label Mapping

| Epic 1 Label | SLM Label |
|:---|:---|
| `security` | `action_required` |
| `action` | `action_required` |
| `meeting` | `respond` |
| `fyi` | `notification` |
| `newsletter` | `notification` |
| `promotion` | `spam` |
| `unsubscribe` | `spam` |
| `social` | `social` |

---

## 5. Training Details (US-2.2)

### Hardware Requirements

| Minimum | Recommended |
|:---|:---|
| 1× GPU, 16 GB VRAM | 1× A100 40 GB or better |
| Python 3.12+ | CUDA 12.1+ |

### Extra Dependencies (GPU-only)

```bash
pip install unsloth trl datasets
```

These are **not included** in the default project `pyproject.toml` to keep the CI lightweight (tests run without GPU).

### Default Hyperparameters

| Hyperparameter | Value |
|:---|:---|
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 5% |
| Weight decay | 0.01 |
| Batch size per GPU | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| BF16 precision | True |
| Gradient checkpointing | Unsloth optimized |

---

## 6. CLI Reference

```powershell
# Step 1: Prepare data from Epic 1 output
email-train prepare --version v0.1.0

# Optional: limit to 500 training samples for quick testing
email-train prepare --version v0.1.0 --max-samples 500

# Step 2: Run training
email-train run

# Customize hyperparameters
email-train run --epochs 5 --learning-rate 1e-4 --lora-r 32

# Step 3: Evaluate
email-train evaluate --adapter models/checkpoints/email-intelligence-adapter

# Step 4: Register to model registry (Epic 6)
email-train export models/checkpoints/email-intelligence-adapter \
    --name email-intelligence-adapter \
    --dataset-version v0.1.0 \
    --promote
```

---

## 7. Prompt Template

The adapter is fine-tuned to extract structured JSON from emails. The prompt format matches `adapter/promt.py`:

```
<|im_start|>system
You are a precise data extraction assistant. You must output ONLY a valid JSON object.
<|im_end|>
<|im_start|>user
Analyze the following email and extract its metadata...

Email to analyze:
Subject: {subject}

{body}

JSON Output:
<|im_end|>
<|im_start|>assistant
{"classification": "...", "priority": "...", ...}
<|im_end|>
```

---

## 8. Output Schema

The model must output a JSON object with these fields:

```json
{
  "classification": "action_required | respond | notification | social | spam",
  "priority": "high | medium | low",
  "summary": "Short summary of the email",
  "entities": {
    "people": ["Alice", "Bob"],
    "date": "2026-08-01",
    "time": "14:00",
    "project_or_product": ["Email Intelligence SLM"],
    "technology": ["Docker", "PostgreSQL"],
    "misc_items": ["50% discount"]
  },
  "recommended_action": "Schedule a meeting | Ignore the offer. | ...",
  "draft": "Dear ..., Thank you for ..."
}
```

---

## 9. Integration with Inference Service (Epic 3)

After training, set the inference service to use the local adapter:

```bash
# Via environment variables
export MODEL_PROVIDER=local
export MODEL_DIR=./models/checkpoints/email-intelligence-adapter

uvicorn email_inference.api:app --port 8000
```

Or via Docker Compose (update `.env`):
```
MODEL_PROVIDER=local
MODEL_DIR=/app/models/email-intelligence-adapter
```

The `LocalModelProvider` in `src/email_inference/providers/local.py` handles:
- Loading the Qwen2.5 base + LoRA adapter via Unsloth
- Mapping adapter classification labels → Epic 3 canonical taxonomy labels
- Returning `AnalysisResult` and `DraftResult` in the standard schema

---

## 10. MLOps Integration (Epic 6)

After evaluating the adapter, register it to the model registry:

```python
from email_training.registry_export import export_to_registry

version = export_to_registry(
    adapter_path=Path("models/checkpoints/email-intelligence-adapter"),
    model_name="email-intelligence-adapter",
    dataset_version="v0.1.0",
    source_run_id="<experiment-run-id>",
    promote_to_staging=True,
)
```

The deployment pipeline (`email-mlops deploy`) will then pick up this version via `get_deployment_target()`.
