---
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/Qwen2.5-7B-Instruct-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- email-intelligence
- email-classification
- email-triage
---

# AI Email Intelligence SLM — LoRA Adapter

## Model Description

This is a **LoRA (Low-Rank Adaptation)** fine-tuned adapter for **Qwen2.5-7B-Instruct**, trained to perform structured email intelligence tasks. Given a raw email, the model extracts metadata as a strict JSON object, enabling automated triage, prioritization, and draft generation for enterprise email workflows.

**Fine-tuned by:** FPT Email Intelligence SLM Team  
**Base model:** [`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)  
**Framework:** [Unsloth](https://github.com/unslothai/unsloth) + [PEFT](https://github.com/huggingface/peft) + [TRL](https://github.com/huggingface/trl)  
**Task:** Email Classification, Priority Scoring, Entity Extraction, Draft Generation  
**License:** MIT

---

## Model Details

### Architecture

| Parameter | Value |
|:---|:---|
| Base model | `Qwen2.5-7B-Instruct` (4-bit quantized via bitsandbytes) |
| Adapter type | LoRA (Low-Rank Adaptation) |
| PEFT version | 0.19.1 |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | `CAUSAL_LM` |
| Inference mode | True (frozen base, trainable adapters) |
| Max token length | 32,768 |
| Quantization | 4-bit (bitsandbytes, NF4) |

### Chat Format

Uses **ChatML** (OpenAI-compatible) format via the `<|im_start|>` / `<|im_end|>` token pair:

```
<|im_start|>system
You are a precise data extraction assistant. You must output ONLY a valid JSON object.
<|im_end|>
<|im_start|>user
Analyze the following email and extract its metadata...
<|im_end|>
<|im_start|>assistant
{ ... JSON output ... }
<|im_end|>
```

---

## Capabilities

The model outputs a strict JSON object with the following fields:

| Field | Type | Description |
|:---|:---|:---|
| `classification` | `string` | One of: `action_required`, `respond`, `notification`, `social`, `spam` |
| `priority` | `string` | One of: `high`, `medium`, `low` |
| `summary` | `string` | Short summary of the email content |
| `entities` | `object` | Extracted entities: `people`, `date`, `time`, `project_or_product`, `technology`, `misc_items` |
| `recommended_action` | `string` | Suggested action for the recipient (empty string if no action needed) |
| `draft` | `string` | Pre-drafted reply (empty string for notifications, spam, or system alerts) |

### Classification Labels

| Label | When to use |
|:---|:---|
| `action_required` | System alerts, build/pipeline failures, urgent professor/manager tasks |
| `respond` | Personal communication, meeting requests, collaborations requiring a human reply |
| `notification` | Informational updates (training job done, patch notes) — no immediate action needed |
| `social` | Casual check-ins, gym/workout schedules, social plans |
| `spam` | Unsolicited promotions, sales, discounts |

### Priority Rules

| Priority | When to use |
|:---|:---|
| `high` | Critical system failures, pipeline breaks, urgent requests |
| `medium` | Normal meeting requests, training completions, standard notifications |
| `low` | Social chats, newsletters, game updates, spam |

---

## Usage

### Quick Start (with Unsloth)

```python
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load base model + LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./adapter",   # path to this adapter directory
    max_seq_length=32768,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def analyze_email(email_text: str) -> str:
    system_prompt = "You are a precise data extraction assistant. You must output ONLY a valid JSON object."
    user_prompt = f"""Analyze the following email and extract its metadata into a strict JSON format with exactly these keys:
- "classification": Choose ONE of: action_required, respond, notification, social, spam
- "priority": Choose ONE of: high, medium, low
- "summary": (a short summary)
- "entities": object with optional keys: people, date, time, project_or_product, technology, misc_items
- "recommended_action": short string or empty string
- "draft": reply draft or empty string for spam/notification/action_required

Email to analyze:
{email_text}

JSON Output:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Example
result = analyze_email("Hi team, the CI pipeline broke at stage 3. Please fix ASAP.")
print(result)
# Expected: {"classification": "action_required", "priority": "high", ...}
```

### Via the Inference Service (Recommended)

Use the built-in `email_inference` service (Epic 3 of this project):

```bash
# Set to HTTP provider pointing at a running model server
export MODEL_PROVIDER=http
export MODEL_ENDPOINT=http://localhost:8000

uvicorn email_inference.api:app --port 8000
```

Or run the full Docker stack:

```bash
docker compose up --build -d
```

---

## Training Details

### Training Framework

| Parameter | Value |
|:---|:---|
| Framework | Unsloth (optimized for QLoRA fine-tuning) |
| Training method | Supervised Fine-Tuning (SFT) |
| Quantization | 4-bit (NF4, bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 16 (scaling = 1.0) |
| Target modules | All attention + MLP projection matrices (7 modules) |

### Training Data

Trained on custom email datasets processed by the **Epic 1 Data Engineering Pipeline** (`email_data_engineering`):

- Emails spanning 5 classification categories
- PII-redacted and deduplicated via cosine similarity
- Augmented with synonym substitution, formatting, case, and whitespace variations
- Dataset versioned and tracked via `DatasetVersion` manifests in `data/processed/`

### Training Procedure

1. Raw email CSV loaded via `email-data pipeline` (Epic 1)
2. Output JSONL converted to ChatML instruction format with `promt.py` template
3. Fine-tuned with Unsloth's `FastLanguageModel` + TRL `SFTTrainer`
4. Adapter weights saved to this directory post-training

---

## Evaluation

### Task Description

The model is evaluated on held-out test emails (20% split from the training dataset, stratified by label).

### Metrics

| Metric | Description |
|:---|:---|
| JSON validity rate | % of outputs that are parseable as valid JSON |
| Classification accuracy | Label match accuracy on test set |
| Priority accuracy | Priority label match on test set |
| Entity F1 | Token-level F1 for extracted entity spans |

---

## Limitations & Bias

- Trained primarily on English-language emails; performance on Vietnamese or multilingual emails may be reduced.
- Classification is rule-based via prompt engineering; edge cases between `respond` and `notification` may overlap.
- The `draft` field generates short replies; complex long-form responses may need human editing.
- 4-bit quantization trades a small amount of quality for significant memory savings (~5 GB VRAM for 7B model).

---

## Integration with Email Intelligence SLM

This adapter is part of the **Email Intelligence SLM** project:

```
Epic 1: Data Engineering  →  Epic 2: Training  →  Epic 3: Inference API
                                    ↑
                          adapter/ (this model)
                                    ↓
                         Epic 4: Review Dashboard
                         Epic 5: Observability
                         Epic 6: MLOps + Deployment
```

See the full [project README](../README.md) for setup and deployment instructions.

---

## Framework Versions

- **PEFT**: 0.19.1
- **Unsloth**: Latest at training time
- **Transformers**: Compatible with Qwen2 architecture
- **Python**: 3.12+