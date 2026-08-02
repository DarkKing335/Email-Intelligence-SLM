"""Local in-process model provider — loads Qwen2.5 + LoRA adapter directly.

This provider is used when you want to run the fine-tuned SLM in the same
process as the inference service, without a separate HTTP model server.

Configuration (via environment variables or InferenceSettings):
    MODEL_PROVIDER=local
    MODEL_DIR=./adapter          # path to the LoRA adapter directory

Requires GPU and:
    pip install unsloth

Inference is run with Unsloth's optimized FastLanguageModel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from email_inference.providers.base import ModelProvider
from email_inference.settings import InferenceSettings
from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
)


# ── Prompt templates (mirrors adapter/promt.py) ────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "You must output ONLY a valid JSON object."
)

_ANALYSIS_PROMPT = """Analyze the following email and extract its metadata into a strict JSON format with exactly these keys:
- "classification": You MUST choose exactly ONE label based on these strict rules:
    * "action_required": Use ONLY for system alerts, build/pipeline failures, or tasks assigned by a professor.
    * "respond": Use ONLY for personal communication, meeting requests, or collaborations that require a human reply.
    * "notification": Use for informational updates (e.g., successful training jobs, patch notes) needing no immediate action.
    * "social": Use ONLY for casual check-ins, like gym/workout schedules or diet plans.
    * "spam": Use ONLY for unsolicited promotions, sales, or discounts.

- "priority": Choose strictly based on these rules:
    * "high": Critical system failures, pipeline breaks, or urgent professor requests.
    * "medium": Normal meeting requests, training job completions, or standard notifications.
    * "low": Social chats, gym plans, game newsletters, or spam.

- "summary": (a short summary)
- "entities": object with optional keys: people, date, time, project_or_product, technology, misc_items
- "recommended_action": short string or empty string ""
- "draft": reply draft or empty string "" for spam/notification/action_required

Email to analyze:
Subject: {subject}

{body}

JSON Output:"""

_DRAFT_PROMPT = """Generate a {tone} email reply for the following email:
Subject: {subject}

{body}

Additional context: {user_context}

Write a professional reply draft only, no JSON:"""


_CLASSIFICATION_MAP = {
    "action_required": "action_required",
    "respond": "respond",
    "notification": "fyi",
    "social": "social",
    "spam": "unsubscribe",
}


def _parse_json(text: str) -> dict | None:
    """Extract and parse JSON object from model output."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Try to fix common issues
        snippet = text[start:end]
        snippet = re.sub(r",\s*}", "}", snippet)  # trailing commas
        snippet = re.sub(r",\s*]", "]", snippet)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None


class LocalModelProvider(ModelProvider):
    """In-process LoRA provider using Unsloth for fast inference.

    Loads the Qwen2.5-7B base + LoRA adapter at startup and serves
    inference requests synchronously.
    """

    def __init__(self, settings: InferenceSettings) -> None:
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._ready = False
        self._load()

    def _load(self) -> None:
        """Load the model and tokenizer using Unsloth or Hugging Face Transformers."""
        model_dir = Path(self._settings.model_dir or "adapter")
        
        # 1. Try Unsloth
        try:
            from unsloth import FastLanguageModel

            logger.info(f"Loading local model with Unsloth from: {model_dir}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            self._model = model
            self._tokenizer = tokenizer
            self._ready = True
            logger.success(f"Local Unsloth model loaded successfully from {model_dir}")
            return
        except ImportError:
            logger.info("Unsloth not installed. Falling back to Hugging Face Transformers + PEFT...")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Unsloth load failed ({exc}). Falling back to Hugging Face Transformers...")

        # 2. Fallback to standard Hugging Face Transformers + PEFT
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check for base model path in adapter_config.json if available
            adapter_config_file = model_dir / "adapter_config.json"
            base_model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
            if adapter_config_file.exists():
                try:
                    cfg_data = json.loads(adapter_config_file.read_text(encoding="utf-8"))
                    base_model_name = cfg_data.get("base_model_name_or_path", base_model_name)
                except Exception:
                    pass

            logger.info(f"Loading tokenizer from: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

            logger.info(f"Loading base model '{base_model_name}' & LoRA weights from '{model_dir}'...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
            model.eval()

            self._model = model
            self._tokenizer = tokenizer
            self._ready = True
            logger.success(f"Local Transformers + PEFT model loaded successfully on {device}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load local model from {model_dir}: {exc}")

    @property
    def ready(self) -> bool:
        return self._ready

    def _generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Run generation with the local model."""
        if not self._ready:
            raise RuntimeError("Local model is not loaded. Check logs for details.")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze an email and return structured AnalysisResult."""
        prompt = _ANALYSIS_PROMPT.format(
            subject=request.subject,
            body=request.body_text,
        )

        raw = self._generate(prompt, max_new_tokens=512, temperature=0.1)
        parsed = _parse_json(raw)

        if not parsed:
            logger.warning(f"Failed to parse JSON from model output: {raw[:200]}")
            # Return a safe fallback
            return AnalysisResult(
                classification="notification",
                priority="medium",
                classification_confidence=0.5,
                priority_confidence=0.5,
                summary=f"Email from {request.sender} regarding '{request.subject}'",
                recommended_action="",
                entities={},
            )

        # Map adapter classification labels → canonical taxonomy labels
        raw_clf = parsed.get("classification", "notification")
        classification = _CLASSIFICATION_MAP.get(raw_clf, raw_clf)

        return AnalysisResult(
            classification=classification,
            priority=parsed.get("priority", "medium"),
            classification_confidence=0.90,
            priority_confidence=0.90,
            summary=parsed.get("summary", ""),
            recommended_action=parsed.get("recommended_action", ""),
            entities=parsed.get("entities", {}),
        )

    async def generate_draft(self, request: DraftRequest) -> DraftResult:
        """Generate a reply draft for an email."""
        prompt = _DRAFT_PROMPT.format(
            tone=request.tone or "professional",
            subject=request.subject,
            body=request.body_text,
            user_context=request.user_context or "",
        )

        # For draft generation use a plain generation without JSON constraint
        messages_text = (
            f"<|im_start|>system\nYou are a professional email writer.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self._tokenizer([messages_text], return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            do_sample=True,
        )
        draft = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        return DraftResult(
            body=draft.strip(),
            tone=request.tone or "professional",
            source="local",
            model_metadata={
                "model_dir": str(self._settings.model_dir or "adapter"),
                "provider": "local",
            },
        )

    async def close(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._ready = False
            logger.info("Local model unloaded.")
