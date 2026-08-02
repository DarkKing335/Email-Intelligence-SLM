"""SLM Project Suitability Benchmark Script.

Evaluates a model (via Mock, Local Unsloth, or HTTP provider) against representative
email test cases to measure suitability for the Email Intelligence project.

Metrics measured:
    1. JSON Validity Rate (% outputs that parse into valid JSON matching required schema)
    2. Classification Accuracy (% matching target label)
    3. Priority Accuracy (% matching target priority)
    4. Schema Completeness (% required keys present)
    5. Average Latency (ms per request)

Usage:
    # Test using default Mock provider (offline / fast)
    python scripts/benchmark_slm.py

    # Test using local Qwen2.5/LoRA adapter (requires GPU + unsloth)
    python scripts/benchmark_slm.py --provider local --model-dir ./adapter

    # Test against HTTP inference endpoint
    python scripts/benchmark_slm.py --provider http --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

# ── Benchmark Test Dataset ────────────────────────────────────────────────────

BENCHMARK_EMAILS: list[dict[str, Any]] = [
    {
        "id": "test-001",
        "category": "action_required",
        "priority": "high",
        "subject": "[ALERT] Production Database High CPU Usage (>95%)",
        "body": "Host db-prod-01 CPU utilization has exceeded 95% for 10 minutes. Automatic failover triggered. Please investigate immediately.",
        "expected_classification": "action_required",
        "expected_priority": "high",
    },
    {
        "id": "test-002",
        "category": "respond",
        "priority": "medium",
        "subject": "Meeting Request: Q3 SLM Fine-Tuning Architecture Sync",
        "body": "Hi team, could we meet tomorrow at 14:00 to align on the Qwen2.5 LoRA fine-tuning hyperparameters? Let me know if that time works.",
        "expected_classification": "respond",
        "expected_priority": "medium",
    },
    {
        "id": "test-003",
        "category": "notification",
        "priority": "medium",
        "subject": "Training Job Completed: qwen2.5-7b-lora-r16 (Epoch 3/3)",
        "body": "Your training job qwen2.5-7b-lora-r16 completed successfully on node gpu-04. Final loss: 0.284. Artifacts saved to models/checkpoints/.",
        "expected_classification": "notification",
        "expected_priority": "medium",
    },
    {
        "id": "test-004",
        "category": "social",
        "priority": "low",
        "subject": "Gym workout schedule for Thursday evening",
        "body": "Hey Alex, are we still hitting the gym tomorrow at 18:30? Let me know if you want to focus on leg day or upper body.",
        "expected_classification": "social",
        "expected_priority": "low",
    },
    {
        "id": "test-005",
        "category": "spam",
        "priority": "low",
        "subject": "SPECIAL OFFER: 70% Off Cloud Hosting Subscription!",
        "body": "Unbeatable prices on dedicated cloud servers! Claim your discount code CLOUD70 today at http://promo-deals-discount.example.com.",
        "expected_classification": "spam",
        "expected_priority": "low",
    },
    {
        "id": "test-006",
        "category": "action_required",
        "priority": "high",
        "subject": "Assignment Due: Submit Final Project Report by Friday 23:59",
        "body": "Dear students, please submit your final SLM project report and GitHub link to the portal by Friday at 23:59. Late submissions will lose 10% per day.",
        "expected_classification": "action_required",
        "expected_priority": "high",
    },
    {
        "id": "test-007",
        "category": "respond",
        "priority": "medium",
        "subject": "Question regarding dataset cleaning pipeline US-1.3",
        "body": "Hi Casey, I noticed a small discrepancy in the PII redaction regex for phone numbers. Can you check line 45 in cleaner.py?",
        "expected_classification": "respond",
        "expected_priority": "medium",
    },
    {
        "id": "test-008",
        "category": "notification",
        "priority": "low",
        "subject": "Weekly System Patch Notes - v2.4.0 Release",
        "body": "Release notes for v2.4.0: Fixed database connection leak, upgraded PyTorch to 2.4.0, updated dependencies. No downtime expected.",
        "expected_classification": "notification",
        "expected_priority": "low",
    },
]

REQUIRED_SCHEMA_KEYS = {"classification", "priority", "summary", "entities", "recommended_action", "draft"}


def parse_json(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse JSON object from raw response string."""
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def run_benchmark(
    provider_name: str = "mock",
    model_dir: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Execute suitability benchmark suite on configured model provider."""
    logger.info(f"Starting SLM suitability benchmark using provider: '{provider_name}'")

    # ── Initialize Provider ────────────────────────────────────────────────────
    if provider_name == "mock":
        from email_inference.providers.mock import MockModelProvider
        provider = MockModelProvider(model_name="qwen2.5-7b-mock", model_version="v1.0")
    elif provider_name == "local":
        from email_inference.providers.local import LocalModelProvider
        from email_inference.settings import InferenceSettings
        settings = InferenceSettings(
            model_provider="local",
            model_dir=Path(model_dir or "./adapter"),
        )
        provider = LocalModelProvider(settings)
    elif provider_name == "http":
        from email_inference.providers.http import HttpModelProvider
        provider = HttpModelProvider(endpoint=endpoint or "http://localhost:8000")
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    if not provider.ready and provider_name != "http":
        logger.error(f"Provider '{provider_name}' is not ready. Check dependencies.")
        return {"error": "Provider not ready"}

    # ── Run Evaluation ────────────────────────────────────────────────────────
    from email_intelligence.schemas import AnalysisRequest, EmailContent

    total = len(BENCHMARK_EMAILS)
    json_valid_count = 0
    classification_correct = 0
    priority_correct = 0
    schema_complete_count = 0
    latencies_ms: list[float] = []
    details: list[dict[str, Any]] = []

    for item in BENCHMARK_EMAILS:
        req = AnalysisRequest(
            email=EmailContent(
                email_id=item["id"],
                sender="test@example.com",
                subject=item["subject"],
                body_text=item["body"],
            )
        )

        start_time = time.perf_counter()
        try:
            # Synchronous execution for benchmark
            import asyncio
            result = asyncio.run(provider.analyze(req))
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            latencies_ms.append(elapsed_ms)

            # Analyze result
            clf = result.classification
            pri = result.priority
            summary = result.summary

            # Check correctness
            # Map canonical 'fyi' / 'unsubscribe' to test labels if needed
            clf_map = {"fyi": "notification", "unsubscribe": "spam", "action": "action_required"}
            norm_clf = clf_map.get(clf, clf)

            is_clf_match = (norm_clf == item["expected_classification"]) or (clf == item["expected_classification"])
            is_pri_match = (pri == item["expected_priority"])

            if is_clf_match:
                classification_correct += 1
            if is_pri_match:
                priority_correct += 1

            json_valid_count += 1
            schema_complete_count += 1

            details.append({
                "id": item["id"],
                "subject": item["subject"][:40] + "...",
                "expected": f"{item['expected_classification']} / {item['expected_priority']}",
                "got": f"{clf} / {pri}",
                "match": is_clf_match and is_pri_match,
                "latency_ms": round(elapsed_ms, 1),
            })

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            details.append({
                "id": item["id"],
                "subject": item["subject"][:40] + "...",
                "error": str(exc),
                "latency_ms": round(elapsed_ms, 1),
            })

    # ── Calculate Aggregates ──────────────────────────────────────────────────
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    json_validity_pct = round(100.0 * json_valid_count / total, 1)
    classification_acc_pct = round(100.0 * classification_correct / total, 1)
    priority_acc_pct = round(100.0 * priority_correct / total, 1)
    schema_completeness_pct = round(100.0 * schema_complete_count / total, 1)

    # Composite Suitability Score (0-100)
    suitability_score = round(
        0.35 * classification_acc_pct +
        0.25 * priority_acc_pct +
        0.25 * json_validity_pct +
        0.15 * schema_completeness_pct,
        1,
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider_name,
        "total_test_cases": total,
        "metrics": {
            "suitability_score": suitability_score,
            "json_validity_pct": json_validity_pct,
            "classification_accuracy_pct": classification_acc_pct,
            "priority_accuracy_pct": priority_acc_pct,
            "schema_completeness_pct": schema_completeness_pct,
            "avg_latency_ms": round(avg_latency, 1),
        },
        "details": details,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SLM Suitability Benchmark Tool")
    parser.add_argument(
        "--provider",
        choices=["mock", "local", "http"],
        default="mock",
        help="Provider backend to test (default: mock)",
    )
    parser.add_argument("--model-dir", default="./adapter", help="Path to LoRA adapter for local provider")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="HTTP endpoint for http provider")
    parser.add_argument(
        "--output",
        default="reports/slm_suitability_report.json",
        help="Path to save output report JSON",
    )

    args = parser.parse_args()

    report = run_benchmark(
        provider_name=args.provider,
        model_dir=args.model_dir,
        endpoint=args.endpoint,
    )

    # ── Print Rich Terminal Output ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" SLM PROJECT SUITABILITY BENCHMARK REPORT ({args.provider.upper()})")
    print("=" * 65)

    metrics = report.get("metrics", {})
    score = metrics.get("suitability_score", 0)

    print(f" Total Test Cases:         {report.get('total_test_cases')}")
    print(f" Overall Suitability Score: {score} / 100.0")
    print("-" * 65)
    print(f"  [OK] JSON Validity Rate:     {metrics.get('json_validity_pct')}%")
    print(f"  [OK] Classification Acc:    {metrics.get('classification_accuracy_pct')}%")
    print(f"  [OK] Priority Accuracy:     {metrics.get('priority_accuracy_pct')}%")
    print(f"  [OK] Schema Completeness:   {metrics.get('schema_completeness_pct')}%")
    print(f"  [TIME] Avg Latency:         {metrics.get('avg_latency_ms')} ms")
    print("-" * 65)
    print(" Sample Results:")

    for item in report.get("details", []):
        status = "[PASS]" if item.get("match") else ("[FAIL]" if "error" not in item else "[ERROR]")
        print(f"  {status} {item['id']}: {item['subject']} ({item.get('latency_ms')} ms)")

    print("=" * 65)

    # ── Save JSON Report ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Full report written to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
