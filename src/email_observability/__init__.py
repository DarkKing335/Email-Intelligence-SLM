"""Email Observability — Safety & Observability utilities (Epic 5).

- **Logging (US-5.1):** structured logs with request tracing and redaction.
- **Monitoring (US-5.2):** in-process metrics, health folding, and alerts.
- **Audit trail (US-5.3):** append-only record of model and user actions.
"""

from email_observability.audit import AuditAction, AuditEvent, AuditTrail
from email_observability.logging import (
    configure_logging,
    get_request_id,
    redact_text,
    request_context,
)
from email_observability.metrics import (
    Alert,
    AlertRule,
    Comparison,
    MetricsCollector,
    alert_rules_from_config,
    evaluate_alerts,
)

__all__ = [
    # Logging (US-5.1)
    "configure_logging",
    "request_context",
    "get_request_id",
    "redact_text",
    # Monitoring (US-5.2)
    "MetricsCollector",
    "AlertRule",
    "Alert",
    "Comparison",
    "evaluate_alerts",
    "alert_rules_from_config",
    # Audit trail (US-5.3)
    "AuditTrail",
    "AuditEvent",
    "AuditAction",
]
