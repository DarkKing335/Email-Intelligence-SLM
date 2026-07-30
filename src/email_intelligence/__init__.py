"""Shared contracts for the Epic 3 email intelligence workflow."""

from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
    EmailContent,
)

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "DraftRequest",
    "DraftResult",
    "EmailContent",
]
