"""Model provider interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
)


class ProviderError(RuntimeError):
    """A safe, provider-independent inference failure."""


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def ready(self) -> bool: ...

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...

    async def generate_draft(self, request: DraftRequest) -> DraftResult: ...

    async def close(self) -> None: ...
