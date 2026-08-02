"""Client abstraction used by the workflow to call the inference gateway."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
from pydantic import ValidationError

from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
)


class InferenceError(RuntimeError):
    pass


@runtime_checkable
class InferenceGateway(Protocol):
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...

    async def generate_draft(self, request: DraftRequest) -> DraftResult: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


class HttpInferenceGateway:
    def __init__(self, base_url: str, timeout_seconds: float = 35.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.is_success
        except httpx.HTTPError:
            return False

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        return await self._post("/v1/analyze", request, AnalysisResult)

    async def generate_draft(self, request: DraftRequest) -> DraftResult:
        return await self._post("/v1/generate-draft", request, DraftResult)

    async def _post(self, path: str, request: object, result_type: type):
        try:
            response = await self._client.post(path, json=request.model_dump(mode="json"))
            response.raise_for_status()
            return result_type.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise InferenceError(f"inference request failed: {type(exc).__name__}") from exc
