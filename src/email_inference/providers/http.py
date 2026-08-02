"""Adapter for a remote server implementing the Epic 3 inference contract."""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from email_inference.providers.base import ProviderError
from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
)


class HttpModelProvider:
    def __init__(
        self,
        endpoint: str,
        api_token: str | None = None,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._client = httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    @property
    def ready(self) -> bool:
        return True

    async def close(self) -> None:
        await self._client.aclose()

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        return await self._post("/v1/analyze", request, AnalysisResult)

    async def generate_draft(self, request: DraftRequest) -> DraftResult:
        return await self._post("/v1/generate-draft", request, DraftResult)

    async def _post(self, path: str, request: object, result_type: type):
        try:
            response = await self._client.post(
                path,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            return result_type.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise ProviderError(f"model provider request failed: {type(exc).__name__}") from exc
