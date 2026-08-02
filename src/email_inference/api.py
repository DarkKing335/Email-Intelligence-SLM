"""FastAPI entry point for the provider-neutral inference gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from email_inference.providers import ModelProvider, ProviderError, create_provider
from email_inference.settings import InferenceSettings
from email_intelligence.schemas import (
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
)


def create_app(
    provider: ModelProvider | None = None,
    settings: InferenceSettings | None = None,
) -> FastAPI:
    resolved_settings = settings or InferenceSettings()
    resolved_provider = provider or create_provider(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await resolved_provider.close()

    app = FastAPI(title="Email Intelligence Inference", version="1.0.0", lifespan=lifespan)
    app.state.provider = resolved_provider

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok" if resolved_provider.ready else "not_ready",
            "provider": resolved_settings.model_provider,
        }

    @app.post("/v1/analyze", response_model=AnalysisResult)
    async def analyze(request: AnalysisRequest) -> AnalysisResult:
        try:
            return await resolved_provider.analyze(request)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/generate-draft", response_model=DraftResult)
    async def generate_draft(request: DraftRequest) -> DraftResult:
        try:
            return await resolved_provider.generate_draft(request)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
