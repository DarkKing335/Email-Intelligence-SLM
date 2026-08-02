"""Provider construction from service configuration."""

from __future__ import annotations

import importlib

from email_inference.providers.base import ModelProvider, ProviderError
from email_inference.providers.http import HttpModelProvider
from email_inference.providers.mock import MockModelProvider
from email_inference.settings import InferenceSettings

# LocalModelProvider is imported lazily to avoid hard GPU dependency at import time


def create_provider(settings: InferenceSettings) -> ModelProvider:
    provider = settings.model_provider.lower().strip()
    if provider == "mock":
        return MockModelProvider(
            model_name=settings.model_name,
            model_version=settings.model_version,
            summary_max_chars=settings.summary_max_chars,
        )
    if provider == "http":
        if not settings.model_endpoint:
            raise ProviderError("MODEL_ENDPOINT is required for the HTTP provider")
        return HttpModelProvider(
            endpoint=settings.model_endpoint,
            api_token=settings.model_api_token,
            timeout_seconds=settings.model_timeout_seconds,
        )
    if provider == "local":
        from email_inference.providers.local import LocalModelProvider  # noqa: PLC0415

        return LocalModelProvider(settings)
    if provider == "custom":
        if not settings.model_provider_class or ":" not in settings.model_provider_class:
            raise ProviderError(
                "MODEL_PROVIDER_CLASS must use the 'package.module:ClassName' format"
            )
        module_name, class_name = settings.model_provider_class.split(":", maxsplit=1)
        try:
            provider_class = getattr(importlib.import_module(module_name), class_name)
            instance = provider_class(settings)
        except (ImportError, AttributeError, TypeError) as exc:
            raise ProviderError("custom model provider could not be loaded") from exc
        if not isinstance(instance, ModelProvider):
            raise ProviderError("custom model provider does not implement ModelProvider")
        return instance
    raise ProviderError(f"unsupported model provider: {provider}")
