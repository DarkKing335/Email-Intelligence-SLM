"""Inference service settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: str = "mock"
    model_provider_class: str | None = None
    model_endpoint: str | None = None
    model_api_token: str | None = None
    model_timeout_seconds: float = Field(default=30.0, gt=0)
    model_name: str = "epic3-mock"
    model_version: str = "development"
    summary_max_chars: int = Field(default=320, ge=80, le=500)
