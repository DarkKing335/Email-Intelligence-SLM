"""Backend API configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./email_intelligence.db"
    inference_url: str = "http://localhost:8000"
    inference_timeout_seconds: float = Field(default=35.0, gt=0)
    auto_create_schema: bool = True
    default_draft_tone: str = "professional"
