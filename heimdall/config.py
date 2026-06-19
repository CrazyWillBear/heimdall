"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Heimdall runtime configuration.

    All values can be set via environment variables (upper-cased field names).
    Secrets should live in a .env file or injected by the deployment environment.
    """

    webhook_secret: str = Field(..., description="GitHub webhook HMAC secret")
    github_app_id: int = Field(..., description="GitHub App numeric ID")
    github_app_private_key: str = Field(..., description="PEM-encoded RSA private key")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    database_url: str = Field(
        default="sqlite+aiosqlite:///./heimdall.db", description="SQLite DB URL"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
