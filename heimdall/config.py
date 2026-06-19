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
    claude_binary: str = Field(
        default="claude", description="Path or name of the claude CLI executable"
    )
    lens_token_cap: int = Field(
        default=400_000, description="Per-agent cumulative-token cap for a lens run"
    )
    lens_timeout_seconds: float = Field(
        default=1_800.0, description="Wall-clock timeout (s) before a lens subprocess is killed"
    )
    review_timeout_seconds: float = Field(
        default=2_400.0,
        description="Per-review wall-clock timeout (s) across the whole pipeline",
    )
    debug_logging: bool = Field(
        default=False,
        description="When True, log findings and code text; default logs are metadata-only",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
