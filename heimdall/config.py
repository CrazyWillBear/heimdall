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
    claude_env_passthrough: list[str] = Field(
        default_factory=list,
        description="Extra env-var names forwarded to the claude child beyond the "
        "PATH/HOME/ANTHROPIC_API_KEY allowlist (e.g. HTTPS_PROXY, NODE_EXTRA_CA_CERTS)",
    )
    bwrap_binary: str = Field(
        default="bwrap",
        description="Path or name of the bubblewrap (bwrap) executable used to sandbox "
        "each lens claude subprocess; resolved on PATH unless an absolute path is given",
    )
    sandbox_extra_read_only_binds: list[str] = Field(
        default_factory=list,
        description="Extra host paths bound read-only into the lens sandbox, for "
        "nonstandard claude/node/CA installs (the seed, OS, CA, DNS, ~/.claude, and venv "
        "are bound automatically; the worker project dir is never bound)",
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

    # extra="ignore": the .env is SHARED with docker-compose (its header says so) and so
    # carries compose/Caddy-only keys (e.g. DOMAIN) that are not Settings fields.  The
    # dotenv source reads every key in the file, so the pydantic-settings default of
    # extra="forbid" would reject the shared .env outright; ignore unknown keys instead.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
