"""Service ``Settings`` (env-based config) loading behaviour."""

from __future__ import annotations

from pathlib import Path

from heimdall.config import Settings


def test_settings_ignores_compose_only_keys_in_env_file(tmp_path: Path) -> None:
    """Settings loads its own fields and ignores compose/Caddy-only keys in the .env.

    The .env is shared with docker-compose (its header says so), so it carries keys
    like ``DOMAIN`` that are not Settings fields.  With pydantic-settings' default
    ``extra="forbid"`` the dotenv source would reject the whole file; Settings must
    ignore the unknown keys instead of failing with extra_forbidden.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WEBHOOK_SECRET=s3cret\n"
        "GITHUB_APP_ID=4242\n"
        "GITHUB_APP_PRIVATE_KEY=dummy-key\n"
        "DOMAIN=heimdall.example.com\n"  # compose/Caddy-only; not a Settings field
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.webhook_secret == "s3cret"
    assert settings.github_app_id == 4242
    assert not hasattr(settings, "domain")
