"""Tests for the Arq worker: posts exactly one review, updates last SHA."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.worker import WorkerSettings, run_review

_REPO = "owner/repo"
_PR = 3
_SHA = "sha1234"
_INSTALL_ID = 42
_APP_ID = 1
_PRIVATE_KEY = "key"


# ---------------------------------------------------------------------------
# run_review: ctx contract — builds GitHubClient per job from app credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_posts_exactly_one_review() -> None:
    """Worker builds a per-job GitHubClient and posts exactly one review."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()
    mock_gh_client.post_review = AsyncMock()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client) as mock_cls,
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_cls.assert_called_once_with(
        app_id=_APP_ID,
        private_key=_PRIVATE_KEY,
        installation_id=_INSTALL_ID,
    )
    mock_gh_client.post_review.assert_awaited_once_with(
        repo_full_name=_REPO,
        pr_number=_PR,
        commit_id=_SHA,
        body="Heimdall received this PR",
        event="COMMENT",
    )
    mock_set.assert_awaited_once_with(
        mock_db, repo_full_name=_REPO, pr_number=_PR, sha=_SHA
    )


@pytest.mark.asyncio
async def test_run_review_skips_already_reviewed_sha() -> None:
    """Worker skips posting if the head SHA was already reviewed."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=_SHA)),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_closes_github_client_after_posting() -> None:
    """run_review closes the GitHubClient after posting a review (no FD leak)."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()
    mock_gh_client.post_review = AsyncMock()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_closes_github_client_on_skip_path() -> None:
    """run_review closes the GitHubClient even when the review is skipped."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=_SHA)),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# WorkerSettings: registration and Redis wiring
# ---------------------------------------------------------------------------


def test_worker_settings_registers_run_review() -> None:
    """WorkerSettings.functions must include run_review."""
    assert run_review in WorkerSettings.functions


def test_worker_settings_has_redis_settings() -> None:
    """WorkerSettings.redis_settings must be an ArqRedisSettings instance."""
    from arq.connections import RedisSettings

    assert isinstance(WorkerSettings.redis_settings, RedisSettings)


# ---------------------------------------------------------------------------
# WorkerSettings.on_startup / on_shutdown lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_startup_populates_ctx() -> None:
    """on_startup stores db, app_id, and private_key in ctx."""
    ctx: dict[str, object] = {}
    mock_db = AsyncMock()

    with (
        patch("heimdall.worker.Database", return_value=mock_db),
        patch("heimdall.worker.settings") as mock_settings,
    ):
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.github_app_id = _APP_ID
        mock_settings.github_app_private_key = _PRIVATE_KEY

        await WorkerSettings.on_startup(ctx)

    mock_db.initialize.assert_awaited_once()
    assert ctx["db"] is mock_db
    assert ctx["app_id"] == _APP_ID
    assert ctx["private_key"] == _PRIVATE_KEY


@pytest.mark.asyncio
async def test_on_shutdown_closes_db() -> None:
    """on_shutdown closes the Database stored in ctx."""
    mock_db = AsyncMock()
    ctx: dict[str, object] = {"db": mock_db}

    await WorkerSettings.on_shutdown(ctx)

    mock_db.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_startup_strips_sqlalchemy_prefix() -> None:
    """on_startup converts SQLAlchemy DSN to plain aiosqlite path for Database."""
    ctx: dict[str, object] = {}
    mock_db = MagicMock()
    mock_db.initialize = AsyncMock()

    with (
        patch("heimdall.worker.Database", return_value=mock_db) as mock_cls,
        patch("heimdall.worker.settings") as mock_settings,
    ):
        mock_settings.database_url = "sqlite+aiosqlite:///./heimdall.db"
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.github_app_id = _APP_ID
        mock_settings.github_app_private_key = _PRIVATE_KEY

        await WorkerSettings.on_startup(ctx)

    # Database must receive a plain filesystem path, not a SQLAlchemy DSN
    call_args = mock_cls.call_args
    db_path: str = call_args[0][0] if call_args[0] else call_args[1].get("path", "")
    assert not db_path.startswith("sqlite+aiosqlite"), (
        f"Database received raw SQLAlchemy DSN: {db_path!r}"
    )
