"""Tests for the Arq worker: runs the security lens and posts one review."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    Finding,
    LensResult,
    LensTimeoutError,
    Severity,
)
from heimdall.worker import WorkerSettings, run_review

_REPO = "owner/repo"
_PR = 3
_SHA = "sha1234"
_INSTALL_ID = 42
_APP_ID = 1
_PRIVATE_KEY = "key"


def _lens_result(findings: list[Finding]) -> LensResult:
    return LensResult(lens_name="security", findings=findings)


def _patch_review_pipeline(
    *,
    lens_result: LensResult | None = None,
    lens_side_effect: BaseException | None = None,
    last_sha: str | None = None,
) -> ExitStack:
    """Patch the worker's seed-assembly, lens run, and SHA helpers in one block.

    Returns an ExitStack-managed context manager; callers use it under ``with``.
    """
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=last_sha))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    if lens_side_effect is not None:
        run_lens_mock: AsyncMock = AsyncMock(side_effect=lens_side_effect)
    else:
        run_lens_mock = AsyncMock(return_value=lens_result)
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))
    return stack


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

    findings = [Finding(severity=Severity.LOW, title="nit", message="style", location=None)]
    with (
        _patch_review_pipeline(lens_result=_lens_result(findings)),
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
    mock_gh_client.post_review.assert_awaited_once()
    assert mock_gh_client.post_review.await_count == 1
    mock_set.assert_awaited_once_with(
        mock_db, repo_full_name=_REPO, pr_number=_PR, sha=_SHA
    )


@pytest.mark.asyncio
async def test_run_review_reflects_planted_finding_in_body() -> None:
    """A planted security finding (mocked lens output) shows up in the review body."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    findings = [
        Finding(
            severity=Severity.HIGH,
            title="SQL injection",
            message="User input concatenated into a query",
            location="app/db.py:12",
        )
    ]
    with (
        _patch_review_pipeline(lens_result=_lens_result(findings)),
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

    posted = mock_gh_client.post_review.await_args.kwargs
    assert "SQL injection" in posted["body"]
    assert "app/db.py:12" in posted["body"]


@pytest.mark.asyncio
async def test_run_review_high_finding_requests_changes() -> None:
    """A high/critical finding posts event=REQUEST_CHANGES."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    findings = [Finding(severity=Severity.HIGH, title="x", message="m", location=None)]
    with (
        _patch_review_pipeline(lens_result=_lens_result(findings)),
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

    assert mock_gh_client.post_review.await_args.kwargs["event"] == "REQUEST_CHANGES"


@pytest.mark.asyncio
async def test_run_review_no_findings_posts_comment() -> None:
    """No findings posts event=COMMENT."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(lens_result=_lens_result([])),
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

    assert mock_gh_client.post_review.await_args.kwargs["event"] == "COMMENT"


@pytest.mark.asyncio
async def test_run_review_handles_lens_failure_without_crashing() -> None:
    """A lens timeout is handled: no crash, no review posted, SHA not recorded."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(lens_side_effect=LensTimeoutError("killed")),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        # Must not raise — the worker swallows lens failures gracefully.
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.post_review.assert_not_called()
    mock_set.assert_not_called()
    mock_gh_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_skips_already_reviewed_sha() -> None:
    """Worker skips posting (and the lens) if the head SHA was already reviewed."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(lens_result=_lens_result([]), last_sha=_SHA),
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
    mock_gh_client = AsyncMock()
    mock_gh_client.post_review = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(lens_result=_lens_result([])),
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
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(lens_result=_lens_result([]), last_sha=_SHA),
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
