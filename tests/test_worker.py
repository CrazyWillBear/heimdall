"""Tests for the Arq worker: fans out three lenses, synthesizes, posts one review."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    Finding,
    LensResult,
    LensTimeoutError,
    Severity,
    SynthesisResult,
    TaggedFinding,
)
from heimdall.worker import WorkerSettings, run_review

_REPO = "owner/repo"
_PR = 3
_SHA = "sha1234"
_INSTALL_ID = 42
_APP_ID = 1
_PRIVATE_KEY = "key"


def _lens_result(findings: list[Finding], *, name: str = "security") -> LensResult:
    return LensResult(lens_name=name, findings=findings)


def _synthesis_from(tagged: list[TaggedFinding]) -> SynthesisResult:
    """Build a SynthesisResult whose verdict/body derive from the tagged survivors."""
    from heimdall.lens import format_synthesis_body, verdict_for_tagged

    return SynthesisResult(
        tagged_findings=tagged,
        verdict=verdict_for_tagged(tagged),
        body=format_synthesis_body(tagged),
    )


def _patch_review_pipeline(
    *,
    lens_results: list[LensResult] | None = None,
    lens_side_effect: BaseException | None = None,
    synthesis_result: SynthesisResult | None = None,
    synthesis_side_effect: BaseException | None = None,
    last_sha: str | None = None,
) -> tuple[ExitStack, AsyncMock]:
    """Patch the worker's seed-assembly, lens run, synthesis, and SHA helpers.

    Returns the ExitStack (used under ``with``) plus the run_synthesis mock so a
    test can assert what reached synthesis.
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
    elif lens_results is not None:
        run_lens_mock = AsyncMock(side_effect=list(lens_results))
    else:
        run_lens_mock = AsyncMock(return_value=_lens_result([]))
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))

    if synthesis_side_effect is not None:
        synth_mock = AsyncMock(side_effect=synthesis_side_effect)
    else:
        synth_mock = AsyncMock(return_value=synthesis_result or _synthesis_from([]))
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    return stack, synth_mock


# ---------------------------------------------------------------------------
# run_review: ctx contract — builds GitHubClient per job from app credentials
# ---------------------------------------------------------------------------


def _tagged(severity: Severity, lens: str, title: str = "t", message: str = "m") -> TaggedFinding:
    return TaggedFinding(
        lens=lens,
        finding=Finding(severity=severity, title=title, message=message, location=None),
    )


@pytest.mark.asyncio
async def test_run_review_posts_exactly_one_review() -> None:
    """Worker builds a per-job GitHubClient and posts exactly one synthesized review."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()
    mock_gh_client.post_review = AsyncMock()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    synthesis = _synthesis_from([_tagged(Severity.LOW, "cleanliness", "nit")])
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
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
async def test_run_review_fans_out_three_lenses_into_synthesis() -> None:
    """All three lenses run and their results reach synthesis (mocked claude)."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    lens_results = [
        _lens_result([Finding(Severity.HIGH, "SecretLeak", "m", None)], name="security"),
        _lens_result([Finding(Severity.MEDIUM, "GodObject", "m", None)], name="design"),
        _lens_result([Finding(Severity.LOW, "DeadCode", "m", None)], name="cleanliness"),
    ]
    stack, synth_mock = _patch_review_pipeline(
        lens_results=lens_results,
        synthesis_result=_synthesis_from([_tagged(Severity.HIGH, "security", "SecretLeak")]),
    )
    with (
        stack,
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

    # Synthesis received every lens's result.
    synth_mock.assert_awaited_once()
    assert synth_mock.await_args is not None
    passed: list[LensResult] = synth_mock.await_args.kwargs["lens_results"]
    names = {r.lens_name for r in passed}
    assert names == {"security", "design", "cleanliness"}


@pytest.mark.asyncio
async def test_run_review_body_is_severity_grouped_and_lens_tagged() -> None:
    """The posted body groups synthesized findings by severity, each tagged by lens."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    synthesis = _synthesis_from(
        [
            _tagged(Severity.CRITICAL, "security", "RCE", "eval"),
            _tagged(Severity.LOW, "cleanliness", "Naming nit", "rename"),
        ]
    )
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
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

    body = mock_gh_client.post_review.await_args.kwargs["body"]
    assert body.index("CRITICAL") < body.index("LOW")
    assert "security" in body.lower()
    assert "cleanliness" in body.lower()
    assert "RCE" in body


@pytest.mark.asyncio
async def test_run_review_verdict_reflects_highest_surviving_severity() -> None:
    """A high/critical survivor of synthesis posts event=REQUEST_CHANGES."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    synthesis = _synthesis_from([_tagged(Severity.HIGH, "design", "x")])
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
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
async def test_run_review_no_surviving_findings_posts_comment() -> None:
    """When synthesis keeps only medium/low (or nothing), the event is COMMENT."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]))[0],
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
async def test_run_review_isolates_one_lens_failure() -> None:
    """One failing lens does not crash the pipeline; synthesis still runs on survivors."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # Security succeeds, Design times out, Cleanliness succeeds.
    lens_outcomes: list[Any] = [
        _lens_result([Finding(Severity.HIGH, "SecretLeak", "m", None)], name="security"),
        LensTimeoutError("design killed"),
        _lens_result([Finding(Severity.LOW, "DeadCode", "m", None)], name="cleanliness"),
    ]
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(
        patch("heimdall.worker.run_lens", new=AsyncMock(side_effect=lens_outcomes))
    )
    synth_mock = AsyncMock(
        return_value=_synthesis_from([_tagged(Severity.HIGH, "security", "SecretLeak")])
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    with (
        stack,
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

    # Synthesis ran on the two surviving lenses; one review still posted.
    synth_mock.assert_awaited_once()
    assert synth_mock.await_args is not None
    surviving = {r.lens_name for r in synth_mock.await_args.kwargs["lens_results"]}
    assert surviving == {"security", "cleanliness"}
    mock_gh_client.post_review.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_handles_all_lenses_failing_without_crashing() -> None:
    """If every lens fails, no review is posted, no SHA recorded, no crash."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    stack, synth_mock = _patch_review_pipeline(lens_side_effect=LensTimeoutError("killed"))
    with (
        stack,
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

    synth_mock.assert_not_called()
    mock_gh_client.post_review.assert_not_called()
    mock_set.assert_not_called()
    mock_gh_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_handles_synthesis_failure_without_crashing() -> None:
    """A synthesis abort is handled: no crash, no review posted, SHA not recorded."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    lens_results = [
        _lens_result([Finding(Severity.HIGH, "x", "m", None)], name="security"),
        _lens_result([], name="design"),
        _lens_result([], name="cleanliness"),
    ]
    stack, _ = _patch_review_pipeline(
        lens_results=lens_results,
        synthesis_side_effect=LensTimeoutError("synthesis killed"),
    )
    with (
        stack,
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
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
    mock_set.assert_not_called()
    mock_gh_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_skips_already_reviewed_sha() -> None:
    """Worker skips posting (and the lenses) if the head SHA was already reviewed."""
    mock_gh_client = AsyncMock()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(last_sha=_SHA)[0],
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
        _patch_review_pipeline(synthesis_result=_synthesis_from([]))[0],
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
        _patch_review_pipeline(last_sha=_SHA)[0],
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
