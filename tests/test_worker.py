"""Tests for the Arq worker: fans out three lenses, synthesizes, posts one review."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    Finding,
    LensError,
    LensResult,
    LensTimeoutError,
    Severity,
    SynthesisResult,
    TaggedFinding,
)
from heimdall.repo_config import LensConfig, RepoConfig, ScopeFilters
from heimdall.worker import WorkerSettings, run_review

_REPO = "owner/repo"
_PR = 3
_SHA = "sha1234"
_INSTALL_ID = 42
_APP_ID = 1
_PRIVATE_KEY = "key"

# Distinguishes "caller did not pass config" (default permissive gate) from an
# explicit config=None (simulate an opt-out / no-config skip).
_SENTINEL = object()


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


def _patch_guardrails(stack: ExitStack) -> None:
    """Stub the issue-#10 guardrail DB helpers so the pipeline proceeds by default.

    The rate/budget check is forced under-limit and the concurrency acquire is
    forced to succeed, so tests not specifically about guardrails behave as before.
    Tests that exercise a cap patch these names themselves.
    """
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True))
    )
    stack.enter_context(
        patch("heimdall.worker.try_record_review_event", new=AsyncMock())
    )
    stack.enter_context(
        patch("heimdall.worker.release_inflight", new=AsyncMock())
    )


def _patch_review_pipeline(
    *,
    lens_results: list[LensResult] | None = None,
    lens_side_effect: BaseException | None = None,
    synthesis_result: SynthesisResult | None = None,
    synthesis_side_effect: BaseException | None = None,
    last_sha: str | None = None,
    prior_review: dict[str, object] | None = None,
    config: object | None = _SENTINEL,
) -> tuple[ExitStack, AsyncMock]:
    """Patch the worker's gate, seed-assembly, lens run, synthesis, and SHA helpers.

    By default the early gate is stubbed to return a permissive RepoConfig so the
    pipeline proceeds; pass ``config=None`` to simulate an opt-out/no-config skip,
    or a specific RepoConfig to exercise tuning.  Also patches the across-push
    persistence helpers (``get_posted_review`` so a prior-review record can be
    injected via ``prior_review``).

    Returns the ExitStack (used under ``with``) plus the run_synthesis mock so a
    test can assert what reached synthesis.
    """
    stack = ExitStack()
    gate_config = RepoConfig() if config is _SENTINEL else config
    stack.enter_context(
        patch("heimdall.worker._gate_review", new=AsyncMock(return_value=gate_config))
    )
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=last_sha))
    )
    stack.enter_context(
        patch(
            "heimdall.worker.get_posted_review",
            new=AsyncMock(return_value=prior_review),
        )
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


def _gh_client(
    *, review_id: int = 1, node_id: str = "NODE", diff: str = ""
) -> AsyncMock:
    """Return a mock GitHubClient whose post_review yields an id and node_id.

    ``get_pr_diff`` returns ``diff`` (empty by default, i.e. no commentable lines so
    every finding falls back to the body) and ``list_review_comments`` returns no
    prior inline comments, so the post step's inline-comment split is exercised
    without requiring each test to wire the diff API.
    """
    client = AsyncMock()
    client.post_review = AsyncMock(return_value={"id": review_id, "node_id": node_id})
    client.get_pr_diff = AsyncMock(return_value=diff)
    client.list_review_comments = AsyncMock(return_value=[])
    client.delete_review_comment = AsyncMock()
    return client


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
    mock_gh_client = _gh_client()

    ctx: dict[str, object] = {
        "db": mock_db,
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
    }

    synthesis = _synthesis_from([_tagged(Severity.LOW, "cleanliness", "nit")])
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    mock_gh_client = _gh_client()
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
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    mock_gh_client = _gh_client()
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
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    synthesis = _synthesis_from([_tagged(Severity.HIGH, "design", "x")])
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]))[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # Security succeeds, Design times out, Cleanliness succeeds.
    lens_outcomes: list[Any] = [
        _lens_result([Finding(Severity.HIGH, "SecretLeak", "m", None)], name="security"),
        LensTimeoutError("design killed"),
        _lens_result([Finding(Severity.LOW, "DeadCode", "m", None)], name="cleanliness"),
    ]
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._gate_review", new=AsyncMock(return_value=RepoConfig()))
    )
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
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
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    """If every lens fails twice (after retry), a terse COMMENT note is posted, SHA recorded."""
    mock_db = AsyncMock()
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": mock_db, "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    stack, synth_mock = _patch_review_pipeline(lens_side_effect=LensTimeoutError("killed"))
    with (
        stack,
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
        patch("heimdall.worker.set_posted_review", new=AsyncMock()) as mock_persist,
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

    # Every lens failed, so synthesis never ran; after retry-once also fails a
    # single terse COMMENT note is posted, the SHA is recorded so the failed
    # commit is not endlessly re-reviewed, and no posted-review record is kept.
    synth_mock.assert_not_called()
    mock_gh_client.post_review.assert_awaited_once()
    assert mock_gh_client.post_review.await_args.kwargs["event"] == "COMMENT"
    mock_set.assert_awaited_once_with(
        mock_db, repo_full_name=_REPO, pr_number=_PR, sha=_SHA
    )
    mock_persist.assert_not_called()
    mock_gh_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_handles_synthesis_failure_without_crashing() -> None:
    """A persistent synthesis abort posts a terse COMMENT note and records the SHA."""
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    lens_results = [
        _lens_result([Finding(Severity.HIGH, "x", "m", None)], name="security"),
        _lens_result([], name="design"),
        _lens_result([], name="cleanliness"),
    ]
    # The retry re-runs the lenses, so feed enough results for two attempts.
    stack, _ = _patch_review_pipeline(
        lens_results=lens_results * 2,
        synthesis_side_effect=LensTimeoutError("synthesis killed"),
    )
    with (
        stack,
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
        patch("heimdall.worker.set_posted_review", new=AsyncMock()) as mock_persist,
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.post_review.assert_awaited_once()
    assert mock_gh_client.post_review.await_args.kwargs["event"] == "COMMENT"
    mock_set.assert_awaited_once()
    mock_persist.assert_not_called()
    mock_gh_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Retry-once + per-review timeout + terse failure note
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_retries_lens_exactly_once_then_posts_terse_note() -> None:
    """A failing pipeline is retried exactly once, then posts a terse error COMMENT."""
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # The whole lens-fanout + synthesis pipeline aborts on both attempts; the retry
    # seam wraps _synthesize_review, so drive the failure there.
    synthesize_mock = AsyncMock(side_effect=LensError("boom"))
    with (
        patch("heimdall.worker._gate_review", new=AsyncMock(return_value=RepoConfig())),
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False)),
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True)),
        patch("heimdall.worker.try_record_review_event", new=AsyncMock()),
        patch("heimdall.worker.release_inflight", new=AsyncMock()),
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker._synthesize_review", new=synthesize_mock),
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

    # Exactly two attempts: the initial run plus a single retry.
    assert synthesize_mock.await_count == 2
    # Terse failure note posted exactly once as a COMMENT, never REQUEST_CHANGES.
    mock_gh_client.post_review.assert_awaited_once()
    posted = mock_gh_client.post_review.await_args.kwargs
    assert posted["event"] == "COMMENT"
    assert "failed" in posted["body"].lower()
    # The failed SHA is recorded so it is not endlessly re-reviewed.
    mock_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_retry_succeeds_posts_real_review() -> None:
    """If the retry succeeds, the real review is posted (no failure note)."""
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # First attempt aborts, second attempt synthesizes a clean review.
    synthesis = _synthesis_from([_tagged(Severity.LOW, "cleanliness", "nit", "style")])
    synthesize_mock = AsyncMock(side_effect=[LensError("boom"), synthesis])
    with (
        patch("heimdall.worker._gate_review", new=AsyncMock(return_value=RepoConfig())),
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False)),
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True)),
        patch("heimdall.worker.try_record_review_event", new=AsyncMock()),
        patch("heimdall.worker.release_inflight", new=AsyncMock()),
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None)),
        patch("heimdall.worker._synthesize_review", new=synthesize_mock),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    assert synthesize_mock.await_count == 2
    mock_gh_client.post_review.assert_awaited_once()
    posted = mock_gh_client.post_review.await_args.kwargs
    # A real verdict, not the terse failure note.
    assert "failed" not in posted["body"].lower()


@pytest.mark.asyncio
async def test_run_review_pipeline_timeout_surfaced_as_failure() -> None:
    """A run exceeding the per-review wall-clock timeout posts a terse failure note."""
    mock_gh_client = AsyncMock()
    # Tiny per-review timeout; the lens sleeps past it on every attempt.
    ctx: dict[str, object] = {
        "db": AsyncMock(),
        "app_id": _APP_ID,
        "private_key": _PRIVATE_KEY,
        "review_timeout_seconds": 0.01,
    }

    async def _slow_lens(*_args: object, **_kwargs: object) -> object:
        import asyncio

        await asyncio.sleep(1.0)
        return _lens_result([])

    run_lens_mock = AsyncMock(side_effect=_slow_lens)
    with (
        patch("heimdall.worker._gate_review", new=AsyncMock(return_value=RepoConfig())),
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False)),
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True)),
        patch("heimdall.worker.try_record_review_event", new=AsyncMock()),
        patch("heimdall.worker.release_inflight", new=AsyncMock()),
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock())),
        patch("heimdall.worker.run_lens", new=run_lens_mock),
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

    # The timeout is surfaced as a failure: terse COMMENT note after retry.
    mock_gh_client.post_review.assert_awaited_once()
    posted = mock_gh_client.post_review.await_args.kwargs
    assert posted["event"] == "COMMENT"
    assert "failed" in posted["body"].lower()


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
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]))[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
# run_review: across-push review lifecycle (dismiss / minimize prior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_review_dismisses_prior_request_changes() -> None:
    """A prior REQUEST_CHANGES review is dismissed before the fresh one is posted."""
    mock_gh_client = _gh_client(review_id=2, node_id="NODE2")
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    prior = {"review_id": 1, "node_id": "NODE1", "verdict": "REQUEST_CHANGES"}
    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]), prior_review=prior)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.dismiss_review.assert_awaited_once()
    assert mock_gh_client.dismiss_review.await_args.kwargs["review_id"] == 1
    mock_gh_client.minimize_review.assert_not_called()
    mock_gh_client.post_review.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_minimizes_prior_comment() -> None:
    """A prior COMMENT review is minimized (not dismissed) before posting the fresh one."""
    mock_gh_client = _gh_client(review_id=2, node_id="NODE2")
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    prior = {"review_id": 1, "node_id": "NODE1", "verdict": "COMMENT"}
    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]), prior_review=prior)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.minimize_review.assert_awaited_once_with(node_id="NODE1")
    mock_gh_client.dismiss_review.assert_not_called()
    mock_gh_client.post_review.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_persists_new_review_after_posting() -> None:
    """run_review stores the fresh review's id, node id, and verdict after posting."""
    mock_db = AsyncMock()
    mock_gh_client = _gh_client(review_id=99, node_id="NODE99")
    ctx: dict[str, object] = {"db": mock_db, "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    synthesis = _synthesis_from([_tagged(Severity.HIGH, "security", "x")])
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()) as mock_persist,
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_persist.assert_awaited_once_with(
        mock_db,
        repo_full_name=_REPO,
        pr_number=_PR,
        review_id=99,
        node_id="NODE99",
        verdict="REQUEST_CHANGES",
    )


@pytest.mark.asyncio
async def test_run_review_first_review_does_not_refresh() -> None:
    """With no prior review on record, nothing is dismissed or minimized."""
    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    with (
        _patch_review_pipeline(synthesis_result=_synthesis_from([]), prior_review=None)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.dismiss_review.assert_not_called()
    mock_gh_client.minimize_review.assert_not_called()
    mock_gh_client.post_review.assert_awaited_once()


# ---------------------------------------------------------------------------
# Inline comments: anchor findings on changed lines; off-diff falls back to body;
# a new push deletes prior inline comments then posts the fresh set.
# ---------------------------------------------------------------------------

_INLINE_DIFF = """\
diff --git a/app/db.py b/app/db.py
index 1111111..2222222 100644
--- a/app/db.py
+++ b/app/db.py
@@ -10,3 +10,4 @@ def q():
     conn = get()
+    cur.execute("SELECT * FROM t WHERE id=" + id)
     return cur
"""


@pytest.mark.asyncio
async def test_run_review_anchors_changed_line_finding_as_inline_comment() -> None:
    """A finding on a changed line is posted as an inline comment at that path:line."""
    mock_gh_client = _gh_client(diff=_INLINE_DIFF)
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # The added line in the diff is new-file line 11; anchor the finding there.
    synthesis = _synthesis_from(
        [
            TaggedFinding(
                lens="security",
                finding=Finding(Severity.HIGH, "SQLi", "raw concat", "app/db.py:11"),
            )
        ]
    )
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    comments = posted["comments"]
    assert len(comments) == 1
    assert comments[0]["path"] == "app/db.py"
    assert comments[0]["line"] == 11
    assert comments[0]["side"] == "RIGHT"
    assert "SQLi" in comments[0]["body"]
    # The anchored finding lives inline, not in the body.
    assert "SQLi" not in posted["body"]


@pytest.mark.asyncio
async def test_run_review_off_diff_finding_falls_back_to_body() -> None:
    """A finding on a line not present in the diff is rendered in the body, not inline."""
    mock_gh_client = _gh_client(diff=_INLINE_DIFF)
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    # Line 999 is not in the diff -> body fallback.
    synthesis = _synthesis_from(
        [
            TaggedFinding(
                lens="design",
                finding=Finding(Severity.MEDIUM, "OffDiff", "elsewhere", "app/db.py:999"),
            )
        ]
    )
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    assert posted["comments"] == []
    assert "OffDiff" in posted["body"]


@pytest.mark.asyncio
async def test_run_review_unparseable_location_falls_back_to_body() -> None:
    """A finding with no/unparseable location is rendered in the body, never inline."""
    mock_gh_client = _gh_client(diff=_INLINE_DIFF)
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    synthesis = _synthesis_from(
        [
            TaggedFinding(
                lens="cleanliness",
                finding=Finding(Severity.LOW, "NoLoc", "no location", None),
            )
        ]
    )
    with (
        _patch_review_pipeline(synthesis_result=synthesis)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
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
    assert posted["comments"] == []
    assert "NoLoc" in posted["body"]


@pytest.mark.asyncio
async def test_run_review_push_deletes_prior_inline_comments_then_posts_fresh() -> None:
    """A new push lists+deletes the prior review's inline comments before posting fresh."""
    mock_gh_client = _gh_client(review_id=2, node_id="NODE2", diff=_INLINE_DIFF)
    mock_gh_client.list_review_comments = AsyncMock(
        return_value=[{"id": 101}, {"id": 102}]
    )
    ctx: dict[str, object] = {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY}

    prior = {"review_id": 1, "node_id": "NODE1", "verdict": "COMMENT"}
    synthesis = _synthesis_from(
        [
            TaggedFinding(
                lens="security",
                finding=Finding(Severity.HIGH, "SQLi", "raw concat", "app/db.py:11"),
            )
        ]
    )
    with (
        _patch_review_pipeline(synthesis_result=synthesis, prior_review=prior)[0],
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.set_posted_review", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    # Prior inline comments listed by the prior review id and each deleted.
    mock_gh_client.list_review_comments.assert_awaited_once()
    assert mock_gh_client.list_review_comments.await_args.kwargs["review_id"] == 1
    deleted = {
        call.kwargs["comment_id"]
        for call in mock_gh_client.delete_review_comment.await_args_list
    }
    assert deleted == {101, 102}
    # Then the fresh review (with its inline comment) is posted exactly once.
    mock_gh_client.post_review.assert_awaited_once()
    assert len(mock_gh_client.post_review.await_args.kwargs["comments"]) == 1


# ---------------------------------------------------------------------------
# Metadata-only logging: no token/secret/findings by default; findings under DEBUG
# ---------------------------------------------------------------------------

_TOKEN = "ghs_supersecretinstallationtoken"
_SECRET_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIsecret\n-----END RSA PRIVATE KEY-----"
_API_KEY = "sk-ant-supersecretanthropickey"
_FINDING_TITLE = "SQL injection via unsanitized id"
_FINDING_MESSAGE = "User input concatenated into a raw query string"


async def _drive_review_capturing_logs(
    *,
    debug_logging: bool,
) -> str:
    """Run a successful review under the given debug flag; return the full log text."""
    import logging

    mock_gh_client = _gh_client()
    ctx: dict[str, object] = {
        "db": AsyncMock(),
        # Secrets the worker handles but must never log.
        "app_id": _APP_ID,
        "private_key": _SECRET_KEY,
        "installation_token": _TOKEN,
        "anthropic_api_key": _API_KEY,
        "debug_logging": debug_logging,
    }
    # A synthesized survivor whose title/message must only surface under DEBUG.
    synthesis = _synthesis_from(
        [
            TaggedFinding(
                lens="security",
                finding=Finding(
                    severity=Severity.HIGH,
                    title=_FINDING_TITLE,
                    message=_FINDING_MESSAGE,
                    location="app/db.py:12",
                ),
            )
        ]
    )

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    worker_logger = logging.getLogger("heimdall.worker")
    prev_level = worker_logger.level
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.DEBUG)
    try:
        with (
            _patch_review_pipeline(synthesis_result=synthesis)[0],
            patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
            patch("heimdall.worker.set_posted_review", new=AsyncMock()),
            patch("heimdall.worker.GitHubClient", return_value=mock_gh_client),
        ):
            await run_review(
                ctx,
                installation_id=_INSTALL_ID,
                repo_full_name=_REPO,
                pr_number=_PR,
                head_sha=_SHA,
            )
    finally:
        worker_logger.removeHandler(handler)
        worker_logger.setLevel(prev_level)

    return "\n".join(record.getMessage() for record in records)


@pytest.mark.asyncio
async def test_default_logs_contain_only_metadata_no_secrets_or_findings() -> None:
    """Default logs carry metadata only — no token, secret, or findings text."""
    log_text = await _drive_review_capturing_logs(debug_logging=False)

    # Metadata is present (repo, PR, SHA, verdict).
    assert _REPO in log_text
    assert _SHA in log_text

    # No secret material ever appears.
    assert _TOKEN not in log_text
    assert _SECRET_KEY not in log_text
    assert _API_KEY not in log_text
    assert "BEGIN RSA PRIVATE KEY" not in log_text

    # No findings/code text in default (metadata-only) logs.
    assert _FINDING_TITLE not in log_text
    assert _FINDING_MESSAGE not in log_text


@pytest.mark.asyncio
async def test_debug_logs_include_findings_and_code() -> None:
    """Under the DEBUG flag, findings/code text appears in the logs."""
    log_text = await _drive_review_capturing_logs(debug_logging=True)

    # Findings/code text appears only because DEBUG logging is enabled.
    assert _FINDING_TITLE in log_text
    assert _FINDING_MESSAGE in log_text

    # Even under DEBUG, secrets are never logged.
    assert _TOKEN not in log_text
    assert _SECRET_KEY not in log_text
    assert _API_KEY not in log_text


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
        patch("heimdall.worker.sandbox_exec_probe", new=AsyncMock()),
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
async def test_on_startup_runs_sandbox_probe_with_configured_bwrap() -> None:
    """on_startup runs the sandbox exec-probe with the configured bwrap binary."""
    ctx: dict[str, object] = {}
    probe = AsyncMock()

    with (
        patch("heimdall.worker.sandbox_exec_probe", new=probe),
        patch("heimdall.worker.Database", return_value=AsyncMock()),
        patch("heimdall.worker.settings") as mock_settings,
    ):
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.github_app_id = _APP_ID
        mock_settings.github_app_private_key = _PRIVATE_KEY
        mock_settings.bwrap_binary = "/custom/bwrap"

        await WorkerSettings.on_startup(ctx)

    probe.assert_awaited_once_with("/custom/bwrap")


@pytest.mark.asyncio
async def test_on_startup_aborts_when_sandbox_probe_fails() -> None:
    """A failing sandbox exec-probe aborts startup; the DB is never opened."""
    from heimdall.lens import SandboxError

    ctx: dict[str, object] = {}
    db_cls = MagicMock()

    with (
        patch(
            "heimdall.worker.sandbox_exec_probe",
            new=AsyncMock(side_effect=SandboxError("bwrap cannot run here")),
        ),
        patch("heimdall.worker.Database", new=db_cls),
        patch("heimdall.worker.settings") as mock_settings,
    ):
        mock_settings.database_url = "sqlite+aiosqlite:///./test.db"
        mock_settings.redis_url = "redis://localhost:6379"
        mock_settings.github_app_id = _APP_ID
        mock_settings.github_app_private_key = _PRIVATE_KEY
        mock_settings.bwrap_binary = "bwrap"

        with pytest.raises(SandboxError):
            await WorkerSettings.on_startup(ctx)

    db_cls.assert_not_called()
    assert "db" not in ctx


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
        patch("heimdall.worker.sandbox_exec_probe", new=AsyncMock()),
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


# ---------------------------------------------------------------------------
# Issue #8 — config gating (opt-in + scope) and lens tuning through run_review
# ---------------------------------------------------------------------------


def _gating_gh_client(
    *,
    config_yaml: str | None,
    draft: bool = False,
    user_type: str = "User",
    association: str = "OWNER",
    head_repo: str = "owner/repo",
    base_ref: str = "main",
    labels: list[str] | None = None,
    changed_paths: list[str] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    """Mock GitHubClient for the real _gate_review path (get_pr/file/files).

    ``files`` overrides the PR-files payload outright (used to drive the diff-size
    guardrail with explicit additions/deletions); otherwise a minimal file list is
    built from ``changed_paths``.
    """
    client = _gh_client()
    client.get_pr = AsyncMock(
        return_value={
            "draft": draft,
            "author_association": association,
            "user": {"login": "alice", "type": user_type},
            "labels": [{"name": name} for name in (labels or [])],
            "base": {"ref": base_ref, "sha": "BASE_SHA", "repo": {"full_name": "owner/repo"}},
            "head": {"ref": "feat", "sha": "HEAD_SHA", "repo": {"full_name": head_repo}},
        }
    )
    client.get_file_content = AsyncMock(return_value=config_yaml)
    pr_files = files if files is not None else [
        {"filename": p} for p in (changed_paths or ["src/app.py"])
    ]
    client.get_pr_files = AsyncMock(return_value=pr_files)
    return client


def _run_with_real_gate(client: AsyncMock, synth_mock: AsyncMock) -> ExitStack:
    """ExitStack patching everything except the real _gate_review under test."""
    stack = ExitStack()
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(
        patch("heimdall.worker.run_lens", new=AsyncMock(return_value=_lens_result([])))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    return stack


async def _drive(client: AsyncMock) -> None:
    await run_review(
        {"db": AsyncMock(), "app_id": _APP_ID, "private_key": _PRIVATE_KEY},
        installation_id=_INSTALL_ID,
        repo_full_name=_REPO,
        pr_number=_PR,
        head_sha=_SHA,
    )


@pytest.mark.asyncio
async def test_run_review_no_config_posts_nothing() -> None:
    """Acceptance #1: a repo with no heimdall.yml is not reviewed (no post)."""
    client = _gating_gh_client(config_yaml=None)
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_fork_reads_config_from_base_ref() -> None:
    """Acceptance #2: a fork PR loads config from the BASE sha, ignoring head."""
    client = _gating_gh_client(config_yaml="", head_repo="attacker/repo")
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    client.get_file_content.assert_awaited_once_with(
        repo_full_name=_REPO,
        path=".github/heimdall.yml",
        ref="BASE_SHA",
        tolerate_missing=True,
    )


@pytest.mark.asyncio
async def test_run_review_custom_lens_reaches_synthesis() -> None:
    """Issue #9: a custom lens defined in config runs and reaches synthesis."""
    client = _gating_gh_client(
        config_yaml=(
            "custom_lenses:\n"
            "  - name: licensing\n"
            "    system_prompt: Review license headers.\n"
        )
    )
    run_lens_mock = AsyncMock(return_value=_lens_result([]))
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    ran = {call.kwargs["lens"].name for call in run_lens_mock.await_args_list}
    assert "licensing" in ran
    assert ran == {"security", "design", "cleanliness", "licensing"}
    custom = next(
        call.kwargs["lens"]
        for call in run_lens_mock.await_args_list
        if call.kwargs["lens"].name == "licensing"
    )
    assert custom.system_prompt == "Review license headers."
    synth_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_review_fork_custom_lens_prompt_from_base_ref() -> None:
    """Issue #9 (SECURITY): a fork PR's custom-lens prompt comes from BASE, not head.

    The config (custom-lens prompt included) is read once from the BASE sha for a
    fork, so the head ref is never consulted for the prompt text.
    """
    client = _gating_gh_client(
        config_yaml=(
            "custom_lenses:\n"
            "  - name: licensing\n"
            "    system_prompt: BASE-TRUSTED license prompt.\n"
        ),
        head_repo="attacker/repo",
    )
    run_lens_mock = AsyncMock(return_value=_lens_result([]))
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    # Config (and thus the custom-lens prompt) is read only from the BASE sha.
    client.get_file_content.assert_awaited_once_with(
        repo_full_name=_REPO,
        path=".github/heimdall.yml",
        ref="BASE_SHA",
        tolerate_missing=True,
    )
    custom = next(
        call.kwargs["lens"]
        for call in run_lens_mock.await_args_list
        if call.kwargs["lens"].name == "licensing"
    )
    assert custom.system_prompt == "BASE-TRUSTED license prompt."


@pytest.mark.asyncio
async def test_run_review_per_lens_instructions_modify_builtin_prompt() -> None:
    """Issue #9: per-lens instructions append to the built-in lens's prompt."""
    from heimdall.lens import SECURITY_LENS

    client = _gating_gh_client(
        config_yaml=(
            "lenses:\n"
            "  security:\n"
            "    instructions: Focus on our auth token handling.\n"
        )
    )
    run_lens_mock = AsyncMock(return_value=_lens_result([]))
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    security = next(
        call.kwargs["lens"]
        for call in run_lens_mock.await_args_list
        if call.kwargs["lens"].name == "security"
    )
    assert security.system_prompt.startswith(SECURITY_LENS.system_prompt)
    assert "Focus on our auth token handling." in security.system_prompt


@pytest.mark.asyncio
async def test_run_review_disabled_lens_does_not_run() -> None:
    """Acceptance #3: a lens disabled in config never runs (not in fanout)."""
    client = _gating_gh_client(config_yaml="lenses:\n  design:\n    enabled: false\n")
    run_lens_mock = AsyncMock(return_value=_lens_result([]))
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    _patch_guardrails(stack)
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(patch("heimdall.worker.run_lens", new=run_lens_mock))
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    ran = {call.kwargs["lens"].name for call in run_lens_mock.await_args_list}
    assert "design" not in ran
    assert ran == {"security", "cleanliness"}


@pytest.mark.asyncio
async def test_run_review_threshold_changes_verdict() -> None:
    """Acceptance #3: a CRITICAL threshold makes a HIGH finding COMMENT, not block."""
    client = _gating_gh_client(config_yaml="severity_threshold: critical")
    captured: dict[str, object] = {}

    async def _synth(**kwargs: object) -> SynthesisResult:
        captured["blocking"] = kwargs["blocking"]
        from heimdall.lens import format_synthesis_body, verdict_for_tagged

        tagged = [_tagged(Severity.HIGH, "security", "x")]
        blocking = kwargs["blocking"]
        assert isinstance(blocking, frozenset)
        return SynthesisResult(
            tagged_findings=tagged,
            verdict=verdict_for_tagged(tagged, blocking=blocking),
            body=format_synthesis_body(tagged),
        )

    synth_mock = AsyncMock(side_effect=_synth)
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    # HIGH does not block under a CRITICAL threshold -> COMMENT.
    assert captured["blocking"] == frozenset({Severity.CRITICAL})
    assert client.post_review.await_args.kwargs["event"] == "COMMENT"


@pytest.mark.asyncio
async def test_run_review_base_branch_filter_skips() -> None:
    """Acceptance #3: a base branch outside the allowlist is skipped (no post)."""
    client = _gating_gh_client(
        config_yaml="scope:\n  base_branches: [release]\n", base_ref="main"
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_path_filter_skips() -> None:
    """Acceptance #3: a PR touching only out-of-scope paths is skipped (no post)."""
    client = _gating_gh_client(
        config_yaml="scope:\n  paths: ['src/**']\n", changed_paths=["docs/x.md"]
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_draft_skipped() -> None:
    """Acceptance #4: a draft PR is skipped per config (no post)."""
    client = _gating_gh_client(config_yaml="", draft=True)
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_bot_author_skipped() -> None:
    """Acceptance #4: a bot-authored PR is skipped per config (no post)."""
    client = _gating_gh_client(config_yaml="", user_type="Bot")
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_run_review_opt_out_label_skipped() -> None:
    """Acceptance #4: a PR carrying the opt-out label is skipped (no post)."""
    client = _gating_gh_client(
        config_yaml="scope:\n  opt_out_label: heimdall-skip\n", labels=["heimdall-skip"]
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


def test_lens_config_and_scope_filters_constructible() -> None:
    """The repo-config models are importable and constructible (contract smoke)."""
    cfg = RepoConfig(
        lenses={"security": LensConfig(model="sonnet")},
        scope=ScopeFilters(base_branches=["main"]),
    )
    assert cfg.lenses["security"].model == "sonnet"
    assert cfg.scope.base_branches == ["main"]


# ---------------------------------------------------------------------------
# Issue #10 — guardrail caps through run_review
#   1. oversized PR skipped WITH a posted note
#   2. per-repo rate/budget beyond limit -> skipped (no post)
#   3. per-installation concurrency never exceeds the cap
#   4. all three configurable; defaults applied when absent
# ---------------------------------------------------------------------------


def _big_files(n: int, *, additions: int = 1, deletions: int = 0) -> list[dict[str, Any]]:
    """Build ``n`` PR-file objects each carrying the given additions/deletions."""
    return [
        {"filename": f"src/f{i}.py", "additions": additions, "deletions": deletions}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_oversized_pr_skipped_with_posted_note() -> None:
    """Acceptance #1: a PR over the file cap is skipped, but a COMMENT note is posted."""
    # Cap files at 2; the PR changes 5 -> over cap.
    client = _gating_gh_client(
        config_yaml="caps:\n  max_files: 2\n",
        files=_big_files(5),
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    # The real _gate_review runs; only the per-repo/concurrency DB helpers are stubbed.
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True))
    )
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=AsyncMock()))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    # The pipeline never ran (the gate skipped it)...
    synth_mock.assert_not_called()
    # ...but a single terse COMMENT note WAS posted explaining the size skip.
    client.post_review.assert_awaited_once()
    posted = client.post_review.await_args.kwargs
    assert posted["event"] == "COMMENT"
    assert "too large" in posted["body"].lower()
    assert "5 files" in posted["body"]


@pytest.mark.asyncio
async def test_oversized_diff_lines_skipped_with_posted_note() -> None:
    """Acceptance #1: a PR over the diff-line cap is also skipped with a posted note."""
    # Cap diff lines at 10; two files each with 8 additions -> 16 changed lines.
    client = _gating_gh_client(
        config_yaml="caps:\n  max_diff_lines: 10\n",
        files=_big_files(2, additions=8),
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True))
    )
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=AsyncMock()))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_awaited_once()
    assert "changed lines" in client.post_review.await_args.kwargs["body"]


@pytest.mark.asyncio
async def test_small_pr_under_default_caps_is_reviewed() -> None:
    """Acceptance #4: with no caps block, a small PR is NOT skipped (defaults are sane)."""
    client = _gating_gh_client(config_yaml="", files=_big_files(2, additions=3))
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with _run_with_real_gate(client, synth_mock):
        await _drive(client)

    # Default caps do not fire for a tiny PR: the pipeline runs and a review posts.
    synth_mock.assert_called()
    client.post_review.assert_awaited_once()


@pytest.mark.asyncio
async def test_rate_budget_exceeded_skips_without_posting() -> None:
    """Acceptance #2: a review beyond the per-repo budget is skipped (no pipeline, no post)."""
    client = _gating_gh_client(config_yaml="caps:\n  max_reviews_per_window: 1\n")
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    # Force the rate gate over-budget; concurrency acquire would succeed but is
    # never reached because the rate gate returns first.
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=True))
    )
    acquire_mock = AsyncMock(return_value=True)
    stack.enter_context(patch("heimdall.worker.try_acquire_inflight", new=acquire_mock))
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=AsyncMock()))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()
    # Over budget short-circuits before claiming a concurrency slot.
    acquire_mock.assert_not_called()


@pytest.mark.asyncio
async def test_rate_budget_uses_real_db_window() -> None:
    """Acceptance #2: the rate gate counts real DB events within the window.

    Drives run_review with a real in-memory Database pre-loaded with events up to
    the budget, so the next review is skipped purely from DB-backed state.
    """
    from heimdall.db import Database, record_review_event

    db = Database(":memory:")
    await db.initialize()
    # Budget of 2 per (large) window; pre-record 2 recent events -> at the limit.
    now = __import__("time").time()
    await record_review_event(db, repo_full_name=_REPO, occurred_at=now)
    await record_review_event(db, repo_full_name=_REPO, occurred_at=now)

    client = _gating_gh_client(
        config_yaml="caps:\n  max_reviews_per_window: 2\n  rate_window_seconds: 3600\n"
    )
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.run_synthesis", new=synth_mock),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()),
        patch("heimdall.worker.GitHubClient", return_value=client),
    ):
        await run_review(
            {"db": db, "app_id": _APP_ID, "private_key": _PRIVATE_KEY},
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    await db.close()
    # Already at the budget -> the fresh review is skipped, nothing posted.
    synth_mock.assert_not_called()
    client.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_concurrency_cap_defers_when_at_limit() -> None:
    """Acceptance #3: when the installation is at its concurrency cap, the run defers."""
    client = _gating_gh_client(config_yaml="caps:\n  max_concurrent_per_installation: 1\n")
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    release_mock = AsyncMock()
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    # Acquire refused -> at cap.
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=False))
    )
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=release_mock))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    synth_mock.assert_not_called()
    client.post_review.assert_not_called()
    # A slot was never claimed, so it must NOT be released (no leak the other way).
    release_mock.assert_not_called()


@pytest.mark.asyncio
async def test_concurrency_slot_released_on_every_path() -> None:
    """Acceptance #3: a claimed slot is released even when the pipeline raises.

    A leaked slot would permanently shrink the installation's budget, so the
    release MUST run in a finally regardless of how the review ends.
    """
    client = _gating_gh_client(config_yaml="")
    release_mock = AsyncMock()
    # The pipeline blows up after the slot is claimed.
    boom = AsyncMock(side_effect=RuntimeError("pipeline exploded"))
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True))
    )
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=release_mock))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(patch("heimdall.worker._review_and_post", new=boom))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack, pytest.raises(RuntimeError):
        await _drive(client)

    # Even though the pipeline raised, the slot was released exactly once.
    release_mock.assert_awaited_once()
    assert release_mock.await_args is not None
    assert release_mock.await_args.kwargs["installation_id"] == _INSTALL_ID


@pytest.mark.asyncio
async def test_concurrency_records_event_and_releases_on_success() -> None:
    """Acceptance #3: a successful review records a rate event and releases its slot."""
    client = _gating_gh_client(config_yaml="")
    record_mock = AsyncMock()
    release_mock = AsyncMock()
    synth_mock = AsyncMock(return_value=_synthesis_from([]))
    stack = ExitStack()
    stack.enter_context(
        patch("heimdall.worker._over_rate_budget", new=AsyncMock(return_value=False))
    )
    stack.enter_context(
        patch("heimdall.worker.try_acquire_inflight", new=AsyncMock(return_value=True))
    )
    stack.enter_context(patch("heimdall.worker.try_record_review_event", new=record_mock))
    stack.enter_context(patch("heimdall.worker.release_inflight", new=release_mock))
    stack.enter_context(
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.get_posted_review", new=AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch("heimdall.worker.assemble_pr_context", new=AsyncMock(return_value=MagicMock()))
    )
    stack.enter_context(
        patch("heimdall.worker.run_lens", new=AsyncMock(return_value=_lens_result([])))
    )
    stack.enter_context(patch("heimdall.worker.run_synthesis", new=synth_mock))
    stack.enter_context(patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.set_posted_review", new=AsyncMock()))
    stack.enter_context(patch("heimdall.worker.GitHubClient", return_value=client))
    with stack:
        await _drive(client)

    record_mock.assert_awaited_once()
    release_mock.assert_awaited_once()
    client.post_review.assert_awaited_once()
