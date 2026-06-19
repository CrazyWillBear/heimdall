"""Tests for the Arq worker: posts exactly one review, updates last SHA."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from heimdall.worker import run_review

_REPO = "owner/repo"
_PR = 3
_SHA = "sha1234"
_INSTALL_ID = 42


@pytest.mark.asyncio
async def test_run_review_posts_exactly_one_review() -> None:
    """Worker calls GitHub API once to post a review, then records the SHA."""
    mock_db = AsyncMock()
    mock_gh_client = AsyncMock()
    mock_gh_client.post_review = AsyncMock()

    ctx: dict[str, object] = {"db": mock_db, "github_client": mock_gh_client}

    with (
        patch("heimdall.worker.get_last_reviewed_sha", new=AsyncMock(return_value=None)),
        patch("heimdall.worker.set_last_reviewed_sha", new=AsyncMock()) as mock_set,
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
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

    ctx: dict[str, object] = {"db": mock_db, "github_client": mock_gh_client}

    with patch(
        "heimdall.worker.get_last_reviewed_sha",
        new=AsyncMock(return_value=_SHA),
    ):
        await run_review(
            ctx,
            installation_id=_INSTALL_ID,
            repo_full_name=_REPO,
            pr_number=_PR,
            head_sha=_SHA,
        )

    mock_gh_client.post_review.assert_not_called()
