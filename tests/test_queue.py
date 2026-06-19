"""Tests for the Arq queue module: enqueue, cancel-stale logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.queue import ReviewJob, cancel_stale_jobs, enqueue_review


@pytest.fixture()
def job() -> ReviewJob:
    return ReviewJob(
        installation_id=42,
        repo_full_name="owner/repo",
        pr_number=7,
        head_sha="deadbeef",
    )


@pytest.mark.asyncio
async def test_enqueue_review_calls_arq(job: ReviewJob) -> None:
    """enqueue_review pushes a job onto the Arq queue with the correct function name."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()

    with patch("heimdall.queue.find_pending_jobs", new=AsyncMock(return_value=[])):
        await enqueue_review(mock_pool, job)

    mock_pool.enqueue_job.assert_awaited_once()
    call_args = mock_pool.enqueue_job.call_args
    assert call_args[0][0] == "run_review"
    # kwargs are passed through to the worker
    assert call_args[1]["repo_full_name"] == job.repo_full_name
    assert call_args[1]["pr_number"] == job.pr_number
    assert call_args[1]["head_sha"] == job.head_sha


@pytest.mark.asyncio
async def test_cancel_stale_jobs_removes_from_queue(job: ReviewJob) -> None:
    """cancel_stale_jobs removes stale queued jobs via ZREM."""
    mock_pool = AsyncMock()
    mock_pool.zrem = AsyncMock()

    stale_job = MagicMock()
    stale_job.job_id = "stale-jid"

    with patch("heimdall.queue.find_pending_jobs", new=AsyncMock(return_value=[stale_job])):
        await cancel_stale_jobs(mock_pool, repo_full_name="owner/repo", pr_number=7)

    mock_pool.zrem.assert_awaited_once()
    assert "stale-jid" in mock_pool.zrem.call_args[0]
