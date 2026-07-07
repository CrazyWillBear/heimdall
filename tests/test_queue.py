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
        action="synchronize",
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
    # The action is threaded through as a kwarg and baked into the job id.
    assert call_args[1]["action"] == job.action
    assert call_args[1]["_job_id"] == (
        f"review:{job.repo_full_name}:{job.pr_number}:{job.head_sha}:{job.action}"
    )


@pytest.mark.asyncio
async def test_action_distinguishes_job_id(job: ReviewJob) -> None:
    """Two jobs identical but for their action produce distinct arq job ids.

    Without this a skipped pre-activation ``synchronize`` would dedup-block a later
    ``review_requested`` for the same sha (arq keeps a finished job's result under its
    id and refuses re-enqueue).
    """
    from dataclasses import replace

    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()

    job_signal = replace(job, action="review_requested")
    with patch("heimdall.queue.find_pending_jobs", new=AsyncMock(return_value=[])):
        await enqueue_review(mock_pool, job)
        await enqueue_review(mock_pool, job_signal)

    job_ids = [call.kwargs["_job_id"] for call in mock_pool.enqueue_job.await_args_list]
    assert job_ids[0] != job_ids[1]


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
