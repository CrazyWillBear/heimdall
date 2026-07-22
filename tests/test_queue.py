"""Tests for the Arq queue module: enqueue, cancel-stale logic."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from arq import ArqRedis

from heimdall.queue import (
    ReviewJob,
    cancel_stale_jobs,
    enqueue_review,
    find_pending_jobs,
)


def _queued_job(job_id: str, *, action: str | None = "synchronize") -> MagicMock:
    """Build a fake arq JobDef for owner/repo PR 7 with the given action kwarg.

    Passing ``action=None`` omits the kwarg entirely, mimicking a legacy job
    enqueued before the action kwarg existed.
    """
    kwargs: dict[str, object] = {"repo_full_name": "owner/repo", "pr_number": 7}
    if action is not None:
        kwargs["action"] = action
    job_def = MagicMock()
    job_def.job_id = job_id
    job_def.kwargs = kwargs
    return job_def


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
    mock_pool.queued_jobs = AsyncMock(return_value=[])

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
    mock_pool.queued_jobs = AsyncMock(return_value=[])

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


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_action", ["ready_for_review", "review_requested"])
async def test_signal_job_is_returned_as_stale(signal_action: str) -> None:
    """find_pending_jobs matches by repo+PR alone — a queued signal job is now cancellable.

    Sparing is gone: enqueue_review folds a pending signal's intent forward onto the
    superseding push (promotion) BEFORE cancelling, so cancel_stale_jobs itself is
    action-agnostic and supersedes every queued job for the PR, signal or not, leaving
    exactly one to run.
    """
    mock_pool = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("signal-jid", action=signal_action)]
    )

    found = await find_pending_jobs(mock_pool, repo_full_name="owner/repo", pr_number=7)

    assert [job.job_id for job in found] == ["signal-jid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_action", ["ready_for_review", "review_requested"])
async def test_pending_signal_promotes_incoming_sync(
    job: ReviewJob, signal_action: str
) -> None:
    """A queued signal job promotes a superseding synchronize enqueue to the signal action.

    Fold-forward: signal(A) queued, then push->sync(B) arrives ⇒ B is enqueued carrying
    the signal action (so the single surviving job still activates the PR and reviews the
    newest sha), and the stale signal job A is cancelled.
    """
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("signal-jid", action=signal_action)]
    )

    await enqueue_review(mock_pool, job)  # job.action == "synchronize"

    call = mock_pool.enqueue_job.call_args
    assert call.kwargs["action"] == signal_action
    assert call.kwargs["_job_id"].endswith(f":{signal_action}")
    # The stale signal job is superseded (cancelled) — exactly one job survives.
    mock_pool.zrem.assert_awaited_once()
    assert "signal-jid" in mock_pool.zrem.call_args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_action", ["ready_for_review", "review_requested"])
async def test_same_sha_promotion_leaves_queued_signal_in_place(
    job: ReviewJob, signal_action: str
) -> None:
    """Same-sha promotion keeps the already-queued signal job: no ZREM, no re-enqueue.

    Regression: signal(sha X) queued, then a synchronize(sha X) for the SAME sha arrives.
    Promotion would reproduce the signal job's exact id. Cancelling + re-enqueuing that
    id is self-defeating — ``cancel_stale_jobs`` ZREMs it from the queue but arq's
    ``arq:job:{id}`` dedup marker (~24h TTL) survives, so the re-enqueue is dedup-dropped
    and zero jobs would remain, silently losing the signal. The queued signal already
    covers this sha, so leave it in place and return the dedup sentinel.
    """
    signal_id = f"review:owner/repo:7:{job.head_sha}:{signal_action}"
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job(signal_id, action=signal_action)]
    )

    result = await enqueue_review(mock_pool, job)  # job.action == "synchronize", same sha

    # The already-queued signal job survives: not cancelled, not re-enqueued.
    mock_pool.zrem.assert_not_awaited()
    mock_pool.enqueue_job.assert_not_awaited()
    assert result == ""


@pytest.mark.asyncio
async def test_same_sha_collision_still_cancels_other_stale_jobs(job: ReviewJob) -> None:
    """On a same-sha collision the matching signal job is kept, but other stale jobs die."""
    signal_id = f"review:owner/repo:7:{job.head_sha}:ready_for_review"
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[
            _queued_job(signal_id, action="ready_for_review"),
            _queued_job("other-stale-jid", action="synchronize"),
        ]
    )

    result = await enqueue_review(mock_pool, job)  # same sha as the queued signal

    # The colliding signal job survives; the other stale job is superseded; no re-enqueue.
    mock_pool.enqueue_job.assert_not_awaited()
    assert result == ""
    zrem_ids = [call.args[1] for call in mock_pool.zrem.await_args_list]
    assert signal_id not in zrem_ids
    assert "other-stale-jid" in zrem_ids


@pytest.mark.asyncio
async def test_plain_sync_not_promoted_without_signal(job: ReviewJob) -> None:
    """A synchronize enqueue with no queued signal keeps its action (no false promotion)."""
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("sync-jid", action="synchronize")]
    )

    await enqueue_review(mock_pool, job)  # job.action == "synchronize"

    call = mock_pool.enqueue_job.call_args
    assert call.kwargs["action"] == "synchronize"
    assert call.kwargs["_job_id"].endswith(":synchronize")
    # The stale synchronize job is still superseded-by-push.
    mock_pool.zrem.assert_awaited_once()
    assert "sync-jid" in mock_pool.zrem.call_args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("signal_action", ["ready_for_review", "review_requested"])
async def test_incoming_signal_not_demoted(job: ReviewJob, signal_action: str) -> None:
    """An incoming signal job stays the signal even when a stale sync is queued."""
    from dataclasses import replace

    signal_job = replace(job, action=signal_action)
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock(job_id="jid-1"))
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("sync-jid", action="synchronize")]
    )

    await enqueue_review(mock_pool, signal_job)

    call = mock_pool.enqueue_job.call_args
    assert call.kwargs["action"] == signal_action
    # The stale sync is cancelled as usual.
    mock_pool.zrem.assert_awaited_once()
    assert "sync-jid" in mock_pool.zrem.call_args[0]


@pytest.mark.asyncio
async def test_synchronize_job_still_cancelled() -> None:
    """Non-signal (synchronize) jobs for the same PR are still cancelled (supersede-by-push)."""
    mock_pool = AsyncMock()
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("sync-jid", action="synchronize")]
    )

    await cancel_stale_jobs(mock_pool, repo_full_name="owner/repo", pr_number=7)

    mock_pool.zrem.assert_awaited_once()
    assert "sync-jid" in mock_pool.zrem.call_args[0]


@pytest.mark.asyncio
async def test_legacy_job_without_action_still_cancelled() -> None:
    """A job enqueued before the action kwarg (missing/None) is treated as cancellable."""
    mock_pool = AsyncMock()
    mock_pool.zrem = AsyncMock()
    mock_pool.queued_jobs = AsyncMock(
        return_value=[_queued_job("legacy-jid", action=None)]
    )

    await cancel_stale_jobs(mock_pool, repo_full_name="owner/repo", pr_number=7)

    mock_pool.zrem.assert_awaited_once()
    assert "legacy-jid" in mock_pool.zrem.call_args[0]


class _FakePool:
    """In-process pool with faithfully shared, mutating state.

    This is NOT a mock of the lock — it implements the real ``SET NX PX`` and
    token-guarded-release contracts the lock relies on, plus a mutating queue, so two
    concurrent ``enqueue_review`` coroutines actually observe each other's effects.
    """

    def __init__(self, initial_jobs: list[SimpleNamespace] | None = None) -> None:
        self._jobs: list[SimpleNamespace] = list(initial_jobs or [])
        self._store: dict[str, str] = {}  # lock keys -> token
        self._seen_ids: set[str] = set()  # arq:job:{id} dedup markers

    async def queued_jobs(self) -> list[SimpleNamespace]:
        # Forced interleave point: yield to the event loop before returning the
        # snapshot so a concurrent coroutine observes the pre-enqueue queue state.
        await asyncio.sleep(0)
        return list(self._jobs)

    async def set(
        self,
        name: str,
        value: str,
        *,
        nx: bool = False,
        px: int | None = None,
    ) -> bool | None:
        # SET NX: no await inside, so acquisition is atomic vs other coroutines,
        # mirroring real Redis. Returns None when the key is already held.
        if nx and name in self._store:
            return None
        self._store[name] = value
        return True

    async def eval(self, script: str, numkeys: int, *args: str) -> int:
        # Mirrors _RELEASE_LOCK_SCRIPT: token-guarded delete (ignores script text).
        key, token = args[0], args[1]
        if self._store.get(key) == token:
            del self._store[key]
            return 1
        return 0

    async def zrem(self, queue_name: str, job_id: str) -> None:
        # Faithful to arq: the arq:job:{id} marker in _seen_ids survives a ZREM.
        self._jobs = [j for j in self._jobs if j.job_id != job_id]

    async def enqueue_job(
        self, function: str, *, _job_id: str, **kwargs: object
    ) -> SimpleNamespace | None:
        if _job_id in self._seen_ids:
            return None  # dedup: id already known
        # Widen the interleave window: yield after the cancel+decide but before the
        # append, so an unserialized concurrent caller can slip its own cancel in
        # between (removing this job's not-yet-appended peer) and double-enqueue.
        await asyncio.sleep(0)
        self._seen_ids.add(_job_id)
        self._jobs.append(SimpleNamespace(job_id=_job_id, kwargs=kwargs))
        return SimpleNamespace(job_id=_job_id)


@pytest.mark.asyncio
async def test_concurrent_same_pr_enqueues_serialized(job: ReviewJob) -> None:
    """Two same-PR enqueues run concurrently and leave exactly one folded survivor.

    Must FAIL on the pre-lock code: the ``queued_jobs`` interleave point lets both
    coroutines read the same pre-enqueue snapshot, so each cancels the other's view and
    both ``enqueue_job`` land -> two survivors. With ``_pr_lock`` serializing the
    decide+cancel+enqueue section, the loser spins on ``SET NX`` until the winner
    releases, then re-reads the mutated queue (winner's promoted ``review_requested`` job
    present), folds again, cancels the winner's job, and enqueues its own -> exactly one
    survivor carrying ``review_requested`` at one of the incoming pushes' shas.
    """
    signal_job = SimpleNamespace(
        job_id="review:owner/repo:7:sha0:review_requested",
        kwargs={
            "repo_full_name": "owner/repo",
            "pr_number": 7,
            "action": "review_requested",
        },
    )
    pool = _FakePool(initial_jobs=[signal_job])

    sync1 = replace(job, head_sha="sha1", action="synchronize")
    sync2 = replace(job, head_sha="sha2", action="synchronize")

    arq_pool = cast(ArqRedis, pool)
    await asyncio.gather(
        enqueue_review(arq_pool, sync1), enqueue_review(arq_pool, sync2)
    )

    assert len(pool._jobs) == 1
    survivor = pool._jobs[0]
    assert survivor.kwargs["action"] == "review_requested"  # signal folded forward
    assert survivor.kwargs["head_sha"] in {"sha1", "sha2"}  # a push won, never sha0
