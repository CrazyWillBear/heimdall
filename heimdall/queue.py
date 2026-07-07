"""Arq queue helpers: job model, enqueue, and cancel-stale logic.

cancel_stale_jobs must be called before enqueue_review so that an earlier
queued job for the same PR is removed before the new one is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass

from arq import ArqRedis
from arq.constants import default_queue_name
from arq.jobs import Job

# Webhook actions that count as an explicit review signal under scope.trigger: on_signal.
# Any review_requested counts (even one not naming Heimdall) because a GitHub App bot
# cannot be picked as a PR reviewer — the request is the human's intent to be reviewed.
# Single source of truth: worker.py imports this for its on_signal gate, and enqueue_review
# uses it to fold a pending signal's intent forward onto a superseding push (promotion).
_SIGNAL_ACTIONS = frozenset({"ready_for_review", "review_requested"})


@dataclass(frozen=True)
class ReviewJob:
    """Serialisable description of a single PR review task.

    Attributes:
        installation_id: GitHub App installation the PR belongs to.
        repo_full_name: e.g. "owner/repo".
        pr_number: Pull-request number.
        head_sha: Commit SHA at the tip of the PR.
        action: The ``pull_request`` webhook action that produced the job; feeds the
            worker's on_signal trigger gate.
    """

    installation_id: int
    repo_full_name: str
    pr_number: int
    head_sha: str
    action: str


async def find_pending_jobs(
    pool: ArqRedis,
    *,
    repo_full_name: str,
    pr_number: int,
) -> list[Job]:
    """Return Job handles for any queued (not-yet-running) jobs for this PR.

    Scans the Arq queue and matches by the repo+PR kwargs embedded at enqueue time —
    every queued job for the PR, signal or not, so ``cancel_stale_jobs`` can supersede
    them all and leave exactly one to run.  A queued signal's intent is not lost to this:
    ``enqueue_review`` folds it forward onto the superseding push (promotion) BEFORE
    cancelling, so the single surviving job still activates the PR.
    """
    queued = await pool.queued_jobs()
    jobs: list[Job] = []
    for job_def in queued:
        kw = job_def.kwargs
        if (
            kw.get("repo_full_name") == repo_full_name
            and kw.get("pr_number") == pr_number
            and job_def.job_id is not None
        ):
            jobs.append(Job(job_def.job_id, pool))
    return jobs


async def _pending_signal_action(
    pool: ArqRedis,
    *,
    repo_full_name: str,
    pr_number: int,
) -> str | None:
    """Return the action of a queued signal job for this PR, if one is pending.

    Scans the Arq queue for a signal job matching this repo+PR whose ``action`` is a
    review signal (see ``_SIGNAL_ACTIONS``).  arq keeps in-progress jobs in the queue
    zset until they complete, so this matches a signal job still in the queue zset —
    queued or currently running (a running signal still folds forward correctly).
    ``enqueue_review`` uses this to fold a pending signal's intent forward onto a
    superseding non-signal push, so the single surviving job still activates the PR when
    it runs.  Returns the first signal action found, or None when no signal job is queued.
    """
    queued = await pool.queued_jobs()
    for job_def in queued:
        kw = job_def.kwargs
        if (
            kw.get("repo_full_name") == repo_full_name
            and kw.get("pr_number") == pr_number
            and kw.get("action") in _SIGNAL_ACTIONS
        ):
            return str(kw["action"])
    return None


async def cancel_stale_jobs(
    pool: ArqRedis,
    *,
    repo_full_name: str,
    pr_number: int,
    keep_job_id: str | None = None,
) -> None:
    """Remove any queued review jobs for this PR from the Arq queue.

    Uses a direct ZREM on the queue sorted set so we don't have to wait for
    a running job to finish (queued-only jobs haven't started yet).

    ``keep_job_id`` spares a single queued job with that exact id.  ``enqueue_review``
    passes it on a same-sha promotion collision so the already-queued signal job that
    covers this sha is left in place rather than ZREM'd and re-enqueued — arq's
    ``arq:job:{id}`` dedup marker survives the ZREM, so a same-id re-enqueue is dropped
    and the job would be lost.
    """
    stale = await find_pending_jobs(pool, repo_full_name=repo_full_name, pr_number=pr_number)
    for job in stale:
        if job.job_id == keep_job_id:
            continue
        await pool.zrem(default_queue_name, job.job_id)


async def enqueue_review(pool: ArqRedis, job: ReviewJob) -> str:
    """Fold any pending signal forward, cancel stale jobs, then enqueue the job.

    Passes the ReviewJob fields as keyword arguments so the worker can receive
    them and Arq's queue-scanning can match on them.

    Promotion (fold-the-intent-forward): if a signal job (``ready_for_review`` /
    ``review_requested``) is still queued for this PR and the incoming ``job`` is a plain
    non-signal push (e.g. ``synchronize``), the incoming job is enqueued as if its action
    were that pending signal.  Every stale queued job for the PR is then cancelled, so
    exactly ONE job survives — and it carries the signal intent, so it both activates the
    PR and reviews the newest sha (avoiding the double-review + stale-anchor hazard of
    letting an older signal job and a superseding push both run).  An incoming signal job
    is never demoted; a plain push with no pending signal is unaffected.

    Same-sha collision guard: when promotion would reproduce the exact ``_job_id`` of a
    job already queued for this PR (the pending signal job IS this repo+PR+sha+signal-
    action), that queued job already covers this sha and will activate the PR when it
    runs.  Cancelling + re-enqueuing it is self-defeating — ``cancel_stale_jobs`` ZREMs it
    from the queue but arq's ``arq:job:{id}`` dedup marker (~24h TTL) survives, so the
    same-id re-enqueue is dropped and zero jobs would remain, silently losing the signal.
    So we leave that job queued, still cancel any OTHER stale jobs, and return the dedup
    sentinel.

    Args:
        pool: The connected Arq Redis pool.
        job: The review job to enqueue.

    Returns:
        The Arq job ID of the newly enqueued job (empty string if deduplicated, or when a
        same-sha promotion collides with the already-queued signal job).
    """
    action = job.action
    if action not in _SIGNAL_ACTIONS:
        pending_signal = await _pending_signal_action(
            pool, repo_full_name=job.repo_full_name, pr_number=job.pr_number
        )
        if pending_signal is not None:
            action = pending_signal
    # The action is part of the job id because arq keeps a finished/skipped job's result
    # under its id (~1h) and refuses to re-enqueue the same id: without it, a skipped
    # pre-activation `synchronize` would dedup-block the later `review_requested` for the
    # same sha, and that signal would never reach the worker.
    job_id = f"review:{job.repo_full_name}:{job.pr_number}:{job.head_sha}:{action}"
    # If promotion reproduces the id of a job already queued for this PR, that queued job
    # already covers this sha: leave it in place (cancel + same-id re-enqueue would be
    # dedup-dropped, losing the job) and only supersede the OTHER stale jobs.
    pending = await find_pending_jobs(
        pool, repo_full_name=job.repo_full_name, pr_number=job.pr_number
    )
    if any(pending_job.job_id == job_id for pending_job in pending):
        await cancel_stale_jobs(
            pool,
            repo_full_name=job.repo_full_name,
            pr_number=job.pr_number,
            keep_job_id=job_id,
        )
        return ""
    await cancel_stale_jobs(
        pool, repo_full_name=job.repo_full_name, pr_number=job.pr_number
    )
    arq_job = await pool.enqueue_job(
        "run_review",
        _job_id=job_id,
        installation_id=job.installation_id,
        repo_full_name=job.repo_full_name,
        pr_number=job.pr_number,
        head_sha=job.head_sha,
        action=action,
    )
    # arq_job is None when the job_id already exists (idempotent re-submission)
    return arq_job.job_id if arq_job is not None else ""
