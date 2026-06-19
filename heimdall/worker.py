"""Arq worker: the run_review task function and WorkerSettings.

Context keys populated by WorkerSettings.on_startup:
    db:                     heimdall.db.Database instance
    app_id:                 GitHub App numeric ID (int)
    private_key:            PEM-encoded RSA private key (str)
    claude_binary:          path/name of the claude CLI (str)
    claude_env_passthrough: extra env keys forwarded to the claude child (list[str])
    lens_token_cap:         per-agent cumulative-token cap (int)
    lens_timeout_seconds:   per-lens wall-clock timeout (float)
    review_timeout_seconds: per-review wall-clock timeout across the pipeline (float)
    debug_logging:          when True, log findings/code text (else metadata-only) (bool)

run_review builds a GitHubClient per-job using ctx["app_id"], ctx["private_key"],
and the per-job installation_id argument.  Before any review work it gates the PR
(see heimdall.repo_config): it loads ``.github/heimdall.yml`` from the trust-resolved
ref (base for forks, head for trusted same-repo PRs) — a missing file means the repo
has not opted in, so nothing is reviewed — then applies scope filters (base-branch
allowlist, path globs, skip drafts/bot authors, opt-out label) and the guardrail caps:
a PR over the diff-size/file-count cap is skipped WITH a posted note, a repo over its
per-window review budget is skipped silently, and a review that would exceed the
per-installation concurrency cap defers (a DB-backed in-flight slot, released on every
exit path).  If it proceeds, it
assembles the PR seed context into a temporary workspace once, fans out the
config-tuned lenses (built-ins Security opus/max, Design-fit sonnet/high, Cleanliness
sonnet/high, each with per-lens model/effort/enable overrides plus optional appended
instructions, alongside any custom lenses defined in the config) over that shared seed —
each bounded by its own token cap and timeout — then runs a 4th synthesis ``claude -p``
pass that dedups overlapping findings across lenses, ranks by severity, writes the
verdict, and formats the review (findings grouped by severity, each tagged with the
originating lens).  Exactly one PR review is posted: findings on a changed diff line
ride as inline comments in that same submission, while off-diff (or unparseable-
location) findings are rendered in the review body.  On a new push the prior review's
inline comments are deleted before the fresh set is posted.  A failure in any single
lens is isolated (logged, that lens dropped); the pipeline only skips posting when
every lens fails or the synthesis pass itself aborts.  Nothing here ever crashes the
worker.

The whole review pipeline is wrapped in a per-review wall-clock timeout (distinct
from, and looser than, the per-lens timeout) and retried exactly once on any
failure/timeout.  If the retry also fails, a terse "review failed" COMMENT note is
posted instead and the SHA recorded so the failed commit is not re-reviewed.

Logging is metadata-only by default — repo/PR/SHA/timing/verdict — and never logs
tokens or secrets.  Findings and code text are logged only when ``debug_logging`` is
set in ctx.

Launch the worker with:
    arq heimdall.worker.WorkerSettings
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from typing import Any

from arq.connections import RedisSettings

from heimdall.context import assemble_pr_context
from heimdall.db import (
    Database,
    count_recent_reviews,
    get_last_reviewed_sha,
    get_posted_review,
    prune_review_events,
    release_inflight,
    set_last_reviewed_sha,
    set_posted_review,
    try_acquire_inflight,
    try_record_review_event,
)
from heimdall.diff_anchor import (
    build_inline_comments,
    commentable_lines,
    render_body_for_offdiff,
    split_findings,
)
from heimdall.github import GitHubClient
from heimdall.lens import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_CAP,
    LensError,
    LensResult,
    SynthesisResult,
    run_lens,
    run_synthesis,
)
from heimdall.repo_config import (
    GuardrailCaps,
    RepoConfig,
    RepoConfigError,
    blocking_severities,
    diff_cap_skip_note,
    load_repo_config,
    skip_reason,
    tuned_lenses,
)

logger = logging.getLogger(__name__)

# Per-review wall-clock timeout across the whole pipeline (assembly + lens fanout +
# synthesis), distinct from and looser than the per-lens timeout: it bounds the total
# job around the multi-lens fanout and synthesis pass.
DEFAULT_REVIEW_TIMEOUT_SECONDS = 2_400.0

# Posted as a COMMENT (never REQUEST_CHANGES) when both the initial run and the
# single retry fail — a deliberately terse, metadata-free note.
_REVIEW_FAILED_NOTE = (
    "Heimdall review failed: the automated review could not complete after a retry. "
    "No verdict was produced for this commit."
)


def _db_path_from_url(database_url: str) -> str:
    """Strip the SQLAlchemy driver prefix from a database URL for aiosqlite.

    aiosqlite.connect expects a plain file path (or ':memory:'), not a full
    SQLAlchemy DSN.  We only support the sqlite+aiosqlite:/// scheme used by
    the default config.
    """
    prefix = "sqlite+aiosqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix):]
    # Fallback: pass through as-is so plain paths still work in tests
    return database_url


async def run_review(
    ctx: dict[str, Any],
    *,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> None:
    """Arq task: fan out three lenses, synthesize, and post one review.

    Skips if the same head SHA was already reviewed (idempotency guard).  On a
    fresh SHA it runs the review pipeline — assemble the seed context once, fan out
    the three lenses over it (each independently bounded; a single lens failure is
    isolated), then run the synthesis pass that dedups overlapping findings, ranks
    by severity, writes the verdict (REQUEST_CHANGES for a high/critical survivor,
    else COMMENT), and formats the severity-grouped, lens-tagged review body — under
    a per-review wall-clock timeout, retrying the whole pipeline exactly once on any
    failure/timeout.  On success it retires any prior Heimdall review and deletes its
    inline comments, posts exactly one PR review (findings on changed diff lines as
    inline comments in the same submission, off-diff findings in the body), and
    records the SHA.  If every lens fails, the synthesis pass
    aborts, or the retry also times out/fails, a terse "review failed" COMMENT note
    is posted and the SHA recorded so the failed commit is not endlessly re-reviewed.


    A GitHubClient is constructed per-job from the app credentials in ctx so
    that each job can target a different GitHub App installation.

    Args:
        ctx: Arq worker context carrying ``db``, ``app_id``, ``private_key``,
            and the optional lens/review/logging knobs.
        installation_id: GitHub App installation ID for this PR.
        repo_full_name: e.g. "owner/repo".
        pr_number: The pull-request number.
        head_sha: The commit SHA to review.
    """
    db = ctx["db"]
    github_client = GitHubClient(
        app_id=ctx["app_id"],
        private_key=ctx["private_key"],
        installation_id=installation_id,
    )
    try:
        last_sha = await get_last_reviewed_sha(
            db, repo_full_name=repo_full_name, pr_number=pr_number
        )
        if last_sha == head_sha:
            logger.info(
                "Skipping already-reviewed SHA %s for %s#%d",
                head_sha,
                repo_full_name,
                pr_number,
            )
            return

        config = await _gate_review(
            github_client, repo_full_name=repo_full_name, pr_number=pr_number
        )
        if config is None:
            # Opt-in absent, scope filters excluded the PR, or the diff-size cap
            # fired (which already posted its own note) — skip cleanly, recording
            # no SHA (a later in-scope push still gets reviewed).
            return

        # Per-repo budget/rate guardrail: too many reviews in the rolling window
        # means skip this one (no SHA recorded, so a later push can still review).
        if await _over_rate_budget(db, repo_full_name=repo_full_name, caps=config.caps):
            logger.info(
                "Skipping review for %s#%d: per-repo rate/budget exceeded",
                repo_full_name,
                pr_number,
            )
            return

        # Per-installation concurrency guardrail: claim an in-flight slot, and if
        # the installation is already at its cap, defer this run (skip without
        # recording the SHA so a later delivery/push reviews the same commit).
        if not await try_acquire_inflight(
            db,
            installation_id=installation_id,
            cap=config.caps.max_concurrent_per_installation,
        ):
            logger.info(
                "Deferring review for %s#%d: installation %d at concurrency cap %d",
                repo_full_name,
                pr_number,
                installation_id,
                config.caps.max_concurrent_per_installation,
            )
            return
        # The slot is now held; release it on EVERY exit path (success, skip,
        # failure, exception) so the counter cannot leak.
        try:
            # Atomically reserve a rate slot. This is the authoritative race-free
            # guard (the _over_rate_budget check above is only a cheap fast-fail): a
            # concurrent review could have filled the window between that read and
            # here, so the slot is recorded only if still under the window budget.
            now = time.time()
            if not await try_record_review_event(
                db,
                repo_full_name=repo_full_name,
                occurred_at=now,
                cutoff=now - config.caps.rate_window_seconds,
                max_reviews=config.caps.max_reviews_per_window,
            ):
                logger.info(
                    "Skipping review for %s#%d: per-repo rate/budget exceeded (raced "
                    "past the fast-fail gate)",
                    repo_full_name,
                    pr_number,
                )
                return
            await _review_and_post(
                ctx,
                github_client,
                db,
                config=config,
                installation_id=installation_id,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                head_sha=head_sha,
            )
        finally:
            await release_inflight(db, installation_id=installation_id)
    finally:
        await github_client.aclose()


async def _over_rate_budget(
    db: Database,
    *,
    repo_full_name: str,
    caps: GuardrailCaps,
) -> bool:
    """Return True when the repo has hit its per-window review budget.

    Counts reviews recorded for the repo within the rolling window
    (``rate_window_seconds``); when that count is at or above
    ``max_reviews_per_window`` a fresh review is over budget and is skipped.
    Stale events outside the window are pruned first so the table stays bounded.
    """
    now = time.time()
    cutoff = now - caps.rate_window_seconds
    await prune_review_events(db, repo_full_name=repo_full_name, before=cutoff)
    recent = await count_recent_reviews(db, repo_full_name=repo_full_name, since=cutoff)
    return recent >= caps.max_reviews_per_window


async def _review_and_post(
    ctx: dict[str, Any],
    github_client: GitHubClient,
    db: Database,
    *,
    config: RepoConfig,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> None:
    """Run the retried pipeline and post exactly one review (or a failure note).

    The core of run_review, extracted so the per-installation concurrency
    acquire/release can wrap it in a clean try/finally.  Mirrors the prior inline
    body: on a None synthesis it posts a terse failure note and records the SHA;
    on success it retires the prior review, splits inline/body, posts once, and
    records both the posted-review and last-reviewed SHA.
    """
    synthesis = await _run_pipeline_with_retry(
        ctx,
        config=config,
        installation_id=installation_id,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        head_sha=head_sha,
    )
    if synthesis is None:
        # Every lens failed, synthesis aborted, or the retry timed out/failed.
        # Post a terse failure note and record the SHA so the failed commit is
        # not endlessly re-reviewed.
        await _post_review_failed_note(
            github_client,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
        )
        await set_last_reviewed_sha(
            db, repo_full_name=repo_full_name, pr_number=pr_number, sha=head_sha
        )
        return

    # Across-push lifecycle: retire the prior Heimdall review (dismiss a
    # REQUEST_CHANGES, minimize a COMMENT) and delete its now-stale inline
    # comments before posting, so only the latest review stays active.
    await _refresh_prior_review(
        github_client,
        db,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
    )
    # Anchor findings to changed diff lines: those on a changed line become
    # inline comments in the same submission; off-diff (or unparseable) ones
    # fall back to the review body.
    body, inline_comments = await _build_inline_split(
        github_client,
        synthesis,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
    )
    logger.info(
        "Posting %s review for %s#%d @ %s (%d inline comments)",
        synthesis.verdict,
        repo_full_name,
        pr_number,
        head_sha,
        len(inline_comments),
    )
    posted = await github_client.post_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        commit_id=head_sha,
        body=body,
        event=synthesis.verdict,
        comments=inline_comments,
    )
    await set_posted_review(
        db,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        review_id=int(posted["id"]),
        node_id=str(posted["node_id"]),
        verdict=synthesis.verdict,
    )
    await set_last_reviewed_sha(
        db, repo_full_name=repo_full_name, pr_number=pr_number, sha=head_sha
    )
    logger.info(
        "Review posted for %s#%d @ %s", repo_full_name, pr_number, head_sha
    )


async def _gate_review(
    github_client: GitHubClient,
    *,
    repo_full_name: str,
    pr_number: int,
) -> RepoConfig | None:
    """Early gate: enforce opt-in and scope filters before any review work.

    Fetches the PR object, loads ``.github/heimdall.yml`` from the trust-resolved
    ref (base for forks, head for trusted same-repo PRs), and applies the scope
    filters.  Returns the loaded config when the PR is in scope, or None to skip
    the review entirely.  A missing config file (no opt-in) or a malformed file
    both skip cleanly — Heimdall never reviews a repo that has not opted in, and a
    broken config must not crash the worker.

    Returns:
        The :class:`RepoConfig` to drive the pipeline, or None to skip the review.
    """
    pr = await github_client.get_pr(repo_full_name=repo_full_name, pr_number=pr_number)
    try:
        config = await load_repo_config(github_client, repo_full_name=repo_full_name, pr=pr)
    except RepoConfigError as exc:
        logger.warning(
            "Skipping review for %s#%d: invalid heimdall.yml: %s",
            repo_full_name,
            pr_number,
            exc,
        )
        return None
    if config is None:
        return None

    files = await github_client.get_pr_files(
        repo_full_name=repo_full_name, pr_number=pr_number
    )
    changed_paths = [str(f.get("filename")) for f in files if f.get("filename")]
    reason = skip_reason(config, pr=pr, changed_paths=changed_paths)
    if reason is not None:
        logger.info("Skipping review for %s#%d: %s", repo_full_name, pr_number, reason)
        return None

    # Diff-size/file-count guardrail: an oversized PR is skipped, but UNLIKE the
    # scope skips above it gets a POSTED note so the author learns why no review
    # came back.  Recording the SHA happens in the caller's normal skip handling.
    note = diff_cap_skip_note(
        config.caps,
        file_count=len(changed_paths),
        diff_lines=_diff_line_count(files),
    )
    if note is not None:
        logger.info(
            "Skipping review for %s#%d: over size cap; posting note", repo_full_name, pr_number
        )
        await github_client.post_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            commit_id=str((pr.get("head") or {}).get("sha", "")),
            body=note,
            event="COMMENT",
        )
        return None
    return config


def _diff_line_count(files: list[dict[str, Any]]) -> int:
    """Sum changed lines (additions + deletions) across a PR's file objects.

    GitHub's PR-files payload reports per-file ``additions`` and ``deletions``;
    their sum is the total churn the size cap is measured against.  Missing
    counters (e.g. a renamed-only entry) contribute zero.
    """
    total = 0
    for f in files:
        total += int(f.get("additions") or 0) + int(f.get("deletions") or 0)
    return total


async def _refresh_prior_review(
    github_client: GitHubClient,
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> None:
    """Retire the prior Heimdall review for a PR, if any, before a fresh post.

    Reads the prior posted-review record from SQLite and acts per its stored
    verdict: a REQUEST_CHANGES review is dismissed (it carries a blocking
    state), while a COMMENT review is minimized (dismissal is invalid for
    COMMENT events).  No-op when there is no prior review on record.
    """
    prior = await get_posted_review(
        db, repo_full_name=repo_full_name, pr_number=pr_number
    )
    if prior is None:
        return

    # Inline comments are review-comment objects separate from the review body, so
    # retiring the body does not remove them — delete the prior review's inline
    # comments explicitly so stale comments don't accumulate across pushes.
    await _delete_prior_inline_comments(
        github_client,
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        review_id=int(prior["review_id"]),
    )

    if prior["verdict"] == "REQUEST_CHANGES":
        await github_client.dismiss_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            review_id=int(prior["review_id"]),
            message="Superseded by a newer push; Heimdall re-reviewed the PR.",
        )
    else:
        await github_client.minimize_review(node_id=str(prior["node_id"]))


async def _delete_prior_inline_comments(
    github_client: GitHubClient,
    *,
    repo_full_name: str,
    pr_number: int,
    review_id: int,
) -> None:
    """Delete every inline comment attached to a prior review.

    Lists the prior review's inline comments by its REST id and deletes each one,
    so the fresh push starts from a clean slate of inline comments.  A failure to
    delete an individual comment is logged and skipped rather than aborting the
    post — a leftover stale comment is preferable to a missing fresh review.
    """
    comments = await github_client.list_review_comments(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        review_id=review_id,
    )
    for comment in comments:
        await github_client.delete_review_comment(
            repo_full_name=repo_full_name,
            comment_id=int(comment["id"]),
        )


async def _build_inline_split(
    github_client: GitHubClient,
    synthesis: SynthesisResult,
    *,
    repo_full_name: str,
    pr_number: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Split synthesis findings into a body + inline-comment array for the post.

    Fetches the PR's unified diff, parses the lines a comment can anchor to, then
    routes each survivor: findings on a changed line become inline comments
    attached to the same review submission; off-diff or unparseable-location
    findings are rendered into the body.

    Returns:
        ``(body, comments)`` ready for :meth:`GitHubClient.post_review`.
    """
    diff = await github_client.get_pr_diff(
        repo_full_name=repo_full_name, pr_number=pr_number
    )
    commentable = commentable_lines(diff)
    inline, body_findings = split_findings(synthesis.tagged_findings, commentable)
    body = render_body_for_offdiff(body_findings)
    return body, build_inline_comments(inline)


async def _run_pipeline_with_retry(
    ctx: dict[str, Any],
    *,
    config: RepoConfig,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> SynthesisResult | None:
    """Run the review pipeline under a per-review timeout, retrying once on failure.

    Wraps :func:`_synthesize_review` in :func:`asyncio.wait_for` to enforce the
    per-review wall-clock budget (separate from, and looser than, the per-lens
    timeout that bounds each lens inside the fanout), then retries the whole wrapped
    pipeline exactly once on any failure/timeout.  The retry seam sits outside the
    pipeline body so the lens-fanout + synthesis restructure lives wholly inside
    :func:`_synthesize_review` without touching this wrapper.

    The inner pipeline already isolates per-lens failures and returns None when no
    lens survives or synthesis aborts; that None is treated like any other failed
    attempt here, so a fully-failed pipeline is also retried once before the caller
    posts the failure note.

    Returns:
        The :class:`SynthesisResult` on success, or None when both the initial run
        and the single retry fail (all lenses/synthesis aborted or per-review
        timeout).
    """
    review_timeout = ctx.get(
        "review_timeout_seconds", DEFAULT_REVIEW_TIMEOUT_SECONDS
    )
    max_attempts = 2  # one initial attempt + exactly one retry
    for attempt in range(1, max_attempts + 1):
        try:
            synthesis = await asyncio.wait_for(
                _synthesize_review(
                    ctx,
                    config=config,
                    installation_id=installation_id,
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                ),
                timeout=review_timeout,
            )
        except (LensError, TimeoutError) as exc:
            # Metadata-only: log the failure class and identifiers, never the
            # underlying findings/code or any secret.
            logger.warning(
                "Review pipeline attempt %d/%d failed for %s#%d @ %s: %s",
                attempt,
                max_attempts,
                repo_full_name,
                pr_number,
                head_sha,
                type(exc).__name__,
            )
            continue
        if synthesis is not None:
            return synthesis
        # Inner pipeline returned None (all lenses failed or synthesis aborted);
        # treat it as a failed attempt and retry once.
        logger.warning(
            "Review pipeline attempt %d/%d produced no review for %s#%d @ %s",
            attempt,
            max_attempts,
            repo_full_name,
            pr_number,
            head_sha,
        )
    logger.warning(
        "Review pipeline failed after retry for %s#%d @ %s; posting failure note",
        repo_full_name,
        pr_number,
        head_sha,
    )
    return None


async def _synthesize_review(
    ctx: dict[str, Any],
    *,
    config: RepoConfig,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
) -> SynthesisResult | None:
    """Assemble the seed, fan out the lenses, and synthesize the final review.

    The inner review pipeline core — wrapped by :func:`_run_pipeline_with_retry`
    for retry-once + per-review timeout.  The seed context is materialized into a
    temporary workspace once and shared by every lens (read via the heimdall-context
    wrapper); the workspace is removed afterwards.  Returns None when no lens
    produced a result (all failed) or when the synthesis pass aborts — the wrapper
    then retries, and the caller posts the failure note if the retry also fails.

    Returns:
        The :class:`SynthesisResult` (tagged survivors, verdict, body), or None.
    """
    workspace = tempfile.mkdtemp(prefix="heimdall-lens-")
    try:
        await assemble_pr_context(
            app_id=ctx["app_id"],
            private_key=ctx["private_key"],
            installation_id=installation_id,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            workspace_dir=workspace,
        )

        lens_results = await _run_lenses(
            ctx,
            config=config,
            workspace_dir=workspace,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )
        if not lens_results:
            logger.warning(
                "All lenses failed for %s#%d; skipping review", repo_full_name, pr_number
            )
            return None

        synthesis = await run_synthesis(
            lens_results=lens_results,
            workspace_dir=workspace,
            claude_binary=ctx.get("claude_binary", "claude"),
            token_cap=ctx.get("lens_token_cap", DEFAULT_TOKEN_CAP),
            timeout_seconds=ctx.get("lens_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            env_passthrough=ctx.get("claude_env_passthrough", []),
            blocking=blocking_severities(config.severity_threshold),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    # Metadata-only by default; the synthesized body (findings + code text) is
    # logged only under the debug-logging flag.
    _log_findings(
        ctx, repo_full_name=repo_full_name, pr_number=pr_number, body=synthesis.body
    )
    return synthesis


async def _run_lenses(
    ctx: dict[str, Any],
    *,
    config: RepoConfig,
    workspace_dir: str,
    repo_full_name: str,
    pr_number: int,
) -> list[LensResult]:
    """Run the config-tuned lenses over the shared workspace, isolating failures.

    The repo config decides which lenses run and with what model/effort/prompt: a
    disabled built-in never runs, a built-in may carry appended per-lens instructions,
    and any custom lenses defined in the config run here too (all via tuned_lenses).
    A disabled lens never reaches synthesis.  Each surviving lens is bounded
    independently (its own token cap + timeout via run_lens).  A lens that aborts
    (timeout or token-cap breach) is logged and dropped so the remaining lenses
    still reach synthesis; an unexpected error in one lens is likewise contained
    rather than crashing the whole run.

    Returns:
        The results of the lenses that succeeded (possibly empty if all failed).
    """
    lenses = tuned_lenses(config)
    outcomes = await asyncio.gather(
        *(
            run_lens(
                lens=lens,
                workspace_dir=workspace_dir,
                claude_binary=ctx.get("claude_binary", "claude"),
                token_cap=ctx.get("lens_token_cap", DEFAULT_TOKEN_CAP),
                timeout_seconds=ctx.get("lens_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
                env_passthrough=ctx.get("claude_env_passthrough", []),
            )
            for lens in lenses
        ),
        return_exceptions=True,
    )

    results: list[LensResult] = []
    for lens, outcome in zip(lenses, outcomes, strict=True):
        if isinstance(outcome, LensResult):
            results.append(outcome)
        elif isinstance(outcome, BaseException):
            logger.warning(
                "Lens %s failed for %s#%d; dropping it from synthesis: %s",
                lens.name,
                repo_full_name,
                pr_number,
                outcome,
            )
    return results


def _log_findings(
    ctx: dict[str, Any],
    *,
    repo_full_name: str,
    pr_number: int,
    body: str,
) -> None:
    """Log the rendered review body only under the DEBUG-logging flag.

    The body carries findings and code-snippet text, so it is emitted only when
    ``ctx['debug_logging']`` is truthy.  Default (metadata-only) logging never
    sees it.
    """
    if ctx.get("debug_logging"):
        logger.debug(
            "Review body for %s#%d:\n%s", repo_full_name, pr_number, body
        )


async def _post_review_failed_note(
    github_client: GitHubClient,
    *,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> None:
    """Post the terse "review failed" COMMENT note after the retry also failed.

    Deliberately a COMMENT (never REQUEST_CHANGES): a pipeline failure is not a
    verdict on the code, so it must not block the PR.
    """
    logger.info(
        "Posting review-failed note for %s#%d @ %s",
        repo_full_name,
        pr_number,
        head_sha,
    )
    await github_client.post_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        commit_id=head_sha,
        body=_REVIEW_FAILED_NOTE,
        event="COMMENT",
    )


def _load_settings() -> Any:
    """Load Settings lazily, allowing tests to patch before first access."""
    from heimdall.config import Settings

    return Settings()  # type: ignore[call-arg]


# Module-level settings instance, imported lazily in on_startup so that
# tests can patch 'heimdall.worker.settings' without triggering env-var
# validation at import time.
settings: Any = None


def main() -> None:
    """Console-script entrypoint: start the Arq worker with WorkerSettings.

    Invoked as ``heimdall-worker`` (see [project.scripts] in pyproject.toml)
    or directly with ``python -m heimdall.worker``.
    """
    from arq.worker import run_worker

    run_worker(WorkerSettings)  # type: ignore[arg-type]


class WorkerSettings:
    """Arq WorkerSettings: registers run_review and wires startup/shutdown.

    Launch the worker process with:
        arq heimdall.worker.WorkerSettings
    """

    functions = [run_review]
    # RedisSettings is initialised from env at worker-launch time via on_startup;
    # the default here points to localhost so the class attribute is always a
    # valid RedisSettings instance (Arq will use it if not overridden).
    redis_settings: RedisSettings = RedisSettings()

    @staticmethod
    async def on_startup(ctx: dict[str, Any]) -> None:
        """Open the database and store app credentials in ctx.

        Reads Settings from the environment, overrides redis_settings on the
        class, then populates ctx with:
            db:                     initialised Database instance
            app_id:                 GitHub App numeric ID
            private_key:            PEM-encoded RSA private key
            claude_binary:          path/name of the claude CLI
            claude_env_passthrough: extra env keys forwarded to the claude child
            lens_token_cap:         per-agent cumulative-token cap
            lens_timeout_seconds:   per-lens wall-clock timeout
            review_timeout_seconds: per-review wall-clock timeout (pipeline-wide)
            debug_logging:          log findings/code text when True
        """
        global settings
        if settings is None:
            settings = _load_settings()

        # Update redis_settings from the live config so the running worker uses
        # the correct Redis URL even if the default was overridden in .env.
        WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)

        db = Database(_db_path_from_url(settings.database_url))
        await db.initialize()
        ctx["db"] = db
        ctx["app_id"] = settings.github_app_id
        ctx["private_key"] = settings.github_app_private_key
        ctx["claude_binary"] = settings.claude_binary
        ctx["claude_env_passthrough"] = settings.claude_env_passthrough
        ctx["lens_token_cap"] = settings.lens_token_cap
        ctx["lens_timeout_seconds"] = settings.lens_timeout_seconds
        ctx["review_timeout_seconds"] = settings.review_timeout_seconds
        ctx["debug_logging"] = settings.debug_logging

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        """Close the database connection."""
        db: Database = ctx["db"]
        await db.close()
