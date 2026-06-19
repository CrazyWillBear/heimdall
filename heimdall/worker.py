"""Arq worker: the run_review task function and WorkerSettings.

Context keys populated by WorkerSettings.on_startup:
    db:                   heimdall.db.Database instance
    app_id:               GitHub App numeric ID (int)
    private_key:          PEM-encoded RSA private key (str)
    claude_binary:        path/name of the claude CLI (str)
    lens_token_cap:       per-agent cumulative-token cap (int)
    lens_timeout_seconds: wall-clock timeout for a lens run (float)

run_review builds a GitHubClient per-job using ctx["app_id"], ctx["private_key"],
and the per-job installation_id argument.  It assembles the PR seed context into
a temporary workspace once, fans out three independent lenses (Security opus/max,
Design-fit sonnet/high, Cleanliness sonnet/high) over that shared seed — each
bounded by its own token cap and timeout — then runs a 4th synthesis ``claude -p``
pass that dedups overlapping findings across lenses, ranks by severity, writes the
verdict, and formats the review (findings grouped by severity, each tagged with the
originating lens).  Exactly one PR review is posted.  A failure in any single lens
is isolated (logged, that lens dropped); the pipeline only skips posting when every
lens fails or the synthesis pass itself aborts.  Nothing here ever crashes the worker.

Launch the worker with:
    arq heimdall.worker.WorkerSettings
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from typing import Any

from arq.connections import RedisSettings

from heimdall.context import assemble_pr_context
from heimdall.db import Database, get_last_reviewed_sha, set_last_reviewed_sha
from heimdall.github import GitHubClient
from heimdall.lens import (
    CLEANLINESS_LENS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_CAP,
    DESIGN_LENS,
    SECURITY_LENS,
    LensError,
    LensResult,
    LensSpec,
    SynthesisResult,
    run_lens,
    run_synthesis,
)

logger = logging.getLogger(__name__)

# The three review lenses fanned out over the shared seed. Order is stable so the
# synthesized review is deterministic; each runs independently bounded.
_LENSES: tuple[LensSpec, ...] = (SECURITY_LENS, DESIGN_LENS, CLEANLINESS_LENS)


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
    fresh SHA it assembles the seed context once, runs the three lenses over it
    (each independently bounded; a single lens failure is isolated), then runs the
    synthesis pass that dedups overlapping findings, ranks by severity, writes the
    verdict (REQUEST_CHANGES for a high/critical survivor, else COMMENT), and
    formats the severity-grouped, lens-tagged review body.  Exactly one PR review
    is posted and the SHA recorded.  When every lens fails, or the synthesis pass
    aborts, the job returns without posting.

    A GitHubClient is constructed per-job from the app credentials in ctx so
    that each job can target a different GitHub App installation.

    Args:
        ctx: Arq worker context carrying ``db``, ``app_id``, ``private_key``,
            and the optional lens knobs.
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

        synthesis = await _synthesize_review(
            ctx,
            installation_id=installation_id,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )
        if synthesis is None:
            return  # Pipeline failed; failure already logged. Do not post or record SHA.

        logger.info(
            "Posting %s review for %s#%d @ %s",
            synthesis.verdict,
            repo_full_name,
            pr_number,
            head_sha,
        )
        await github_client.post_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            commit_id=head_sha,
            body=synthesis.body,
            event=synthesis.verdict,
        )
        await set_last_reviewed_sha(
            db, repo_full_name=repo_full_name, pr_number=pr_number, sha=head_sha
        )
        logger.info(
            "Review posted for %s#%d @ %s", repo_full_name, pr_number, head_sha
        )
    finally:
        await github_client.aclose()


async def _synthesize_review(
    ctx: dict[str, Any],
    *,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
) -> SynthesisResult | None:
    """Assemble the seed, fan out the lenses, and synthesize the final review.

    The seed context is materialized into a temporary workspace once and shared by
    every lens (read via the heimdall-context wrapper); the workspace is removed
    afterwards.  Returns None when no lens produced a result (all failed) or when
    the synthesis pass aborts — the caller then skips posting.

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
            workspace_dir=workspace,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )
        if not lens_results:
            logger.warning(
                "All lenses failed for %s#%d; skipping review", repo_full_name, pr_number
            )
            return None

        try:
            return await run_synthesis(
                lens_results=lens_results,
                workspace_dir=workspace,
                claude_binary=ctx.get("claude_binary", "claude"),
                token_cap=ctx.get("lens_token_cap", DEFAULT_TOKEN_CAP),
                timeout_seconds=ctx.get("lens_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            )
        except LensError:
            logger.exception(
                "Synthesis aborted for %s#%d; skipping review",
                repo_full_name,
                pr_number,
            )
            return None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def _run_lenses(
    ctx: dict[str, Any],
    *,
    workspace_dir: str,
    repo_full_name: str,
    pr_number: int,
) -> list[LensResult]:
    """Run every lens over the shared workspace, isolating per-lens failures.

    Each lens is bounded independently (its own token cap + timeout via run_lens).
    A lens that aborts (timeout or token-cap breach) is logged and dropped so the
    remaining lenses still reach synthesis; an unexpected error in one lens is
    likewise contained rather than crashing the whole run.

    Returns:
        The results of the lenses that succeeded (possibly empty if all failed).
    """
    outcomes = await asyncio.gather(
        *(
            run_lens(
                lens=lens,
                workspace_dir=workspace_dir,
                claude_binary=ctx.get("claude_binary", "claude"),
                token_cap=ctx.get("lens_token_cap", DEFAULT_TOKEN_CAP),
                timeout_seconds=ctx.get("lens_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            )
            for lens in _LENSES
        ),
        return_exceptions=True,
    )

    results: list[LensResult] = []
    for lens, outcome in zip(_LENSES, outcomes, strict=True):
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
            db:                   initialised Database instance
            app_id:               GitHub App numeric ID
            private_key:          PEM-encoded RSA private key
            claude_binary:        path/name of the claude CLI
            lens_token_cap:       per-agent cumulative-token cap
            lens_timeout_seconds: wall-clock timeout for a lens run
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
        ctx["lens_token_cap"] = settings.lens_token_cap
        ctx["lens_timeout_seconds"] = settings.lens_timeout_seconds

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        """Close the database connection."""
        db: Database = ctx["db"]
        await db.close()
