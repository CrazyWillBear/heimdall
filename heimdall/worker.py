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
a temporary workspace, runs the Security lens (``claude -p``) over it, maps the
findings to a verdict, and posts exactly one PR review.  A lens failure (timeout
or token-cap abort) is logged and the job ends without posting — it never crashes
the worker.

Launch the worker with:
    arq heimdall.worker.WorkerSettings
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from typing import Any

from arq.connections import RedisSettings

from heimdall.context import assemble_pr_context
from heimdall.db import (
    Database,
    get_last_reviewed_sha,
    get_posted_review,
    set_last_reviewed_sha,
    set_posted_review,
)
from heimdall.github import GitHubClient
from heimdall.lens import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_CAP,
    SECURITY_LENS,
    LensError,
    format_review_body,
    run_lens,
    verdict_for,
)

logger = logging.getLogger(__name__)


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
    """Arq task: run the Security lens over the PR and post one review.

    Skips if the same head SHA was already reviewed (idempotency guard).  On a
    fresh SHA it assembles the seed context, runs the Security lens, maps the
    findings to a verdict (REQUEST_CHANGES for any high/critical finding, else
    COMMENT), posts exactly one PR review, and records the SHA.  A lens failure
    (timeout or token-cap abort) is logged and the job returns without posting.

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

        review = await _run_security_lens(
            ctx,
            installation_id=installation_id,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )
        if review is None:
            return  # Lens failed; failure already logged. Do not post or record SHA.

        body, event = review
        # Across-push lifecycle: retire the prior Heimdall review (dismiss a
        # REQUEST_CHANGES, minimize a COMMENT) before posting so only the latest
        # review stays active.
        await _refresh_prior_review(
            github_client,
            db,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
        )
        logger.info(
            "Posting %s review for %s#%d @ %s",
            event,
            repo_full_name,
            pr_number,
            head_sha,
        )
        posted = await github_client.post_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            commit_id=head_sha,
            body=body,
            event=event,
        )
        await set_posted_review(
            db,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            review_id=int(posted["id"]),
            node_id=str(posted["node_id"]),
            verdict=event,
        )
        await set_last_reviewed_sha(
            db, repo_full_name=repo_full_name, pr_number=pr_number, sha=head_sha
        )
        logger.info(
            "Review posted for %s#%d @ %s", repo_full_name, pr_number, head_sha
        )
    finally:
        await github_client.aclose()


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

    if prior["verdict"] == "REQUEST_CHANGES":
        await github_client.dismiss_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            review_id=int(prior["review_id"]),
            message="Superseded by a newer push; Heimdall re-reviewed the PR.",
        )
    else:
        await github_client.minimize_review(node_id=str(prior["node_id"]))


async def _run_security_lens(
    ctx: dict[str, Any],
    *,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
) -> tuple[str, str] | None:
    """Assemble seed context and run the Security lens, returning (body, event).

    The seed context is materialized into a temporary workspace that the lens
    reads via the heimdall-context wrapper; the workspace is removed afterwards.

    Returns:
        A ``(review_body, review_event)`` tuple, or None when the lens aborts
        (timeout or token-cap breach) — the caller then skips posting.
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
        result = await run_lens(
            lens=SECURITY_LENS,
            workspace_dir=workspace,
            claude_binary=ctx.get("claude_binary", "claude"),
            token_cap=ctx.get("lens_token_cap", DEFAULT_TOKEN_CAP),
            timeout_seconds=ctx.get("lens_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        )
    except LensError:
        logger.exception(
            "Security lens aborted for %s#%d; skipping review",
            repo_full_name,
            pr_number,
        )
        return None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return format_review_body(result.findings), verdict_for(result.findings)


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
