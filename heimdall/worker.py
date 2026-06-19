"""Arq worker: the run_review task function and WorkerSettings.

Context keys populated by WorkerSettings.on_startup:
    db:          heimdall.db.Database instance
    app_id:      GitHub App numeric ID (int)
    private_key: PEM-encoded RSA private key (str)

run_review builds a GitHubClient per-job using ctx["app_id"], ctx["private_key"],
and the per-job installation_id argument.  This sidesteps the single-installation
constraint of GitHubClient.__init__ and is consistent with future issues (#3, #4)
where each job may target a different installation.

Launch the worker with:
    arq heimdall.worker.WorkerSettings
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import RedisSettings

from heimdall.db import Database, get_last_reviewed_sha, set_last_reviewed_sha
from heimdall.github import GitHubClient

logger = logging.getLogger(__name__)

_REVIEW_BODY = "Heimdall received this PR"


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
    """Arq task: post a hardcoded review comment for the given PR.

    Skips if the same head SHA was already reviewed (idempotency guard).
    Records the head SHA after a successful review so it is not re-posted.

    A GitHubClient is constructed per-job from the app credentials in ctx so
    that each job can target a different GitHub App installation.

    Args:
        ctx: Arq worker context carrying ``db``, ``app_id``, and ``private_key``.
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

        logger.info(
            "Posting review for %s#%d @ %s", repo_full_name, pr_number, head_sha
        )
        await github_client.post_review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            commit_id=head_sha,
            body=_REVIEW_BODY,
            event="COMMENT",
        )
        await set_last_reviewed_sha(
            db, repo_full_name=repo_full_name, pr_number=pr_number, sha=head_sha
        )
        logger.info(
            "Review posted for %s#%d @ %s", repo_full_name, pr_number, head_sha
        )
    finally:
        await github_client.aclose()


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
            db:          initialised Database instance
            app_id:      GitHub App numeric ID
            private_key: PEM-encoded RSA private key
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

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        """Close the database connection."""
        db: Database = ctx["db"]
        await db.close()
