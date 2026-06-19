"""Arq worker: the run_review task function.

Context keys expected in ctx (populated by WorkerSettings.on_startup):
    db:            heimdall.db.Database instance
    github_client: heimdall.github.GitHubClient instance

The worker posts a single hardcoded COMMENT review to signal that Heimdall has
received the PR.  Future issues will replace this with the real 3-lens analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from heimdall.db import get_last_reviewed_sha, set_last_reviewed_sha

logger = logging.getLogger(__name__)

_REVIEW_BODY = "Heimdall received this PR"


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

    Args:
        ctx: Arq worker context carrying ``db`` and ``github_client``.
        installation_id: GitHub App installation ID (unused here, for future use).
        repo_full_name: e.g. "owner/repo".
        pr_number: The pull-request number.
        head_sha: The commit SHA to review.
    """
    db = ctx["db"]
    github_client = ctx["github_client"]

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

    logger.info("Posting review for %s#%d @ %s", repo_full_name, pr_number, head_sha)
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
    logger.info("Review posted for %s#%d @ %s", repo_full_name, pr_number, head_sha)
