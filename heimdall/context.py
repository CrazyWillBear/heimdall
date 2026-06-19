"""PR seed-context assembly.

Fetches the unified diff, changed-file contents, and PR metadata from GitHub
via the REST API (no code execution), then materializes the result to a
caller-supplied or temporary workspace directory on disk.

The assembled PRContext is the shared seed that downstream review lenses consume.
No PR build steps, tests, or scripts are ever run — all data comes from GitHub API
calls only.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from heimdall.github import GitHubClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PRContext:
    """Immutable seed context assembled from a pull request.

    Attributes:
        repo_full_name: e.g. "owner/repo".
        pr_number: The pull-request number.
        title: PR title.
        body: PR description body.
        author: GitHub login of the PR author.
        base_sha: Commit SHA of the base branch tip.
        head_sha: Commit SHA of the PR head.
        base_ref: Base branch name (e.g. "main").
        head_ref: Head branch name.
        linked_issues: Issues referenced by this PR (may be empty).
        diff: Unified diff of all changes.
        changed_files: List of file-change objects from the GitHub API.
        file_contents: Map from file path to full decoded file content at head_sha.
        convention_docs: Map from doc name to text (e.g. STYLEGUIDE.md content).
    """

    repo_full_name: str
    pr_number: int
    title: str
    body: str
    author: str
    base_sha: str
    head_sha: str
    base_ref: str
    head_ref: str
    linked_issues: list[dict[str, Any]]
    diff: str
    changed_files: list[dict[str, Any]]
    file_contents: dict[str, str]
    convention_docs: dict[str, str]


async def assemble_pr_context(
    *,
    app_id: int,
    private_key: str,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    workspace_dir: str | None = None,
) -> PRContext:
    """Assemble the seed context for a pull request.

    Fetches PR metadata, the unified diff, changed-file list, and full contents
    of each changed file from GitHub.  No PR code is executed; all data comes
    from the GitHub REST API.

    If ``workspace_dir`` is provided the assembled context is also materialized
    to disk there (diff.patch, pr_metadata.json, files/<path>).  When omitted a
    temporary directory is created, the context is materialized into it, and the
    directory is cleaned up before this function returns.

    Args:
        app_id: The GitHub App numeric ID.
        private_key: PEM-encoded RSA private key.
        installation_id: The App installation ID.
        repo_full_name: e.g. "owner/repo".
        pr_number: The pull-request number.
        workspace_dir: Optional caller-managed directory to materialize into.

    Returns:
        The assembled PRContext with all seed fields populated.
    """
    github = GitHubClient(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )
    try:
        pr_meta, diff, files, linked = await _fetch_pr_data(
            github, repo_full_name=repo_full_name, pr_number=pr_number
        )
        file_contents = await _fetch_file_contents(
            github,
            repo_full_name=repo_full_name,
            files=files,
            ref=pr_meta["head"]["sha"],
        )
    finally:
        await github.aclose()

    ctx = PRContext(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        title=pr_meta["title"],
        body=pr_meta.get("body") or "",
        author=pr_meta["user"]["login"],
        base_sha=pr_meta["base"]["sha"],
        head_sha=pr_meta["head"]["sha"],
        base_ref=pr_meta["base"]["ref"],
        head_ref=pr_meta["head"]["ref"],
        linked_issues=linked,
        diff=diff,
        changed_files=files,
        file_contents=file_contents,
        convention_docs={},
    )

    if workspace_dir is not None:
        _materialize(ctx, workspace_dir)
    else:
        # Create a temp workspace, materialize, then clean up.
        tmp = tempfile.mkdtemp(prefix="heimdall-pr-")
        try:
            _materialize(ctx, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return ctx


async def _fetch_pr_data(
    github: GitHubClient,
    *,
    repo_full_name: str,
    pr_number: int,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch PR metadata, diff, file list, and linked issues in parallel.

    Returns:
        (pr_metadata, diff_text, changed_files, linked_issues)
    """
    import asyncio

    pr_meta_task = asyncio.create_task(
        github.get_pr(repo_full_name=repo_full_name, pr_number=pr_number)
    )
    diff_task = asyncio.create_task(
        github.get_pr_diff(repo_full_name=repo_full_name, pr_number=pr_number)
    )
    files_task = asyncio.create_task(
        github.get_pr_files(repo_full_name=repo_full_name, pr_number=pr_number)
    )
    linked_task = asyncio.create_task(
        github.get_linked_issues(repo_full_name=repo_full_name, pr_number=pr_number)
    )
    pr_meta, diff, files, linked = await asyncio.gather(
        pr_meta_task, diff_task, files_task, linked_task
    )
    return pr_meta, diff, files, linked


async def _fetch_file_contents(
    github: GitHubClient,
    *,
    repo_full_name: str,
    files: list[dict[str, Any]],
    ref: str,
) -> dict[str, str]:
    """Fetch full contents for all non-deleted changed files.

    Files with status "removed" are skipped — we only read files that exist at
    ``ref``.  Per-file failures are isolated: binary files (UnicodeDecodeError)
    and oversize/unavailable files (HTTPStatusError) are logged and omitted from
    the result rather than aborting the whole gather.

    Returns:
        Dict mapping filename → decoded text content, with unreadable files omitted.
    """
    import asyncio

    # Only fetch files that exist at the head ref; skip deletions.
    fetchable = [f for f in files if f.get("status") != "removed"]

    async def _fetch_one(filename: str) -> tuple[str, str]:
        content = await github.get_file_content(
            repo_full_name=repo_full_name,
            path=filename,
            ref=ref,
        )
        return filename, content

    # return_exceptions=True prevents one bad file from aborting the gather.
    raw_results = await asyncio.gather(
        *(_fetch_one(f["filename"]) for f in fetchable),
        return_exceptions=True,
    )

    contents: dict[str, str] = {}
    for item in raw_results:
        if isinstance(item, UnicodeDecodeError):
            # Binary file — can't decode as UTF-8; skip with a log entry.
            logger.warning(
                "Skipping binary file in PR context (UnicodeDecodeError): %s", item
            )
        elif isinstance(item, httpx.HTTPStatusError):
            # Oversize or unavailable file — GitHub Contents API returned an error.
            logger.warning(
                "Skipping file in PR context (HTTP %s): %s",
                item.response.status_code,
                item.request.url,
            )
        elif isinstance(item, BaseException):
            # Unexpected error — re-raise so it isn't silently swallowed.
            raise item
        else:
            filename, content = item
            contents[filename] = content

    return contents


def _materialize(ctx: PRContext, directory: str) -> None:
    """Write the assembled context to disk in ``directory``.

    Produces:
      <directory>/diff.patch          — unified diff
      <directory>/pr_metadata.json    — PR metadata as JSON
      <directory>/files/<path>        — full content of each changed file
    """
    root = Path(directory)

    (root / "diff.patch").write_text(ctx.diff, encoding="utf-8")

    metadata = {
        "repo_full_name": ctx.repo_full_name,
        "pr_number": ctx.pr_number,
        "title": ctx.title,
        "body": ctx.body,
        "author": ctx.author,
        "base_sha": ctx.base_sha,
        "head_sha": ctx.head_sha,
        "base_ref": ctx.base_ref,
        "head_ref": ctx.head_ref,
        "linked_issues": ctx.linked_issues,
    }
    (root / "pr_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    files_root = root / "files"
    files_root.mkdir(exist_ok=True)
    for filename, content in ctx.file_contents.items():
        file_path = files_root / Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
