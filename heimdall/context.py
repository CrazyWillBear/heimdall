"""PR seed-context assembly.

Fetches the unified diff, changed-file contents, PR metadata, and the PR's
conversation (timeline) comments from GitHub via the REST API (no code execution),
then materializes the result to a caller-supplied or temporary workspace directory
on disk.

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
from heimdall.repo_config import _DEFAULT_DOCS

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
        docs: Map from doc name to text (e.g. STYLEGUIDE.md content).
        comments: Kept conversation (timeline) comments — human and Heimdall's own —
            each with its ``body``, ``author`` login, and ``author_association``.
            Untrusted third-party data, never instructions.  May be empty.
        review_threads: Kept inline review comments grouped into parent-anchored reply
            threads — each thread carries its ``body``, ``author``, ``author_association``,
            ``path``/``line`` anchor, a ``replies`` list, and an ``is_resolved`` flag
            (its resolution state from the GraphQL ``reviewThreads`` signal, defaulting to
            ``False`` when unknown).  Same author filter and untrusted-data posture as
            ``comments``.  May be empty.
        review_summaries: Kept submitted-review summary bodies — human and Heimdall's own —
            each with its ``body``, ``author`` login, ``author_association``, and ``event``
            (APPROVE/REQUEST_CHANGES/COMMENT).  Same author filter and untrusted-data
            posture as ``comments``.  May be empty.
        own_prior_review: Heimdall's own latest prior review on this PR — its ``body``,
            ``author``, ``author_association``, ``event``, and an ``inline_comments`` list
            — fetched BEFORE the across-push retire/delete step destroys it.  None when
            Heimdall has not reviewed this PR yet.  Untrusted-self continuity context,
            never an instruction.
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
    docs: dict[str, str]
    comments: list[dict[str, Any]]
    review_threads: list[dict[str, Any]]
    review_summaries: list[dict[str, Any]]
    own_prior_review: dict[str, Any] | None


async def assemble_pr_context(
    *,
    app_id: int,
    private_key: str,
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    workspace_dir: str | None = None,
    docs: list[str] | None = None,
) -> PRContext:
    """Assemble the seed context for a pull request.

    Fetches PR metadata, the unified diff, changed-file list, and full contents
    of each changed file from GitHub.  No PR code is executed; all data comes
    from the GitHub REST API.

    If ``workspace_dir`` is provided the assembled context is also materialized
    to disk there (diff.patch, pr_metadata.json, files/<path>, comments.json when any
    conversation comments were kept, review_threads.json when any inline review threads
    were kept, review_summaries.json when any submitted-review summaries were kept, and
    own_prior_review.json when Heimdall has a prior review on the PR).  When omitted a
    temporary directory is created, the context is materialized into it, and the
    directory is cleaned up before this function returns.

    Args:
        app_id: The GitHub App numeric ID.
        private_key: PEM-encoded RSA private key.
        installation_id: The App installation ID.
        repo_full_name: e.g. "owner/repo".
        pr_number: The pull-request number.
        workspace_dir: Optional caller-managed directory to materialize into.
        docs: Repo-relative doc paths to fetch from the PR head (tolerate-missing).
            ``None`` uses the four built-in defaults; ``[]`` fetches no docs.  The
            worker passes the loaded ``config.docs`` (validated, from the trusted
            ref) so the list is trusted even though contents come from the head.

    Returns:
        The assembled PRContext with all seed fields populated.
    """
    doc_names = list(_DEFAULT_DOCS) if docs is None else docs
    github = GitHubClient(
        app_id=app_id,
        private_key=private_key,
        installation_id=installation_id,
    )
    try:
        (
            pr_meta,
            diff,
            files,
            linked,
            comments,
            review_threads,
            review_summaries,
            own_prior_review,
        ) = await _fetch_pr_data(
            github, repo_full_name=repo_full_name, pr_number=pr_number
        )
        head_sha = pr_meta["head"]["sha"]
        file_contents, fetched_docs = await _fetch_file_contents_and_docs(
            github,
            repo_full_name=repo_full_name,
            files=files,
            ref=head_sha,
            doc_names=doc_names,
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
        head_sha=head_sha,
        base_ref=pr_meta["base"]["ref"],
        head_ref=pr_meta["head"]["ref"],
        linked_issues=linked,
        diff=diff,
        changed_files=files,
        file_contents=file_contents,
        docs=fetched_docs,
        comments=comments,
        review_threads=review_threads,
        review_summaries=review_summaries,
        own_prior_review=own_prior_review,
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
) -> tuple[
    dict[str, Any],
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Fetch PR metadata, diff, files, linked issues, and all comment sources in parallel.

    Heimdall's own prior review is fetched HERE (during seed assembly) so it is read
    BEFORE the across-push retire/delete step in the worker dismisses/minimizes that
    review and deletes its inline comments — assembly always precedes that step, so the
    context is captured before it is destroyed.

    Returns:
        (pr_metadata, diff_text, changed_files, linked_issues, conversation_comments,
        review_threads, review_summaries, own_prior_review)
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
    comments_task = asyncio.create_task(
        github.get_pr_conversation_comments(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
    )
    review_threads_task = asyncio.create_task(
        github.get_pr_review_comments(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
    )
    review_summaries_task = asyncio.create_task(
        github.get_pr_review_summaries(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
    )
    own_prior_task = asyncio.create_task(
        github.get_own_prior_review(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
    )
    # All tasks are already scheduled (create_task) so they run concurrently.  Gather in
    # two groups of <= six: asyncio.gather's typed overloads stop at six awaitables and
    # would erase the per-task result types past that arity, so a single eight-way gather
    # loses precise typing.  Both groups await tasks that are already running, so the
    # concurrency is unchanged; the first group's exception still propagates first.
    pr_meta, diff, files, linked = await asyncio.gather(
        pr_meta_task,
        diff_task,
        files_task,
        linked_task,
    )
    comments, review_threads, review_summaries, own_prior_review = await asyncio.gather(
        comments_task,
        review_threads_task,
        review_summaries_task,
        own_prior_task,
    )
    return (
        pr_meta,
        diff,
        files,
        linked,
        comments,
        review_threads,
        review_summaries,
        own_prior_review,
    )


async def _fetch_file_contents_and_docs(
    github: GitHubClient,
    *,
    repo_full_name: str,
    files: list[dict[str, Any]],
    ref: str,
    doc_names: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch changed-file contents and repo docs in parallel.

    Changed files with status "removed" are skipped.  Per-file failures for
    changed files are isolated: binary files (UnicodeDecodeError) and
    oversize/unavailable files (HTTPStatusError) are logged and omitted rather
    than aborting the whole gather.  Docs absent from the repo are silently
    omitted (404 is tolerated via ``tolerate_missing``).

    Args:
        github: Authenticated GitHub API client.
        repo_full_name: e.g. "owner/repo".
        files: Changed-file objects from the PR files API.
        ref: Git ref (commit SHA) to read from.
        doc_names: Repo-relative doc paths to fetch (the validated config list).

    Returns:
        A ``(file_contents, docs)`` tuple where each value maps a path/name to
        the decoded file content, with unreadable files omitted.
    """
    import asyncio

    fetchable = [f for f in files if f.get("status") != "removed"]

    async def _fetch_changed(filename: str) -> tuple[str, str | None]:
        content = await github.get_file_content(
            repo_full_name=repo_full_name,
            path=filename,
            ref=ref,
        )
        return filename, content

    async def _fetch_doc(name: str) -> tuple[str, str | None]:
        content = await github.get_file_content(
            repo_full_name=repo_full_name,
            path=name,
            ref=ref,
            tolerate_missing=True,
        )
        return name, content

    n_changed = len(fetchable)
    changed_tasks = [_fetch_changed(f["filename"]) for f in fetchable]
    doc_tasks = [_fetch_doc(name) for name in doc_names]

    # Gather all tasks in one shot. return_exceptions=True isolates per-file
    # errors in the changed-file slice so one bad file cannot abort the whole
    # gather; doc tasks handle missing-file 404s via tolerate_missing.
    all_results: list[tuple[str, str | None] | BaseException] = await asyncio.gather(
        *changed_tasks,
        *doc_tasks,
        return_exceptions=True,
    )

    changed_raw = all_results[:n_changed]
    doc_raw = all_results[n_changed:]

    file_contents: dict[str, str] = {}
    for item in changed_raw:
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
            if content is not None:
                file_contents[filename] = content

    docs: dict[str, str] = {}
    for item in doc_raw:
        if isinstance(item, BaseException):
            raise item
        name, content = item
        if content is not None:
            docs[name] = content

    return file_contents, docs


def _safe_file_path(files_root: Path, filename: str) -> Path | None:
    """Return the resolved path for ``filename`` inside ``files_root``, or None.

    Rejects filenames that would escape ``files_root`` — including absolute
    paths (e.g. ``/etc/passwd``) and relative traversals (e.g. ``../../x``).
    The check resolves symlinks so that a cleverly crafted path cannot bypass
    the guard via a symlink outside the tree.

    Args:
        files_root: The directory all changed files must be written under.
        filename: The PR-supplied filename (attacker-controlled).

    Returns:
        The safe resolved ``Path`` if ``filename`` stays inside ``files_root``,
        or ``None`` if it would escape.
    """
    # Resolve files_root once; this is our confinement boundary.
    resolved_root = files_root.resolve()

    # Reject absolute paths outright — Path('/foo') / Path('/etc/passwd') would
    # discard the base entirely and write to /etc/passwd.
    raw = Path(filename)
    if raw.is_absolute():
        return None

    candidate = (files_root / raw).resolve()

    # Verify the resolved path is still inside files_root.
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None

    return candidate


def _materialize(ctx: PRContext, directory: str) -> None:
    """Write the assembled context to disk in ``directory``.

    Produces:
      <directory>/diff.patch               — unified diff
      <directory>/pr_metadata.json         — PR metadata as JSON
      <directory>/files/<path>             — full content of each changed file
      <directory>/docs/<name>              — repo docs (if any)
      <directory>/comments.json            — conversation comments (only if any)
      <directory>/review_threads.json       — inline review threads (only if any)
      <directory>/review_summaries.json     — submitted-review summary bodies (only if any)
      <directory>/own_prior_review.json     — Heimdall's own prior review (only if present)
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

    # Conversation comments are written only when present: an empty set leaves no
    # comments.json so the CLI/synthesis paths see "no comments" cleanly rather than
    # an empty-array file the reader must special-case.
    if ctx.comments:
        (root / "comments.json").write_text(
            json.dumps(ctx.comments, indent=2), encoding="utf-8"
        )

    # Inline review threads follow the same write-only-when-present rule as
    # comments.json, so an empty set leaves no file and the CLI/synthesis paths
    # see "no threads" cleanly rather than an empty-array file to special-case.
    if ctx.review_threads:
        (root / "review_threads.json").write_text(
            json.dumps(ctx.review_threads, indent=2), encoding="utf-8"
        )

    # Review summaries follow the same write-only-when-present rule: an empty set
    # leaves no review_summaries.json so the reader sees "no summaries" cleanly.
    if ctx.review_summaries:
        (root / "review_summaries.json").write_text(
            json.dumps(ctx.review_summaries, indent=2), encoding="utf-8"
        )

    # Heimdall's own prior review is written only when one exists; absence leaves no
    # own_prior_review.json so the CLI's own-prior subcommand reads back ``null``.
    if ctx.own_prior_review:
        (root / "own_prior_review.json").write_text(
            json.dumps(ctx.own_prior_review, indent=2), encoding="utf-8"
        )

    files_root = root / "files"
    files_root.mkdir(exist_ok=True)
    for filename, content in ctx.file_contents.items():
        file_path = _safe_file_path(files_root, filename)
        if file_path is None:
            # Reject filenames that would escape the workspace (absolute or traversal).
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    if ctx.docs:
        docs_root = root / "docs"
        docs_root.mkdir(exist_ok=True)
        for name, content in ctx.docs.items():
            # Doc names come from the trusted config list (already rejected at load for
            # absolute/.. entries), but route through the same guard as files/ as
            # defense in depth so nothing can escape the workspace.
            doc_path = _safe_file_path(docs_root, name)
            if doc_path is None:
                continue
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(content, encoding="utf-8")
