"""GitHub App authentication and API client.

Implements JWT generation (for App-level auth) and installation token exchange,
then wraps the GitHub REST calls needed by Heimdall: posting PR reviews and
fetching PR metadata, diffs, file lists, and file contents for seed-context assembly.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
import jwt


def parse_linked_issues_from_body(body: str) -> list[dict[str, Any]]:
    """Extract issues referenced by closing keywords in a PR body.

    Recognises the GitHub closing keywords (closes, fixes, resolves) followed
    by ``#N`` anywhere in the body.  Matching is case-insensitive.  Duplicate
    issue numbers are collapsed to a single entry.

    Args:
        body: PR description text.

    Returns:
        List of dicts with a ``number`` key for each uniquely referenced issue,
        in the order first seen.
    """
    pattern = re.compile(
        r"(?:closes|fixes|resolves)\s+#(\d+)",
        re.IGNORECASE,
    )
    seen: set[int] = set()
    results: list[dict[str, Any]] = []
    for match in pattern.finditer(body):
        number = int(match.group(1))
        if number not in seen:
            seen.add(number)
            results.append({"number": number})
    return results


def _shape_comment(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw GitHub comment object to the fields the seed records.

    Keeps only the ``body``, author ``login``, and ``author_association`` — the
    minimum a downstream lens needs to weigh a conversation comment — and drops
    every other (potentially large or sensitive) field from the API payload.

    Args:
        raw: A single comment object from the issues-comments REST endpoint.

    Returns:
        A dict with ``body``, ``author``, and ``author_association`` keys.
    """
    return {
        "body": raw.get("body") or "",
        "author": str((raw.get("user") or {}).get("login", "")),
        "author_association": str(raw.get("author_association", "")),
    }


def _shape_inline_comment(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw GitHub review (inline) comment to the fields a thread records.

    Builds on :func:`_shape_comment` (``body``/``author``/``author_association``) and
    adds the line anchor a thread needs: the ``path`` the comment is attached to and the
    ``line`` it points at — ``line`` is the comment's current line, falling back to the
    pre-image ``original_line`` for comments on an outdated diff hunk.  Like
    :func:`_shape_comment`, every other (potentially large or sensitive) API field is
    dropped.

    Args:
        raw: A single comment object from the pulls review-comments REST endpoint.

    Returns:
        A dict with ``body``, ``author``, ``author_association``, ``path``, and ``line``.
    """
    shaped = _shape_comment(raw)
    line = raw.get("line")
    if line is None:
        line = raw.get("original_line")
    shaped["path"] = str(raw.get("path") or "")
    shaped["line"] = line
    return shaped


# GitHub's submitted-review ``state`` (past tense) maps to the create-review ``event``
# verb Heimdall already uses everywhere else (APPROVE/REQUEST_CHANGES/COMMENT).  Using
# the event verb keeps a review summary tagged the same way as the verdict Heimdall posts.
_REVIEW_STATE_TO_EVENT = {
    "APPROVED": "APPROVE",
    "CHANGES_REQUESTED": "REQUEST_CHANGES",
    "COMMENTED": "COMMENT",
}


def _review_event(raw: dict[str, Any]) -> str:
    """Map a submitted review's ``state`` to its create-review event verb.

    GitHub reports a submitted review's ``state`` in the past tense (APPROVED /
    CHANGES_REQUESTED / COMMENTED); Heimdall tags review summaries with the matching
    create-review event verb (APPROVE / REQUEST_CHANGES / COMMENT) it uses elsewhere.
    An unrecognised state (e.g. DISMISSED, PENDING) is passed through verbatim so the
    raw signal is never silently dropped.
    """
    state = str(raw.get("state", ""))
    return _REVIEW_STATE_TO_EVENT.get(state, state)


def _shape_review_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw submitted-review object to the fields a summary records.

    Builds on :func:`_shape_comment` (``body``/``author``/``author_association``) and
    adds the ``event`` type (APPROVE / REQUEST_CHANGES / COMMENT) so a downstream lens
    can tell an approval from a change-request from a plain comment.  Every other
    (potentially large or sensitive) API field is dropped, like the other shapers.

    Args:
        raw: A single review object from the pulls reviews REST endpoint.

    Returns:
        A dict with ``body``, ``author``, ``author_association``, and ``event``.
    """
    shaped = _shape_comment(raw)
    shaped["event"] = _review_event(raw)
    return shaped


def group_review_comments_into_threads(
    comments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group flat inline review comments into parent-anchored reply threads.

    A PR's review-comments endpoint returns a flat list where a reply carries an
    ``in_reply_to_id`` pointing at the comment it answers; a top-level comment has none.
    This rebuilds the tree: each top-level comment becomes a thread carrying its file/line
    anchor and a ``replies`` list, and every reply is appended (in API order) under the
    thread whose root id it answers.  A reply whose parent was dropped by author filtering
    is promoted to its own thread so its content is never silently lost.

    Args:
        comments: Raw review-comment objects (already author-filtered), in API order.

    Returns:
        One dict per thread: the shaped root fields (``body``/``author``/
        ``author_association``/``path``/``line``) plus a ``replies`` list of shaped
        replies, in API (chronological) order.
    """
    threads_by_id: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for raw in comments:
        parent_id = raw.get("in_reply_to_id")
        shaped = _shape_inline_comment(raw)
        parent = threads_by_id.get(parent_id) if parent_id is not None else None
        if parent is not None:
            parent["replies"].append(shaped)
            continue
        # A root comment, or a reply whose parent was filtered out: start a new thread.
        thread = {**shaped, "replies": []}
        ordered.append(thread)
        comment_id = raw.get("id")
        if isinstance(comment_id, int):
            threads_by_id[comment_id] = thread
    return ordered


def _raise_with_body(response: httpx.Response) -> None:
    """Like ``response.raise_for_status()`` but include the response body in the error.

    httpx omits the body from :class:`httpx.HTTPStatusError`, yet GitHub's 4xx body
    carries the actual reason — e.g. a 422 on create-review whose body says an inline
    comment's "line must be part of the diff" — so an opaque status code alone is
    undebuggable.  Re-raise the same error class (callers/tests still see
    HTTPStatusError) with the truncated body appended.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"{exc}: {response.text[:1000]}",
            request=exc.request,
            response=exc.response,
        ) from exc


def make_jwt(*, app_id: int, private_key: str) -> str:
    """Return a short-lived GitHub App JWT signed with the App's private key.

    Args:
        app_id: The numeric GitHub App ID.
        private_key: PEM-encoded RSA private key string.

    Returns:
        A signed JWT string valid for 60 seconds.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,  # issued-at slightly in the past to allow clock skew
        "exp": now + (10 * 60),  # 10-minute max per GitHub docs
        "iss": str(app_id),
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


class GitHubClient:
    """Async GitHub API client for a specific App installation.

    Args:
        app_id: The numeric GitHub App ID.
        private_key: PEM-encoded RSA private key string.
        installation_id: The installation ID to authenticate as.
        http_client: Optional injected httpx.AsyncClient (for testing).
    """

    _BASE = "https://api.github.com"

    def __init__(
        self,
        *,
        app_id: int,
        private_key: str,
        installation_id: int,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._app_id = app_id
        self._private_key = private_key
        self._installation_id = installation_id
        # Track ownership so aclose() only closes clients we created, not injected ones.
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._cached_token: str | None = None

    async def aclose(self) -> None:
        """Close the underlying httpx client, but only if we created it.

        Injected clients (passed via http_client) are owned by the caller and
        must not be closed here.
        """
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> GitHubClient:
        """Support ``async with GitHubClient(...)`` usage."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the owned httpx client when leaving the async context."""
        await self.aclose()

    async def get_installation_token(self) -> str:
        """Exchange the App JWT for a short-lived installation access token."""
        app_jwt = make_jwt(app_id=self._app_id, private_key=self._private_key)
        url = f"{self._BASE}/app/installations/{self._installation_id}/access_tokens"
        response = await self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        token: str = response.json()["token"]
        return token

    async def post_review(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
        commit_id: str,
        body: str,
        event: str,
        comments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Post a pull-request review via the GitHub REST API.

        The optional ``comments`` array attaches line-anchored inline comments in
        the SAME review submission (GitHub's create-review ``comments`` field), so
        the body and its inline comments land atomically rather than as separate
        calls.  Each entry is a dict with ``path``, ``line``, ``side``, and ``body``
        (the shape produced by :func:`heimdall.diff_anchor.build_inline_comments`).

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.
            commit_id: The head commit SHA the review targets.
            body: Review body text.
            event: One of APPROVE, REQUEST_CHANGES, COMMENT.
            comments: Optional inline comments to attach to this submission.

        Returns:
            The parsed JSON response from GitHub.
        """
        token = await self.get_installation_token()
        url = f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        payload: dict[str, Any] = {
            "commit_id": commit_id,
            "body": body,
            "event": event,
        }
        if comments:
            payload["comments"] = comments
        response = await self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=payload,
        )
        _raise_with_body(response)
        result: dict[str, Any] = response.json()
        return result

    async def dismiss_review(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
        review_id: int,
        message: str,
    ) -> None:
        """Dismiss a prior PR review via the GitHub REST API.

        Dismissal applies to REQUEST_CHANGES (and APPROVED) reviews: it clears
        the blocking state while leaving the review visible with a dismissal
        note.  GitHub requires a non-empty ``message`` explaining the dismissal.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.
            review_id: The REST review id returned by ``post_review``.
            message: Explanation recorded with the dismissal.
        """
        url = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}"
            f"/reviews/{review_id}/dismissals"
        )
        response = await self._http.put(
            url,
            headers=await self._gh_headers(),
            json={"message": message, "event": "DISMISS"},
        )
        response.raise_for_status()

    async def minimize_review(self, *, node_id: str) -> None:
        """Minimize a prior review body via the GraphQL ``minimizeComment`` mutation.

        COMMENT-event reviews cannot be dismissed (dismissal is only valid for
        REQUEST_CHANGES/APPROVED), so their body is collapsed as outdated via
        GraphQL using the review's global node id.

        Args:
            node_id: The GraphQL global node id of the review (its ``node_id``
                in the REST ``post_review`` response).

        Raises:
            RuntimeError: If the GraphQL response carries an ``errors`` payload.
        """
        query = (
            "mutation($id: ID!) {"
            " minimizeComment(input: {subjectId: $id, classifier: OUTDATED})"
            " { minimizedComment { isMinimized } } }"
        )
        response = await self._http.post(
            f"{self._BASE}/graphql",
            headers=await self._gh_headers(),
            json={"query": query, "variables": {"id": node_id}},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GraphQL minimizeComment failed: {payload['errors']}")

    async def list_review_comments(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
        review_id: int,
    ) -> list[dict[str, Any]]:
        """List the inline comments attached to a specific PR review.

        Used on a new push to find the prior review's inline comments so they can
        be deleted before the fresh set is posted.  Follows pagination so every
        comment is returned even for a large prior review.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.
            review_id: The REST review id whose comments to list.

        Returns:
            All inline-comment objects (each carrying an ``id``) for the review.
        """
        url: str | None = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}"
            f"/reviews/{review_id}/comments"
        )
        headers = await self._gh_headers()
        all_comments: list[dict[str, Any]] = []
        first_page = True
        while url is not None:
            kwargs: dict[str, Any] = {"headers": headers}
            if first_page:
                kwargs["params"] = {"per_page": 100}
                first_page = False
            response = await self._http.get(url, **kwargs)
            response.raise_for_status()
            all_comments.extend(response.json())
            url = self._next_page_url(response.headers.get("link", ""))
        return all_comments

    async def delete_review_comment(
        self,
        *,
        repo_full_name: str,
        comment_id: int,
    ) -> None:
        """Delete a single PR review (inline) comment by its id.

        Inline comments are review-comment objects distinct from the review body,
        so retiring the prior review does not remove them — they must be deleted
        explicitly on a new push to stop stale inline comments accumulating.

        Args:
            repo_full_name: e.g. "owner/repo".
            comment_id: The id of the review comment to delete.
        """
        url = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/comments/{comment_id}"
        )
        response = await self._http.delete(url, headers=await self._gh_headers())
        response.raise_for_status()

    async def _gh_headers(self) -> dict[str, str]:
        """Return GitHub API headers with a fresh installation token."""
        token = await self.get_installation_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def get_pr(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> dict[str, Any]:
        """Fetch pull-request metadata.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            Parsed JSON of the PR object from the GitHub REST API.
        """
        url = f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}"
        response = await self._http.get(url, headers=await self._gh_headers())
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def get_pr_diff(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Fetch the unified diff for a pull request.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            The raw unified diff string.
        """
        url = f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}"
        headers = await self._gh_headers()
        # Request diff media type to get the raw unified diff
        headers["Accept"] = "application/vnd.github.diff"
        response = await self._http.get(url, headers=headers)
        response.raise_for_status()
        return response.text

    def _next_page_url(self, link_header: str) -> str | None:
        """Parse the Link response header and return the ``rel="next"`` URL, or None.

        GitHub paginates via RFC 5988 Link headers, e.g.:
            <https://api.github.com/…?page=2>; rel="next", <…>; rel="last"
        """
        for part in link_header.split(","):
            part = part.strip()
            if 'rel="next"' in part:
                # Extract the URL between < and >
                url_part = part.split(";")[0].strip()
                if url_part.startswith("<") and url_part.endswith(">"):
                    return url_part[1:-1]
        return None

    async def get_pr_files(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """List files changed in a pull request, following pagination to collect all files.

        The GitHub API returns at most 100 files per page.  Large PRs require
        following ``Link: rel="next"`` headers until no next page is returned.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            List of all file objects (filename, status, patch, …) from the GitHub API.
        """
        url: str | None = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/files"
        )
        headers = await self._gh_headers()
        all_files: list[dict[str, Any]] = []
        first_page = True

        while url is not None:
            # Only attach params on the first request. GitHub's next-page URLs
            # already carry page= and per_page= in their query string; passing
            # params= again would replace the entire query string (httpx
            # behaviour), stripping page= and causing an infinite loop.
            kwargs: dict[str, Any] = {"headers": headers}
            if first_page:
                kwargs["params"] = {"per_page": 100}
                first_page = False
            response = await self._http.get(url, **kwargs)
            response.raise_for_status()
            all_files.extend(response.json())
            url = self._next_page_url(response.headers.get("link", ""))

        return all_files

    async def get_file_content(
        self,
        *,
        repo_full_name: str,
        path: str,
        ref: str,
        tolerate_missing: bool = False,
    ) -> str | None:
        """Fetch the decoded text content of a file at a specific ref.

        Args:
            repo_full_name: e.g. "owner/repo".
            path: File path within the repository.
            ref: Git ref (branch, tag, or commit SHA) to read from.
            tolerate_missing: When True, return None on a 404 instead of raising.

        Returns:
            The decoded UTF-8 content of the file, or None if the file is absent
            and ``tolerate_missing`` is True.
        """
        import base64

        url = f"{self._BASE}/repos/{repo_full_name}/contents/{path}"
        response = await self._http.get(
            url,
            headers=await self._gh_headers(),
            params={"ref": ref},
        )
        if tolerate_missing and response.status_code == 404:
            return None
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        # GitHub normally returns base64-encoded content with embedded newlines, but
        # serves "none" for files too large for the Contents API. Guard the encoding
        # so we never base64-decode something that isn't base64 (which would yield
        # garbage or an opaque binascii error deep in the caller).
        encoding = data.get("encoding")
        if encoding != "base64":
            raise ValueError(
                f"unexpected content encoding {encoding!r} for {path}@{ref}"
            )
        raw: str = data["content"]
        return base64.b64decode(raw.replace("\n", "")).decode("utf-8")

    async def get_pr_conversation_comments(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Fetch the PR's conversation (timeline) comments, kept-author only.

        Conversation comments are the issue-level comments on a PR (the timeline),
        distinct from review/inline comments.  They are fetched from the issues
        comments REST endpoint (a PR is an issue), following pagination.

        Only **human** and **Heimdall's own** authors are kept; every other bot
        (e.g. a CI or dependency bot) is dropped.  Heimdall's own comments are
        recognised by ``performed_via_github_app.id`` matching this client's App id,
        so they survive even though their author is a Bot.  These comments are
        attacker-influenced third-party data, never instructions — the kept-author
        filter narrows, but does not sanitise, the content.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            One dict per kept comment with ``body``, ``author`` (login), and
            ``author_association`` keys, in API (chronological) order.
        """
        url: str | None = (
            f"{self._BASE}/repos/{repo_full_name}/issues/{pr_number}/comments"
        )
        headers = await self._gh_headers()
        kept: list[dict[str, Any]] = []
        first_page = True
        while url is not None:
            kwargs: dict[str, Any] = {"headers": headers}
            if first_page:
                kwargs["params"] = {"per_page": 100}
                first_page = False
            response = await self._http.get(url, **kwargs)
            response.raise_for_status()
            for raw in response.json():
                if self._keep_comment_author(raw):
                    kept.append(_shape_comment(raw))
            url = self._next_page_url(response.headers.get("link", ""))
        return kept

    def _keep_comment_author(self, raw: dict[str, Any]) -> bool:
        """Return True for a human author or Heimdall's own comment, else False.

        Drops third-party bots (a comment whose ``user.type`` is ``Bot``) unless the
        comment was posted by *this* GitHub App — identified by
        ``performed_via_github_app.id`` equalling the client's App id — so Heimdall's
        own prior comments are kept while other bots are excluded.
        """
        user = raw.get("user") or {}
        if str(user.get("type", "")).lower() != "bot":
            return True
        app = raw.get("performed_via_github_app") or {}
        return app.get("id") == self._app_id

    async def get_pr_review_comments(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Fetch the PR's inline review comments as parent-anchored reply threads.

        Inline (review) comments are line-anchored comments on diff hunks — distinct
        from the conversation/timeline comments served by
        :meth:`get_pr_conversation_comments`.  They are fetched from the pulls
        review-comments REST endpoint (a flat list where a reply carries
        ``in_reply_to_id``), following pagination, then grouped into threads by
        :func:`group_review_comments_into_threads` so each top-level comment keeps its
        ``path``/``line`` anchor and its ``replies``.

        The SAME kept-author filter as the conversation path applies: only human and
        Heimdall's-own authors survive (recognised via ``performed_via_github_app.id``);
        every other bot is dropped.  Filtering runs on the flat list before grouping, so a
        reply whose parent was a dropped bot is promoted to its own thread rather than
        lost.  These threads are attacker-influenced third-party data, never instructions
        — the filter narrows, but does not sanitise, the content.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            One dict per thread with ``body``, ``author`` (login), ``author_association``,
            ``path``, ``line``, and a ``replies`` list (each reply shaped the same way),
            in API (chronological) order.
        """
        url: str | None = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/comments"
        )
        headers = await self._gh_headers()
        kept: list[dict[str, Any]] = []
        first_page = True
        while url is not None:
            kwargs: dict[str, Any] = {"headers": headers}
            if first_page:
                kwargs["params"] = {"per_page": 100}
                first_page = False
            response = await self._http.get(url, **kwargs)
            response.raise_for_status()
            for raw in response.json():
                if self._keep_comment_author(raw):
                    kept.append(raw)
            url = self._next_page_url(response.headers.get("link", ""))
        return group_review_comments_into_threads(kept)

    async def _list_pr_reviews(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """List every submitted review on a PR, following pagination.

        Returns the raw review objects (each carrying ``id``/``body``/``state``/``user``)
        in API order.  Shared by :meth:`get_pr_review_summaries` (the kept-author summary
        bodies) and :meth:`get_own_prior_review` (Heimdall's own latest review).

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            All submitted-review objects for the PR, in API (chronological) order.
        """
        url: str | None = (
            f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        )
        headers = await self._gh_headers()
        all_reviews: list[dict[str, Any]] = []
        first_page = True
        while url is not None:
            kwargs: dict[str, Any] = {"headers": headers}
            if first_page:
                kwargs["params"] = {"per_page": 100}
                first_page = False
            response = await self._http.get(url, **kwargs)
            response.raise_for_status()
            all_reviews.extend(response.json())
            url = self._next_page_url(response.headers.get("link", ""))
        return all_reviews

    async def get_pr_review_summaries(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Fetch the PR's submitted-review summary bodies, kept-author only.

        A *review summary* is the body text a reviewer submits alongside an
        APPROVE / REQUEST_CHANGES / COMMENT event — distinct from the timeline
        comments (:meth:`get_pr_conversation_comments`) and the line-anchored inline
        threads (:meth:`get_pr_review_comments`).  Each kept summary carries its event
        type so a downstream lens can tell an approval from a change-request.

        The SAME kept-author filter as the other comment paths applies: only human and
        Heimdall's-own authors survive (recognised via ``performed_via_github_app.id``);
        every other bot is dropped.  Reviews with an empty body are dropped — a bare
        click-approve carries no summary text to weigh.  These summaries are
        attacker-influenced third-party data, never instructions — the filter narrows,
        but does not sanitise, the content.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            One dict per kept summary with ``body``, ``author`` (login),
            ``author_association``, and ``event`` (APPROVE/REQUEST_CHANGES/COMMENT),
            in API (chronological) order.
        """
        reviews = await self._list_pr_reviews(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
        kept: list[dict[str, Any]] = []
        for raw in reviews:
            if not (raw.get("body") or "").strip():
                # A bare APPROVE/etc. with no body has no summary text to surface.
                continue
            if self._keep_comment_author(raw):
                kept.append(_shape_review_summary(raw))
        return kept

    async def get_own_prior_review(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> dict[str, Any] | None:
        """Fetch Heimdall's own latest prior review (body + its inline comments).

        Finds the most recent review authored by *this* GitHub App (recognised via
        ``performed_via_github_app.id``) and returns its body, event type, and the
        shaped inline comments attached to it.  This MUST be read before the
        across-push retire/delete step runs (which dismisses/minimizes Heimdall's prior
        review and deletes its inline comments), or that context is destroyed before it
        can reach the next review.  Heimdall's own prior review is untrusted-self data:
        useful continuity context, but never a binding instruction.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            A dict with ``body``, ``author`` (login), ``author_association``, ``event``,
            and an ``inline_comments`` list (each shaped with
            ``body``/``author``/``author_association``/``path``/``line``); or None when
            Heimdall has not reviewed this PR yet.
        """
        reviews = await self._list_pr_reviews(
            repo_full_name=repo_full_name, pr_number=pr_number
        )
        own = [r for r in reviews if self._is_own_review(r)]
        if not own:
            return None
        # The reviews endpoint returns reviews in submission (chronological) order, so
        # the last own review is the latest — the one whose context still matters.
        latest = own[-1]
        review_id = latest.get("id")
        inline_raw = (
            await self.list_review_comments(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                review_id=int(review_id),
            )
            if isinstance(review_id, int)
            else []
        )
        summary = _shape_review_summary(latest)
        summary["inline_comments"] = [_shape_inline_comment(c) for c in inline_raw]
        return summary

    def _is_own_review(self, raw: dict[str, Any]) -> bool:
        """Return True only for a review posted by *this* GitHub App.

        Stricter than :meth:`_keep_comment_author` (which also keeps humans): an own
        review is identified solely by ``performed_via_github_app.id`` matching the
        client's App id, so a human review is never mistaken for Heimdall's own.
        """
        app = raw.get("performed_via_github_app") or {}
        return app.get("id") == self._app_id

    async def get_linked_issues(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Return issues referenced by closing keywords in the PR body.

        GitHub has no direct REST "linked issues" endpoint, so we fetch the PR
        and parse closing keywords (closes/fixes/resolves #N) from its body.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            List of dicts with a ``number`` key for each linked issue.
        """
        pr_data = await self.get_pr(repo_full_name=repo_full_name, pr_number=pr_number)
        body: str = pr_data.get("body") or ""
        return parse_linked_issues_from_body(body)
