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
        response.raise_for_status()
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
        # GitHub returns base64-encoded content with embedded newlines
        raw: str = data["content"]
        return base64.b64decode(raw.replace("\n", "")).decode("utf-8")

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
