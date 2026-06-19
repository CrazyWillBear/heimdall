"""GitHub App authentication and API client.

Implements JWT generation (for App-level auth) and installation token exchange,
then wraps the GitHub REST calls needed by Heimdall: posting PR reviews and
fetching PR metadata, diffs, file lists, and file contents for seed-context assembly.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt


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
    ) -> dict[str, Any]:
        """Post a pull-request review via the GitHub REST API.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.
            commit_id: The head commit SHA the review targets.
            body: Review body text.
            event: One of APPROVE, REQUEST_CHANGES, COMMENT.

        Returns:
            The parsed JSON response from GitHub.
        """
        token = await self.get_installation_token()
        url = f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        response = await self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"commit_id": commit_id, "body": body, "event": event},
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

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

    async def get_pr_files(
        self,
        *,
        repo_full_name: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """List files changed in a pull request.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            List of file objects (filename, status, patch, …) from the GitHub API.
        """
        url = f"{self._BASE}/repos/{repo_full_name}/pulls/{pr_number}/files"
        response = await self._http.get(url, headers=await self._gh_headers())
        response.raise_for_status()
        result: list[dict[str, Any]] = response.json()
        return result

    async def get_file_content(
        self,
        *,
        repo_full_name: str,
        path: str,
        ref: str,
    ) -> str:
        """Fetch the decoded text content of a file at a specific ref.

        Args:
            repo_full_name: e.g. "owner/repo".
            path: File path within the repository.
            ref: Git ref (branch, tag, or commit SHA) to read from.

        Returns:
            The decoded UTF-8 content of the file.
        """
        import base64

        url = f"{self._BASE}/repos/{repo_full_name}/contents/{path}"
        response = await self._http.get(
            url,
            headers=await self._gh_headers(),
            params={"ref": ref},
        )
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
        """Return issues mentioned in the PR body via closing keywords.

        GitHub does not expose a direct "linked issues" API endpoint, so we use
        the GraphQL timeline API to surface cross-references.  For now we return
        an empty list and leave the GraphQL implementation as a future extension;
        the seed-context caller can fill this via the PR body parser if needed.

        Args:
            repo_full_name: e.g. "owner/repo".
            pr_number: The PR number.

        Returns:
            List of linked issue dicts (may be empty if none found).
        """
        # The REST API has no direct "linked issues" endpoint.
        # Return empty list; a future issue can wire GraphQL here.
        _ = repo_full_name, pr_number  # acknowledged unused for now
        return []
