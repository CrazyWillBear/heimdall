"""GitHub App authentication and API client.

Implements JWT generation (for App-level auth) and installation token exchange,
then wraps the GitHub REST calls needed by Heimdall: posting PR reviews.
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
