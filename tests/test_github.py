"""Tests for GitHub App auth and API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from heimdall.github import GitHubClient, make_jwt


def test_make_jwt_returns_string() -> None:
    """make_jwt produces a non-empty string JWT."""
    # Use a real RSA key stub — just check it doesn't crash with a valid key shape.
    # We patch jwt.encode so no actual crypto needed.
    with patch("heimdall.github.jwt.encode", return_value="fake.jwt.token") as mock_enc:
        token = make_jwt(app_id=123, private_key="-----BEGIN RSA PRIVATE KEY-----\n...")
    assert token == "fake.jwt.token"
    mock_enc.assert_called_once()


@pytest.mark.asyncio
async def test_get_installation_token() -> None:
    """GitHubClient fetches an installation token via JWT."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"token": "ghs_test"})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        token = await client.get_installation_token()

    assert token == "ghs_test"
    mock_http.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_review() -> None:
    """GitHubClient posts a PR review to the correct GitHub API endpoint."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"id": 1})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.post_review(
            repo_full_name="owner/repo",
            pr_number=5,
            commit_id="sha999",
            body="Heimdall received this PR",
            event="COMMENT",
        )

    call_args = mock_http.post.call_args
    assert "pulls/5/reviews" in call_args[0][0]
    sent_json = call_args[1]["json"]
    assert sent_json["body"] == "Heimdall received this PR"
    assert sent_json["event"] == "COMMENT"


# ---------------------------------------------------------------------------
# GitHubClient lifecycle: aclose and async context-manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_owned_http_client() -> None:
    """aclose() closes self._http when GitHubClient created it (no injection)."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)

    with patch("heimdall.github.httpx.AsyncClient", return_value=mock_http):
        client = GitHubClient(app_id=1, private_key="key", installation_id=42)
        await client.aclose()

    mock_http.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_does_not_close_injected_http_client() -> None:
    """aclose() must NOT close an http_client that was injected by the caller."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    client = GitHubClient(
        app_id=1, private_key="key", installation_id=42, http_client=mock_http
    )
    await client.aclose()

    mock_http.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_context_manager_closes_owned_client() -> None:
    """async with GitHubClient(...) closes the owned httpx client on exit."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)

    with patch("heimdall.github.httpx.AsyncClient", return_value=mock_http):
        async with GitHubClient(app_id=1, private_key="key", installation_id=42):
            pass

    mock_http.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_context_manager_does_not_close_injected_client() -> None:
    """async with GitHubClient(..., http_client=x) does not close the injected client."""
    mock_http = AsyncMock(spec=httpx.AsyncClient)

    async with GitHubClient(
        app_id=1, private_key="key", installation_id=42, http_client=mock_http
    ):
        pass

    mock_http.aclose.assert_not_awaited()
