"""Tests for GitHub App auth and API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from heimdall.github import GitHubClient, make_jwt, parse_linked_issues_from_body


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


# ---------------------------------------------------------------------------
# get_pr_files pagination
# ---------------------------------------------------------------------------


def _make_file(name: str) -> dict[str, str]:
    return {"filename": name, "status": "modified"}


def _page_response(
    files: list[dict[str, str]], next_url: str | None = None
) -> MagicMock:
    """Return a mock httpx response for one page of PR files."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=files)
    link_header = (
        f'<{next_url}>; rel="next", <https://api.github.com/last>; rel="last"'
        if next_url
        else ""
    )
    resp.headers = {"link": link_header}
    return resp


@pytest.mark.asyncio
async def test_get_pr_files_single_page() -> None:
    """get_pr_files returns all files when the response fits in one page."""
    files = [_make_file("a.py"), _make_file("b.py")]
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=_page_response(files))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        result = await client.get_pr_files(repo_full_name="owner/repo", pr_number=1)

    assert result == files
    mock_http.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_pr_files_multi_page_collects_all_files() -> None:
    """get_pr_files follows Link headers until exhausted and returns all files."""
    page1 = [_make_file(f"file{i}.py") for i in range(3)]
    page2 = [_make_file(f"file{i}.py") for i in range(3, 6)]
    page3 = [_make_file(f"file{i}.py") for i in range(6, 8)]

    page2_url = "https://api.github.com/repos/owner/repo/pulls/7/files?page=2"
    page3_url = "https://api.github.com/repos/owner/repo/pulls/7/files?page=3"

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(
        side_effect=[
            _page_response(page1, next_url=page2_url),
            _page_response(page2, next_url=page3_url),
            _page_response(page3, next_url=None),
        ]
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        result = await client.get_pr_files(repo_full_name="owner/repo", pr_number=7)

    assert result == page1 + page2 + page3
    assert mock_http.get.await_count == 3


@pytest.mark.asyncio
async def test_get_pr_files_uses_per_page_100() -> None:
    """get_pr_files requests 100 files per page to minimize round-trips."""
    files = [_make_file("x.py")]
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=_page_response(files))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.get_pr_files(repo_full_name="owner/repo", pr_number=1)

    call_kwargs = mock_http.get.call_args[1]
    assert call_kwargs.get("params", {}).get("per_page") == 100


# ---------------------------------------------------------------------------
# parse_linked_issues_from_body: closing-keyword parser
# ---------------------------------------------------------------------------


def test_parse_linked_issues_closes_keyword() -> None:
    """'closes #N' extracts issue number N."""
    results = parse_linked_issues_from_body("closes #42")
    assert results == [{"number": 42}]


def test_parse_linked_issues_fixes_keyword() -> None:
    """'fixes #N' extracts issue number N."""
    results = parse_linked_issues_from_body("Fixes #7")
    assert results == [{"number": 7}]


def test_parse_linked_issues_resolves_keyword() -> None:
    """'resolves #N' extracts issue number N."""
    results = parse_linked_issues_from_body("resolves #100")
    assert results == [{"number": 100}]


def test_parse_linked_issues_multiple() -> None:
    """Multiple closing references in one body are all extracted."""
    body = "This PR closes #1 and fixes #2.\nAlso resolves #3."
    results = parse_linked_issues_from_body(body)
    numbers = [r["number"] for r in results]
    assert sorted(numbers) == [1, 2, 3]


def test_parse_linked_issues_deduplicates() -> None:
    """The same issue referenced twice yields one entry."""
    body = "closes #5\ncloses #5"
    results = parse_linked_issues_from_body(body)
    assert results == [{"number": 5}]


def test_parse_linked_issues_empty_body() -> None:
    """An empty body yields no linked issues."""
    assert parse_linked_issues_from_body("") == []


def test_parse_linked_issues_no_keywords() -> None:
    """A body with no closing keywords yields an empty list."""
    assert parse_linked_issues_from_body("Just a plain description with #5 but no keyword.") == []


def test_parse_linked_issues_case_insensitive() -> None:
    """Keyword matching is case-insensitive."""
    results = parse_linked_issues_from_body("CLOSES #10")
    assert results == [{"number": 10}]


# ---------------------------------------------------------------------------
# get_linked_issues: integration with PR body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_linked_issues_parses_pr_body() -> None:
    """get_linked_issues fetches the PR and parses closing keywords from the body."""
    mock_pr_response = MagicMock()
    mock_pr_response.raise_for_status = MagicMock()
    mock_pr_response.json = MagicMock(
        return_value={
            "number": 1,
            "title": "Test PR",
            "body": "Fixes #99 and closes #100.",
            "user": {"login": "dev"},
            "base": {"sha": "aaa", "ref": "main", "repo": {"full_name": "owner/repo"}},
            "head": {"sha": "bbb", "ref": "feat"},
        }
    )
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_pr_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        results = await client.get_linked_issues(repo_full_name="owner/repo", pr_number=1)

    numbers = sorted(r["number"] for r in results)
    assert numbers == [99, 100]


@pytest.mark.asyncio
async def test_get_linked_issues_empty_when_no_keywords() -> None:
    """get_linked_issues returns [] when the PR body has no closing keywords."""
    mock_pr_response = MagicMock()
    mock_pr_response.raise_for_status = MagicMock()
    mock_pr_response.json = MagicMock(
        return_value={
            "number": 2,
            "title": "No issues",
            "body": "Just a description.",
            "user": {"login": "dev"},
            "base": {"sha": "aaa", "ref": "main", "repo": {"full_name": "owner/repo"}},
            "head": {"sha": "bbb", "ref": "feat"},
        }
    )
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_pr_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        results = await client.get_linked_issues(repo_full_name="owner/repo", pr_number=2)

    assert results == []


# ---------------------------------------------------------------------------
# get_file_content: 404 tolerance for convention doc fetching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_file_content_returns_none_on_404() -> None:
    """get_file_content returns None when the file does not exist (404)."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=mock_response)
    )

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        result = await client.get_file_content(
            repo_full_name="owner/repo",
            path="MISSING.md",
            ref="abc123",
            tolerate_missing=True,
        )

    assert result is None
