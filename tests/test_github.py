"""Tests for GitHub App auth and API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

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


@pytest.mark.asyncio
async def test_post_review_error_includes_github_response_body() -> None:
    """A 4xx from create-review surfaces GitHub's body (the actual reason).

    httpx's raise_for_status() reports only the status + URL; GitHub's 422 body holds
    the real cause (e.g. an inline comment's line not part of the diff), so post_review
    must include it in the raised error or failures are undebuggable.
    """
    request = httpx.Request("POST", "https://api.github.com/repos/o/r/pulls/5/reviews")
    error_response = httpx.Response(422, request=request)

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "422 Unprocessable Entity", request=request, response=error_response
        )
    )
    mock_response.text = '{"message":"line must be part of the diff"}'

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.post_review(
                repo_full_name="o/r",
                pr_number=5,
                commit_id="sha",
                body="b",
                event="COMMENT",
                comments=[{"path": "x.py", "line": 9, "side": "RIGHT", "body": "c"}],
            )

    assert "line must be part of the diff" in str(excinfo.value)


@pytest.mark.asyncio
async def test_post_review_attaches_inline_comments_in_same_submission() -> None:
    """post_review attaches the comments array to the single create-review call."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"id": 1})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    comments = [{"path": "a.py", "line": 3, "side": "RIGHT", "body": "x"}]
    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.post_review(
            repo_full_name="owner/repo",
            pr_number=5,
            commit_id="sha999",
            body="body",
            event="COMMENT",
            comments=comments,
        )

    # Exactly one POST: the comments ride in the same create-review submission.
    assert mock_http.post.await_count == 1
    sent_json = mock_http.post.call_args[1]["json"]
    assert sent_json["comments"] == comments


@pytest.mark.asyncio
async def test_post_review_omits_comments_key_when_none() -> None:
    """With no inline comments, the JSON payload carries no ``comments`` key."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"id": 1})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.post_review(
            repo_full_name="owner/repo",
            pr_number=5,
            commit_id="sha999",
            body="body",
            event="COMMENT",
        )

    assert "comments" not in mock_http.post.call_args[1]["json"]


@pytest.mark.asyncio
async def test_list_review_comments_hits_review_comments_endpoint() -> None:
    """list_review_comments GETs the review's comments endpoint and returns the rows."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=[{"id": 11}, {"id": 12}])
    mock_response.headers = {}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        rows = await client.list_review_comments(
            repo_full_name="owner/repo", pr_number=5, review_id=77
        )

    assert "pulls/5/reviews/77/comments" in mock_http.get.call_args[0][0]
    assert rows == [{"id": 11}, {"id": 12}]


@pytest.mark.asyncio
async def test_delete_review_comment_hits_pulls_comments_endpoint() -> None:
    """delete_review_comment DELETEs the pull review comment by id."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.delete = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.delete_review_comment(
            repo_full_name="owner/repo", comment_id=101
        )

    assert "pulls/comments/101" in mock_http.delete.call_args[0][0]
    mock_response.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# Conversation (timeline) comments: fetch + kept-author filter
# ---------------------------------------------------------------------------


def _comment(
    *,
    body: str,
    login: str,
    user_type: str = "User",
    association: str = "CONTRIBUTOR",
    app_id: int | None = None,
) -> dict[str, object]:
    """Build a raw issues-comments API object for the filter tests."""
    raw: dict[str, object] = {
        "body": body,
        "user": {"login": login, "type": user_type},
        "author_association": association,
    }
    if app_id is not None:
        raw["performed_via_github_app"] = {"id": app_id}
    return raw


@pytest.mark.asyncio
async def test_get_pr_conversation_comments_hits_issues_comments_endpoint() -> None:
    """Conversation comments come from the issues comments endpoint (a PR is an issue)."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value=[_comment(body="hi", login="alice")]
    )
    mock_response.headers = {}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        rows = await client.get_pr_conversation_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert "issues/5/comments" in mock_http.get.call_args[0][0]
    assert rows == [
        {"body": "hi", "author": "alice", "author_association": "CONTRIBUTOR"}
    ]


@pytest.mark.asyncio
async def test_get_pr_conversation_comments_keeps_humans_and_heimdall_drops_bots() -> None:
    """Humans + Heimdall's own are kept; other bots are dropped."""
    app_id = 4242
    raw = [
        _comment(body="from a human", login="alice", association="MEMBER"),
        _comment(body="from CI bot", login="ci-bot", user_type="Bot"),
        _comment(
            body="from heimdall",
            login="heimdall[bot]",
            user_type="Bot",
            app_id=app_id,
        ),
        _comment(
            body="from another app bot",
            login="dependabot[bot]",
            user_type="Bot",
            app_id=9999,
        ),
    ]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=raw)
    mock_response.headers = {}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        rows = await client.get_pr_conversation_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    bodies = [r["body"] for r in rows]
    assert bodies == ["from a human", "from heimdall"]
    # Each kept comment carries body, author login, and author_association.
    assert rows[0] == {
        "body": "from a human",
        "author": "alice",
        "author_association": "MEMBER",
    }
    assert rows[1]["author"] == "heimdall[bot]"


@pytest.mark.asyncio
async def test_get_pr_conversation_comments_empty() -> None:
    """No comments yields an empty list (clean empty-set handling)."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=[])
    mock_response.headers = {}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        rows = await client.get_pr_conversation_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert rows == []


# ---------------------------------------------------------------------------
# Across-push review lifecycle: dismiss (REQUEST_CHANGES) / minimize (COMMENT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dismiss_review_puts_to_dismissals_endpoint() -> None:
    """dismiss_review PUTs a message to the review dismissals REST endpoint."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    mock_http = AsyncMock()
    mock_http.put = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.dismiss_review(
            repo_full_name="owner/repo",
            pr_number=5,
            review_id=77,
            message="Superseded by a newer push.",
        )

    call_args = mock_http.put.call_args
    assert "pulls/5/reviews/77/dismissals" in call_args[0][0]
    assert call_args[1]["json"]["message"] == "Superseded by a newer push."
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_minimize_review_calls_graphql_minimize_comment() -> None:
    """minimize_review issues a GraphQL minimizeComment mutation on the node id."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
    )

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        await client.minimize_review(node_id="REVIEW_NODE_ID")

    call_args = mock_http.post.call_args
    assert call_args[0][0].endswith("/graphql")
    sent_json = call_args[1]["json"]
    assert "minimizeComment" in sent_json["query"]
    assert sent_json["variables"]["id"] == "REVIEW_NODE_ID"
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_minimize_review_raises_on_graphql_errors() -> None:
    """minimize_review surfaces a GraphQL ``errors`` payload as a RuntimeError."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"errors": [{"message": "boom"}]})

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=mock_response)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with pytest.raises(RuntimeError, match="boom"):
            await client.minimize_review(node_id="REVIEW_NODE_ID")


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
    """get_pr_files follows Link headers until exhausted and returns all files.

    The mock simulates httpx's param-merging: it builds the effective URL by
    merging the ``params`` kwarg into the base URL (exactly as httpx does),
    then keys its response on the resulting ``page`` query param.  This means a
    stripped-page regression — where ``params={'per_page': 100}`` overwrites the
    existing ``page=N`` query string — causes the mock to see ``page=1`` on every
    call and the hard cap assertion fires instead of the test silently passing.
    """
    page1_files = [_make_file(f"file{i}.py") for i in range(3)]
    page2_files = [_make_file(f"file{i}.py") for i in range(3, 6)]
    page3_files = [_make_file(f"file{i}.py") for i in range(6, 8)]

    base_url = "https://api.github.com/repos/owner/repo/pulls/7/files"
    page2_url = f"{base_url}?page=2&per_page=100"
    page3_url = f"{base_url}?page=3&per_page=100"

    pages: dict[int, MagicMock] = {
        1: _page_response(page1_files, next_url=page2_url),
        2: _page_response(page2_files, next_url=page3_url),
        3: _page_response(page3_files, next_url=None),
    }

    _call_count = 0

    async def _url_keyed_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal _call_count
        _call_count += 1
        # Hard cap: more than 5 calls means pagination is not advancing.
        assert _call_count <= 5, (
            f"get_pr_files made {_call_count} requests — likely stuck in an "
            "infinite loop due to a stripped page= parameter"
        )
        # Simulate httpx's behaviour: params= replaces the query string, so
        # build the effective URL the same way httpx would.
        effective_url = httpx.Request("GET", url, params=kwargs.get("params")).url  # type: ignore[arg-type]
        qs = parse_qs(str(effective_url).split("?", 1)[-1])
        page = int(qs.get("page", ["1"])[0])
        assert page in pages, f"Unexpected page={page} requested (effective URL: {effective_url})"
        return pages[page]

    mock_http = AsyncMock()
    mock_http.get = _url_keyed_get

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        result = await client.get_pr_files(repo_full_name="owner/repo", pr_number=7)

    assert result == page1_files + page2_files + page3_files
    assert _call_count == 3


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
# get_file_content: 404 tolerance for doc fetching
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


@pytest.mark.asyncio
async def test_get_file_content_decodes_base64() -> None:
    """The standard base64 Contents API response is decoded to UTF-8 text."""
    import base64

    body = base64.b64encode(b"severity_threshold: low\n").decode("ascii")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"content": body, "encoding": "base64"})

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        result = await client.get_file_content(
            repo_full_name="owner/repo", path="x.yml", ref="abc123"
        )

    assert result == "severity_threshold: low\n"


@pytest.mark.asyncio
async def test_get_file_content_rejects_non_base64_encoding() -> None:
    """A non-base64 encoding (e.g. a file too large for the Contents API) raises.

    Guards against silently base64-decoding non-base64 content into garbage.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"content": "", "encoding": "none"})

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)

    with patch("heimdall.github.make_jwt", return_value="fake.jwt"), patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with pytest.raises(ValueError):
            await client.get_file_content(
                repo_full_name="owner/repo", path="big.bin", ref="abc123"
            )
