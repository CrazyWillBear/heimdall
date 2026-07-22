"""Tests for GitHub App auth and API client."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

from heimdall.github import (
    _MAX_PAGINATION_PAGES,
    _MAX_READ_ATTEMPTS,
    GitHubClient,
    make_jwt,
    parse_linked_issues_from_body,
)


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
# Inline review comments: fetch + group into reply threads + kept-author filter
# ---------------------------------------------------------------------------


def _review_comment(
    *,
    body: str,
    login: str,
    comment_id: int,
    path: str = "foo.py",
    line: int | None = 3,
    original_line: int | None = None,
    in_reply_to_id: int | None = None,
    user_type: str = "User",
    association: str = "CONTRIBUTOR",
    app_id: int | None = None,
) -> dict[str, object]:
    """Build a raw pulls review-comments API object for the thread tests."""
    raw: dict[str, object] = {
        "id": comment_id,
        "body": body,
        "user": {"login": login, "type": user_type},
        "author_association": association,
        "path": path,
        "line": line,
    }
    if original_line is not None:
        raw["original_line"] = original_line
    if in_reply_to_id is not None:
        raw["in_reply_to_id"] = in_reply_to_id
    if app_id is not None:
        raw["performed_via_github_app"] = {"id": app_id}
    return raw


def _graphql_threads_response(
    threads: list[dict[str, object]], *, errors: object | None = None
) -> MagicMock:
    """Build a mock httpx response for a reviewThreads GraphQL page.

    ``threads`` is the list of node dicts (each ``{"isResolved": bool, "comments":
    {"nodes": [{"databaseId": int}]}}``).  ``errors`` injects a GraphQL errors payload
    so the degrade-clean path can be exercised.
    """
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if errors is not None:
        resp.json = MagicMock(return_value={"errors": errors})
    else:
        resp.json = MagicMock(
            return_value={
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": threads,
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }
            }
        )
    return resp


def _single_page(
    rows: list[dict[str, object]],
    *,
    resolutions: list[dict[str, object]] | None = None,
) -> AsyncMock:
    """Build a mock http client whose GET returns one page of ``rows``.

    ``post`` (the reviewThreads GraphQL call) returns ``resolutions`` — a list of
    thread node dicts, defaulting to none so unmatched threads degrade to unresolved.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=rows)
    mock_response.headers = {}
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.post = AsyncMock(
        return_value=_graphql_threads_response(resolutions or [])
    )
    return mock_http


@pytest.mark.asyncio
async def test_get_pr_review_comments_hits_pulls_comments_endpoint() -> None:
    """Inline review comments come from the pulls comments endpoint (not issues)."""
    mock_http = _single_page(
        [_review_comment(body="anchored", login="alice", comment_id=1)]
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert "pulls/5/comments" in mock_http.get.call_args[0][0]
    assert threads == [
        {
            "body": "anchored",
            "author": "alice",
            "author_association": "CONTRIBUTOR",
            "path": "foo.py",
            "line": 3,
            "is_outdated": False,
            "replies": [],
            "is_resolved": False,
        }
    ]


@pytest.mark.asyncio
async def test_get_pr_review_comments_groups_replies_under_parent() -> None:
    """A reply (in_reply_to_id) is nested under its parent thread, anchor preserved."""
    rows = [
        _review_comment(body="root", login="alice", comment_id=1, path="a.py", line=7),
        _review_comment(
            body="reply", login="bob", comment_id=2, in_reply_to_id=1, line=7
        ),
    ]
    mock_http = _single_page(rows)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert len(threads) == 1
    thread = threads[0]
    assert thread["body"] == "root"
    assert thread["path"] == "a.py"
    assert thread["line"] == 7
    assert [r["body"] for r in thread["replies"]] == ["reply"]
    assert thread["replies"][0]["author"] == "bob"


@pytest.mark.asyncio
async def test_get_pr_review_comments_keeps_humans_and_heimdall_drops_bots() -> None:
    """The conversation-path author filter applies to inline comments and replies."""
    app_id = 4242
    rows = [
        _review_comment(body="from human", login="alice", comment_id=1),
        _review_comment(
            body="from CI bot", login="ci[bot]", comment_id=2, user_type="Bot"
        ),
        _review_comment(
            body="heimdall reply",
            login="heimdall[bot]",
            comment_id=3,
            in_reply_to_id=1,
            user_type="Bot",
            app_id=app_id,
        ),
        _review_comment(
            body="other-bot reply",
            login="dependabot[bot]",
            comment_id=4,
            in_reply_to_id=1,
            user_type="Bot",
            app_id=9999,
        ),
    ]
    mock_http = _single_page(rows)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    # The CI-bot root and the other-app-bot reply are dropped; the Heimdall reply stays.
    assert len(threads) == 1
    assert threads[0]["body"] == "from human"
    assert [r["body"] for r in threads[0]["replies"]] == ["heimdall reply"]


@pytest.mark.asyncio
async def test_get_pr_review_comments_falls_back_to_original_line() -> None:
    """An outdated comment with no current ``line`` anchors on ``original_line``."""
    mock_http = _single_page(
        [
            _review_comment(
                body="outdated", login="alice", comment_id=1, line=None, original_line=9
            )
        ]
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert threads[0]["line"] == 9
    # The line fell back to original_line, so the thread is flagged outdated.
    assert threads[0]["is_outdated"] is True


@pytest.mark.asyncio
async def test_get_pr_review_comments_live_thread_not_outdated() -> None:
    """A comment with a current ``line`` is not flagged outdated."""
    mock_http = _single_page(
        [_review_comment(body="live", login="alice", comment_id=1, line=4)]
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert threads[0]["line"] == 4
    assert threads[0]["is_outdated"] is False


@pytest.mark.asyncio
async def test_get_pr_review_comments_empty() -> None:
    """No inline comments yields an empty thread list (clean empty handling)."""
    mock_http = _single_page([])

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert threads == []


def test_group_review_comments_promotes_orphan_reply_to_thread() -> None:
    """A reply whose parent was filtered out becomes its own thread, not lost."""
    from heimdall.github import group_review_comments_into_threads

    # The root (id=1) is absent from the input (e.g. dropped by author filtering);
    # only the reply remains, and it must still surface as a standalone thread.
    threads = group_review_comments_into_threads(
        [_review_comment(body="orphan reply", login="bob", comment_id=2, in_reply_to_id=1)]
    )

    assert len(threads) == 1
    assert threads[0]["body"] == "orphan reply"
    assert threads[0]["replies"] == []
    assert threads[0]["is_resolved"] is False


# ---------------------------------------------------------------------------
# Inline-thread resolution state via GraphQL reviewThreads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_pr_review_comments_tags_resolved_thread() -> None:
    """A reviewThread reported isResolved tags its correlated REST thread resolved."""
    rows = [
        _review_comment(body="root", login="alice", comment_id=1, path="a.py", line=7),
        _review_comment(
            body="reply", login="bob", comment_id=2, in_reply_to_id=1, line=7
        ),
        _review_comment(body="open", login="carol", comment_id=9, path="b.py", line=2),
    ]
    # GraphQL: thread for comments 1/2 is resolved; the comment-9 thread is open.
    resolutions = [
        {
            "isResolved": True,
            "comments": {"nodes": [{"databaseId": 1}, {"databaseId": 2}]},
        },
        {"isResolved": False, "comments": {"nodes": [{"databaseId": 9}]}},
    ]
    mock_http = _single_page(rows, resolutions=resolutions)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    # The GraphQL query went to /graphql and named reviewThreads.
    graphql_call = mock_http.post.call_args
    assert graphql_call[0][0].endswith("/graphql")
    assert "reviewThreads" in graphql_call[1]["json"]["query"]

    by_body = {t["body"]: t for t in threads}
    assert by_body["root"]["is_resolved"] is True
    assert by_body["open"]["is_resolved"] is False


@pytest.mark.asyncio
async def test_get_pr_review_comments_defaults_unresolved_without_graphql_match() -> None:
    """A thread with no matching reviewThread node defaults to unresolved, not crash."""
    rows = [_review_comment(body="lonely", login="alice", comment_id=1)]
    # GraphQL returns a thread for an unrelated comment id, so nothing correlates.
    resolutions = [
        {"isResolved": True, "comments": {"nodes": [{"databaseId": 999}]}}
    ]
    mock_http = _single_page(rows, resolutions=resolutions)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert threads[0]["is_resolved"] is False


@pytest.mark.asyncio
async def test_get_pr_review_comments_graphql_error_degrades_to_unresolved() -> None:
    """A GraphQL errors payload degrades cleanly: threads default to unresolved."""
    rows = [_review_comment(body="root", login="alice", comment_id=1)]
    mock_http = _single_page(rows)
    mock_http.post = AsyncMock(
        return_value=_graphql_threads_response([], errors=[{"message": "boom"}])
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert len(threads) == 1
    assert threads[0]["is_resolved"] is False


@pytest.mark.asyncio
async def test_get_pr_review_comments_graphql_raises_degrades_to_unresolved() -> None:
    """A transport-level GraphQL failure never crashes the review path."""
    rows = [_review_comment(body="root", login="alice", comment_id=1)]
    mock_http = _single_page(rows)
    mock_http.post = AsyncMock(side_effect=httpx.ConnectError("network down"))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        threads = await client.get_pr_review_comments(
            repo_full_name="owner/repo", pr_number=5
        )

    assert threads[0]["is_resolved"] is False


@pytest.mark.asyncio
async def test_get_review_thread_resolutions_follows_graphql_pagination() -> None:
    """reviewThreads pages are followed; every node's databaseId maps to isResolved."""
    page1 = MagicMock()
    page1.raise_for_status = MagicMock()
    page1.json = MagicMock(
        return_value={
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [
                                {
                                    "isResolved": True,
                                    "comments": {"nodes": [{"databaseId": 1}]},
                                }
                            ],
                            "pageInfo": {
                                "hasNextPage": True,
                                "endCursor": "CURSOR1",
                            },
                        }
                    }
                }
            }
        }
    )
    page2 = _graphql_threads_response(
        [{"isResolved": False, "comments": {"nodes": [{"databaseId": 2}]}}]
    )
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=[page1, page2])

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        resolutions = await client.get_review_thread_resolutions(
            repo_full_name="owner/repo", pr_number=5
        )

    assert resolutions == {1: True, 2: False}
    # The second page passed the endCursor from page one.
    assert mock_http.post.call_args_list[1][1]["json"]["variables"]["after"] == "CURSOR1"


# ---------------------------------------------------------------------------
# Comment/review pagination page ceiling (_MAX_PAGINATION_PAGES): an attacker-
# influenceable PR with a pathologically large discussion can't drive unbounded
# pagination. Each loop stops at the ceiling and logs a truncation WARNING.
# ---------------------------------------------------------------------------


def _perpetual_next_get() -> tuple[AsyncMock, list[int]]:
    """Build a GET mock that ALWAYS returns one row with a perpetual rel="next" link.

    Returns the mock alongside a one-element call counter (``calls[0]``). The Link
    header never terminates, so a loop with no ceiling would never stop — the ceiling
    must cap the GET count at ``_MAX_PAGINATION_PAGES``.
    """
    calls = [0]

    async def _get(url: str, **_kwargs: object) -> MagicMock:
        calls[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=[{"id": calls[0]}])
        resp.headers = {
            "link": '<https://api.github.com/next>; rel="next"'
        }
        return resp

    mock_get = AsyncMock(side_effect=_get)
    return mock_get, calls


@pytest.mark.asyncio
async def test_get_pr_conversation_comments_stops_at_page_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A never-ending next-link is bounded at _MAX_PAGINATION_PAGES with a WARNING."""
    mock_get, calls = _perpetual_next_get()
    mock_http = AsyncMock()
    mock_http.get = mock_get

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with caplog.at_level(logging.WARNING):
            rows = await client.get_pr_conversation_comments(
                repo_full_name="owner/repo", pr_number=5
            )

    assert calls[0] == _MAX_PAGINATION_PAGES
    assert len(rows) == _MAX_PAGINATION_PAGES
    assert any(
        "ceiling" in r.message and "get_pr_conversation_comments" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_get_pr_review_comments_stops_at_page_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The inline-comment fetch is bounded at the ceiling, then groups what it has."""
    mock_get, calls = _perpetual_next_get()
    mock_http = AsyncMock()
    mock_http.get = mock_get
    # The trailing resolutions GraphQL call must still succeed (empty threads).
    mock_http.post = AsyncMock(return_value=_graphql_threads_response([]))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with caplog.at_level(logging.WARNING):
            await client.get_pr_review_comments(
                repo_full_name="owner/repo", pr_number=5
            )

    assert calls[0] == _MAX_PAGINATION_PAGES
    assert any(
        "ceiling" in r.message and "get_pr_review_comments" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_list_pr_reviews_stops_at_page_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The submitted-reviews fetch is bounded at the ceiling (via the public path)."""
    mock_get, calls = _perpetual_next_get()
    mock_http = AsyncMock()
    mock_http.get = mock_get

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with caplog.at_level(logging.WARNING):
            await client.get_pr_review_summaries(
                repo_full_name="owner/repo", pr_number=5
            )

    assert calls[0] == _MAX_PAGINATION_PAGES
    assert any(
        "ceiling" in r.message and "_list_pr_reviews" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_list_review_comments_stops_at_page_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The per-review inline-comment fetch is bounded at the ceiling with a WARNING."""
    mock_get, calls = _perpetual_next_get()
    mock_http = AsyncMock()
    mock_http.get = mock_get

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with caplog.at_level(logging.WARNING):
            rows = await client.list_review_comments(
                repo_full_name="owner/repo", pr_number=5, review_id=77
            )

    assert calls[0] == _MAX_PAGINATION_PAGES
    assert len(rows) == _MAX_PAGINATION_PAGES
    assert any(
        "ceiling" in r.message and "list_review_comments" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_get_review_thread_resolutions_stops_at_page_ceiling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A never-ending GraphQL pageInfo is bounded at the ceiling with a WARNING."""
    calls = [0]

    async def _post(url: str, **_kwargs: object) -> MagicMock:
        calls[0] += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(
            return_value={
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [
                                    {
                                        "isResolved": True,
                                        "comments": {
                                            "nodes": [{"databaseId": calls[0]}]
                                        },
                                    }
                                ],
                                "pageInfo": {
                                    "hasNextPage": True,
                                    "endCursor": "CURSOR",
                                },
                            }
                        }
                    }
                }
            }
        )
        return resp

    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=_post)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        with caplog.at_level(logging.WARNING):
            resolutions = await client.get_review_thread_resolutions(
                repo_full_name="owner/repo", pr_number=5
            )

    assert calls[0] == _MAX_PAGINATION_PAGES
    assert resolutions
    assert any(
        "ceiling" in r.message and "get_review_thread_resolutions" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Review summaries: submitted-review bodies + their event type, kept-author only
# ---------------------------------------------------------------------------


def _review(
    *,
    body: str,
    login: str,
    review_id: int,
    state: str = "COMMENTED",
    user_type: str = "User",
    association: str = "CONTRIBUTOR",
    app_id: int | None = None,
) -> dict[str, object]:
    """Build a raw pulls-reviews API object for the review-summary tests."""
    raw: dict[str, object] = {
        "id": review_id,
        "body": body,
        "state": state,
        "user": {"login": login, "type": user_type},
        "author_association": association,
    }
    if app_id is not None:
        raw["performed_via_github_app"] = {"id": app_id}
    return raw


@pytest.mark.asyncio
async def test_get_pr_review_summaries_hits_pulls_reviews_endpoint() -> None:
    """Review summaries come from the pulls reviews endpoint, with their event type."""
    mock_http = _single_page(
        [_review(body="LGTM overall", login="alice", review_id=1, state="APPROVED")]
    )

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        rows = await client.get_pr_review_summaries(
            repo_full_name="owner/repo", pr_number=5
        )

    assert "pulls/5/reviews" in mock_http.get.call_args[0][0]
    assert rows == [
        {
            "body": "LGTM overall",
            "author": "alice",
            "author_association": "CONTRIBUTOR",
            "event": "APPROVE",
        }
    ]


@pytest.mark.asyncio
async def test_get_pr_review_summaries_maps_states_to_events() -> None:
    """Each review state maps to its APPROVE/REQUEST_CHANGES/COMMENT event type."""
    rows = [
        _review(body="approve me", login="a", review_id=1, state="APPROVED"),
        _review(body="needs work", login="b", review_id=2, state="CHANGES_REQUESTED"),
        _review(body="just a note", login="c", review_id=3, state="COMMENTED"),
    ]
    mock_http = _single_page(rows)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        summaries = await client.get_pr_review_summaries(
            repo_full_name="owner/repo", pr_number=5
        )

    assert [s["event"] for s in summaries] == [
        "APPROVE",
        "REQUEST_CHANGES",
        "COMMENT",
    ]


@pytest.mark.asyncio
async def test_get_pr_review_summaries_drops_empty_body_reviews() -> None:
    """A bare review with no body (e.g. a click-approve) carries no summary text."""
    rows = [
        _review(body="", login="a", review_id=1, state="APPROVED"),
        _review(body="real feedback", login="b", review_id=2, state="COMMENTED"),
    ]
    mock_http = _single_page(rows)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=mock_http
        )
        summaries = await client.get_pr_review_summaries(
            repo_full_name="owner/repo", pr_number=5
        )

    assert [s["body"] for s in summaries] == ["real feedback"]


@pytest.mark.asyncio
async def test_get_pr_review_summaries_keeps_humans_and_heimdall_drops_bots() -> None:
    """The shared author filter applies: humans + Heimdall kept, other bots dropped."""
    app_id = 4242
    rows = [
        _review(body="from human", login="alice", review_id=1, state="COMMENTED"),
        _review(
            body="from CI bot",
            login="ci[bot]",
            review_id=2,
            state="COMMENTED",
            user_type="Bot",
        ),
        _review(
            body="heimdall prior",
            login="heimdall[bot]",
            review_id=3,
            state="CHANGES_REQUESTED",
            user_type="Bot",
            app_id=app_id,
        ),
        _review(
            body="other-bot review",
            login="dependabot[bot]",
            review_id=4,
            state="COMMENTED",
            user_type="Bot",
            app_id=9999,
        ),
    ]
    mock_http = _single_page(rows)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        summaries = await client.get_pr_review_summaries(
            repo_full_name="owner/repo", pr_number=5
        )

    assert [s["body"] for s in summaries] == ["from human", "heimdall prior"]


# ---------------------------------------------------------------------------
# Heimdall's own prior review: body + its inline comments, before retire/delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_own_prior_review_returns_body_and_inline_comments() -> None:
    """Heimdall's own latest review surfaces its body plus its shaped inline comments."""
    app_id = 4242
    reviews = [
        _review(body="human review", login="alice", review_id=1, state="COMMENTED"),
        _review(
            body="Heimdall review: 1 finding",
            login="heimdall[bot]",
            review_id=7,
            state="CHANGES_REQUESTED",
            user_type="Bot",
            app_id=app_id,
        ),
    ]

    inline = [
        _review_comment(body="inline nit", login="heimdall[bot]", comment_id=11),
    ]

    def _get(url: str, **_kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {}
        resp.json = MagicMock(return_value=inline if "reviews/7/comments" in url else reviews)
        return resp

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=_get)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        own = await client.get_own_prior_review(
            repo_full_name="owner/repo", pr_number=5
        )

    assert own is not None
    assert own["body"] == "Heimdall review: 1 finding"
    assert own["event"] == "REQUEST_CHANGES"
    assert own["author"] == "heimdall[bot]"
    assert [c["body"] for c in own["inline_comments"]] == ["inline nit"]
    assert own["inline_comments"][0]["path"] == "foo.py"
    assert own["inline_comments"][0]["line"] == 3


@pytest.mark.asyncio
async def test_get_own_prior_review_returns_latest_when_several() -> None:
    """When Heimdall posted several reviews, the most recent one is returned."""
    app_id = 4242
    reviews = [
        _review(
            body="older heimdall",
            login="heimdall[bot]",
            review_id=3,
            state="COMMENTED",
            user_type="Bot",
            app_id=app_id,
        ),
        _review(
            body="newer heimdall",
            login="heimdall[bot]",
            review_id=9,
            state="COMMENTED",
            user_type="Bot",
            app_id=app_id,
        ),
    ]

    def _get(url: str, **_kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {}
        resp.json = MagicMock(return_value=[] if "comments" in url else reviews)
        return resp

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=_get)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        own = await client.get_own_prior_review(
            repo_full_name="owner/repo", pr_number=5
        )

    assert own is not None
    assert own["body"] == "newer heimdall"
    # The latest review's inline comments are listed by its id (9), not an older one.
    listed = [c for c in mock_http.get.await_args_list if "reviews/9/comments" in str(c)]
    assert listed


@pytest.mark.asyncio
async def test_get_own_prior_review_none_when_no_own_review() -> None:
    """No Heimdall review on the PR yields None (nothing of our own to surface)."""
    app_id = 4242
    reviews = [_review(body="just humans", login="alice", review_id=1)]
    mock_http = _single_page(reviews)

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ):
        client = GitHubClient(
            app_id=app_id, private_key="key", installation_id=42, http_client=mock_http
        )
        own = await client.get_own_prior_review(
            repo_full_name="owner/repo", pr_number=5
        )

    assert own is None


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


# ---------------------------------------------------------------------------
# Timeout + read-path retry (transient network / 5xx resilience)
# ---------------------------------------------------------------------------


def test_owned_http_client_has_explicit_timeout_above_five_seconds() -> None:
    """The client's own httpx.AsyncClient sets explicit read/write timeouts.

    httpx's bare default (5s connect/read/write/pool) drops a routine slow GitHub
    response; read/write must be raised well above it so ordinary latency doesn't
    raise ReadTimeout.
    """
    client = GitHubClient(app_id=1, private_key="key", installation_id=42)
    timeout = client._http.timeout
    assert timeout.connect == 5.0
    assert timeout.read is not None and timeout.read >= 30.0
    assert timeout.write is not None and timeout.write >= 30.0


@pytest.mark.asyncio
async def test_get_pr_retries_transient_transport_failure_then_succeeds() -> None:
    """A transport that fails transiently then succeeds is retried and the call succeeds.

    Uses a REAL httpx.AsyncClient wired to a MockTransport so the retry loop is
    exercised against actual httpx call semantics, not a mocked-away retry.
    """
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < _MAX_READ_ATTEMPTS:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json={"number": 5, "title": "ok"})

    real_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ), patch("heimdall.github.asyncio.sleep", new=AsyncMock()):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=real_http
        )
        result = await client.get_pr(repo_full_name="owner/repo", pr_number=5)
    await real_http.aclose()

    assert result["number"] == 5
    assert call_count["n"] == _MAX_READ_ATTEMPTS


@pytest.mark.asyncio
async def test_get_pr_retries_transient_5xx_then_succeeds() -> None:
    """A transient 5xx response is retried, distinct from a transport-level failure."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"number": 9})

    real_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ), patch("heimdall.github.asyncio.sleep", new=AsyncMock()):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=real_http
        )
        result = await client.get_pr(repo_full_name="owner/repo", pr_number=9)
    await real_http.aclose()

    assert result["number"] == 9
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_get_pr_persistent_transport_failure_surfaces_after_retry_bound() -> None:
    """A persistently failing transport still raises once the retry bound is spent."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        raise httpx.ConnectError("connection reset", request=request)

    real_http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with patch.object(
        GitHubClient, "get_installation_token", new=AsyncMock(return_value="ghs_tok")
    ), patch("heimdall.github.asyncio.sleep", new=AsyncMock()):
        client = GitHubClient(
            app_id=1, private_key="key", installation_id=42, http_client=real_http
        )
        with pytest.raises(httpx.ConnectError):
            await client.get_pr(repo_full_name="owner/repo", pr_number=5)
    await real_http.aclose()

    assert call_count["n"] == _MAX_READ_ATTEMPTS
