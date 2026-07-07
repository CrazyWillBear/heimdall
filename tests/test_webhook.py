"""Tests for the webhook endpoint: signature validation, enqueue, async ack."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from heimdall.app import create_app
from heimdall.config import Settings
from heimdall.queue import ReviewJob
from heimdall.webhook import compute_signature, verify_signature

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECRET = "test-webhook-secret"


def _make_settings() -> Settings:
    return Settings(
        webhook_secret=_SECRET,
        github_app_id=12345,
        github_app_private_key="dummy-key",
        redis_url="redis://localhost:6379",
        database_url="sqlite+aiosqlite:///./test.db",
    )


def _sign(payload: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _pr_payload(action: str = "opened", pr_number: int = 1, head_sha: str = "abc123") -> dict:  # type: ignore[type-arg]
    return {
        "action": action,
        "number": pr_number,
        "pull_request": {
            "number": pr_number,
            "head": {"sha": head_sha},
            "draft": False,
        },
        "repository": {
            "id": 999,
            "full_name": "owner/repo",
        },
        "installation": {"id": 42},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client() -> Iterator[tuple[TestClient, MagicMock]]:
    """Yield (TestClient, mock_enqueue) with the patch active for the whole test."""
    settings = _make_settings()
    mock_enqueue = AsyncMock(return_value="jid-test")
    with patch("heimdall.webhook.enqueue_review", mock_enqueue):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=True)
        yield client, mock_enqueue


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def test_compute_signature_matches_github_format() -> None:
    """compute_signature returns the ``sha256=<hex>`` header GitHub sends."""
    payload = b'{"hello": "world"}'
    assert compute_signature(payload, _SECRET) == _sign(payload, _SECRET)


def test_verify_accepts_compute_signature_output() -> None:
    """verify_signature accepts a header produced by compute_signature (round-trip)."""
    payload = b"some-body-bytes"
    # Must not raise: the verify path and the signer share one implementation.
    verify_signature(payload, _SECRET, compute_signature(payload, _SECRET))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_webhook_returns_202(app_client: tuple[TestClient, MagicMock]) -> None:
    """A well-formed PR webhook returns 202 immediately."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 202
    mock_enqueue.assert_called_once()


def test_enqueue_failure_returns_5xx() -> None:
    """If enqueue raises, the handler returns 5xx (not 202) so GitHub redelivers.

    The enqueue is awaited inline before the ack; a failure must not be swallowed by a
    background task after a 202 has already been sent. Uses raise_server_exceptions=False
    so the TestClient surfaces the 500 response instead of re-raising.
    """
    settings = _make_settings()
    failing_enqueue = AsyncMock(side_effect=RuntimeError("redis down"))
    payload = json.dumps(_pr_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    with patch("heimdall.webhook.enqueue_review", failing_enqueue):
        app = create_app(settings)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code >= 500
    failing_enqueue.assert_called_once()


def test_invalid_signature_rejected(app_client: tuple[TestClient, MagicMock]) -> None:
    """An invalid signature returns 401 and no job is enqueued."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": "sha256=bad",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    mock_enqueue.assert_not_called()


def test_missing_signature_rejected(app_client: tuple[TestClient, MagicMock]) -> None:
    """Missing signature header returns 401 and no job is enqueued."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 401
    mock_enqueue.assert_not_called()


def test_non_pr_event_ignored(app_client: tuple[TestClient, MagicMock]) -> None:
    """Non pull_request events return 204 without enqueuing."""
    client, mock_enqueue = app_client
    payload = json.dumps({"action": "created"}).encode()
    headers = {
        "X-GitHub-Event": "issue_comment",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


def test_draft_pr_ignored(app_client: tuple[TestClient, MagicMock]) -> None:
    """Draft PRs are silently ignored (204)."""
    client, mock_enqueue = app_client
    data = _pr_payload()
    data["pull_request"]["draft"] = True
    payload = json.dumps(data).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


def test_irrelevant_action_ignored(app_client: tuple[TestClient, MagicMock]) -> None:
    """PR actions other than opened/reopened/synchronize/ready_for_review/review_requested
    are ignored."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload(action="closed")).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


def test_review_requested_enqueues(app_client: tuple[TestClient, MagicMock]) -> None:
    """A review_requested event enqueues a review job carrying that action."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload(action="review_requested")).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 202
    mock_enqueue.assert_called_once()
    job: ReviewJob = mock_enqueue.call_args.args[1]
    assert job.action == "review_requested"


def test_review_requested_on_draft_ignored(
    app_client: tuple[TestClient, MagicMock],
) -> None:
    """A review_requested on a draft PR is ignored (204), nothing enqueued."""
    client, mock_enqueue = app_client
    data = _pr_payload(action="review_requested")
    data["pull_request"]["draft"] = True
    payload = json.dumps(data).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 204
    mock_enqueue.assert_not_called()


@pytest.mark.parametrize(
    "action",
    ["opened", "reopened", "synchronize", "ready_for_review", "review_requested"],
)
def test_job_carries_action(
    app_client: tuple[TestClient, MagicMock], action: str
) -> None:
    """The enqueued ReviewJob's action matches the payload for every relevant action."""
    client, mock_enqueue = app_client
    payload = json.dumps(_pr_payload(action=action)).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    response = client.post("/webhook", content=payload, headers=headers)
    assert response.status_code == 202
    job: ReviewJob = mock_enqueue.call_args.args[1]
    assert job.action == action


# ---------------------------------------------------------------------------
# Lifespan / app.state pool wiring tests
# ---------------------------------------------------------------------------


def test_lifespan_creates_and_closes_pool() -> None:
    """The lifespan creates an Arq pool on startup and closes it on shutdown."""
    settings = _make_settings()
    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()

    with (
        patch("arq.create_pool", return_value=mock_pool) as mock_create,
        patch("heimdall.webhook.enqueue_review", AsyncMock(return_value="jid")),
    ):
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=True):
            # Inside the context manager the lifespan startup has run.
            mock_create.assert_called_once()
        # After exiting the context manager the lifespan shutdown has run.
        mock_pool.close.assert_called_once()


def test_webhook_reads_pool_from_app_state() -> None:
    """The webhook handler uses the pool from app.state, not a captured closure value.

    Patches arq.create_pool so the lifespan doesn't need a real Redis.  The pool
    placed on app.state by the lifespan is what the handler must receive.
    """
    settings = _make_settings()
    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()
    captured_pools: list[object] = []

    async def fake_enqueue(pool: object, job: ReviewJob) -> str:
        captured_pools.append(pool)
        return "jid"

    payload = json.dumps(_pr_payload()).encode()
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": _sign(payload, _SECRET),
        "Content-Type": "application/json",
    }
    with (
        patch("arq.create_pool", return_value=mock_pool),
        patch("heimdall.webhook.enqueue_review", side_effect=fake_enqueue),
    ):
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=True) as client:
            response = client.post("/webhook", content=payload, headers=headers)

    assert response.status_code == 202
    assert len(captured_pools) == 1
    # The handler must have received the pool the lifespan put on app.state.
    assert captured_pools[0] is mock_pool
