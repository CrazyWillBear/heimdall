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
    """PR actions other than opened/reopened/synchronize/ready_for_review are ignored."""
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
