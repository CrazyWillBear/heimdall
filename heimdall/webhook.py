"""FastAPI webhook router: signature verification and PR event dispatch.

Verifies the HMAC-SHA256 signature on every incoming webhook, ignores irrelevant
events, and enqueues a review job for actionable pull_request events.  The enqueue is
awaited inline before the 202 so a failed enqueue surfaces as 5xx (GitHub redelivers)
rather than vanishing after the ack; the review itself runs asynchronously in the worker.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response

from heimdall.queue import ReviewJob, enqueue_review

logger = logging.getLogger(__name__)

router = APIRouter()

_RELEVANT_ACTIONS = frozenset(
    {"opened", "reopened", "synchronize", "ready_for_review", "review_requested"}
)


def compute_signature(payload: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` HMAC header GitHub puts in ``X-Hub-Signature-256``.

    The single source of truth for the webhook HMAC: :func:`verify_signature` checks
    against it and the replay helper signs with it, so the verify and sign paths can
    never drift.

    Args:
        payload: The raw request body bytes.
        secret: The webhook secret configured on the GitHub App.

    Returns:
        The signature header value, ``sha256=`` followed by the hex HMAC-SHA256 digest.
    """
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, secret: str, signature_header: str | None) -> None:
    """Raise HTTP 401 if the webhook signature is absent or invalid.

    GitHub signs every webhook with HMAC-SHA256 using the configured secret and
    puts the result in the ``X-Hub-Signature-256`` header as ``sha256=<hex>``.

    Args:
        payload: The raw request body bytes.
        secret: The webhook secret configured on the GitHub App.
        signature_header: The value of ``X-Hub-Signature-256``.

    Raises:
        HTTPException: 401 if the signature is missing or doesn't match.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing webhook signature")
    expected = compute_signature(payload, secret)
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


def _build_job(body: dict[str, Any]) -> ReviewJob:
    pr = body["pull_request"]
    return ReviewJob(
        installation_id=body["installation"]["id"],
        repo_full_name=body["repository"]["full_name"],
        pr_number=pr["number"],
        head_sha=pr["head"]["sha"],
        action=body["action"],
    )


def make_webhook_router(*, webhook_secret: str) -> APIRouter:
    """Return a configured APIRouter with the /webhook POST endpoint.

    The handler reads the Arq pool from ``request.app.state.arq_pool`` at
    request time so it picks up the pool created by the app's lifespan hook,
    even though the router is built before the lifespan runs.

    Args:
        webhook_secret: The HMAC secret to verify incoming requests.
    """
    router = APIRouter()

    @router.post("/webhook", status_code=202)
    async def handle_webhook(
        request: Request,
        x_github_event: str | None = Header(default=None),
        x_hub_signature_256: str | None = Header(default=None),
    ) -> Response:
        """Receive a GitHub webhook, verify it, and enqueue a review if relevant."""
        payload = await request.body()
        verify_signature(payload, webhook_secret, x_hub_signature_256)

        if x_github_event != "pull_request":
            return Response(status_code=204)

        body: dict[str, Any] = await request.json()
        action: str = body.get("action", "")
        pr: dict[str, Any] = body.get("pull_request", {})

        if action not in _RELEVANT_ACTIONS:
            return Response(status_code=204)

        if pr.get("draft", False):
            return Response(status_code=204)

        job = _build_job(body)
        # Resolve the pool at request time from app.state so the lifespan-created
        # pool is always used, regardless of when the router was constructed.
        pool = request.app.state.arq_pool
        # Enqueue inline before acking: if it raises, the handler returns 5xx and
        # GitHub redelivers, rather than a background failure silently dropping the
        # job after a 202 has already been sent.
        await enqueue_review(pool, job)

        logger.info(
            "Enqueued review for %s#%d action=%s sha=%s",
            job.repo_full_name,
            job.pr_number,
            action,
            job.head_sha,
        )
        return Response(status_code=202)

    return router
