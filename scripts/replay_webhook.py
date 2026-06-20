"""Replay a signed ``pull_request`` webhook at a running Heimdall service.

Dogfooding without a public tunnel: GitHub never delivers to a localhost service, so
this builds the same ``pull_request`` payload GitHub would send, signs it with the shared
``WEBHOOK_SECRET`` exactly as :func:`heimdall.webhook.compute_signature` expects, and POSTs
it to ``/webhook``.  The worker then fetches the real PR via the installation token and
posts a real review — so the App credentials in the service's ``.env`` must be for a real
installed App even though the webhook itself is replayed.

Usage:
    uv run python scripts/replay_webhook.py \\
        --repo owner/repo --pr 42 --sha <head-sha> --installation-id 12345

``--secret`` defaults to ``$WEBHOOK_SECRET`` and ``--url`` to the local web service.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import httpx

from heimdall.webhook import compute_signature

_DEFAULT_URL = "http://localhost:8000/webhook"

# GitHub's pull_request webhook always carries the full 40-hex head SHA.
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git_rev_parse(sha: str) -> str:
    """Resolve ``sha`` to a full commit OID via ``git rev-parse`` in the local repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def resolve_commit_sha(sha: str, *, run: Callable[[str], str] = _git_rev_parse) -> str:
    """Expand a possibly-abbreviated ``--sha`` to the full 40-char commit OID.

    GitHub's create-review REST endpoint rejects an abbreviated ``commit_id`` with a
    422 ("Variable $commitOID of type GitObjectID was provided invalid value"), and
    the real ``pull_request`` webhook always sends the full head SHA.  To mirror GitHub
    faithfully the replay expands a short sha against the local repo before forwarding
    it, so the dogfood exercises the same input production would receive.

    Args:
        sha: The ``--sha`` value (full or abbreviated commit-ish).
        run: Resolver mapping a commit-ish to ``git rev-parse`` output (injectable for
            tests).

    Returns:
        The full 40-hex commit SHA.

    Raises:
        ValueError: If ``sha`` cannot be resolved to a full commit SHA.
    """
    sha = sha.strip()
    if _FULL_SHA.match(sha):
        return sha
    resolved = run(sha).strip()
    if not _FULL_SHA.match(resolved):
        raise ValueError(
            f"could not resolve {sha!r} to a full commit SHA (got {resolved!r})"
        )
    return resolved


def build_pull_request_payload(
    *,
    action: str,
    pr_number: int,
    head_sha: str,
    repo_full_name: str,
    installation_id: int,
    draft: bool = False,
) -> dict[str, Any]:
    """Build the ``pull_request`` webhook body the service's ``_build_job`` reads.

    Mirrors the fields GitHub sends and Heimdall consumes: the top-level ``action``,
    ``pull_request.number``/``head.sha``/``draft``, ``repository.full_name``, and
    ``installation.id``.
    """
    return {
        "action": action,
        "number": pr_number,
        "pull_request": {
            "number": pr_number,
            "head": {"sha": head_sha},
            "draft": draft,
        },
        "repository": {"full_name": repo_full_name},
        "installation": {"id": installation_id},
    }


def post_replay(
    *,
    url: str,
    secret: str,
    payload: dict[str, Any],
    client: httpx.Client,
) -> httpx.Response:
    """Sign ``payload`` and POST it as a ``pull_request`` webhook over ``client``.

    Signs the exact JSON bytes that are sent (the signature must match the body GitHub's
    receiver verifies), and sets the ``X-GitHub-Event`` / ``X-Hub-Signature-256`` headers.
    """
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": compute_signature(body, secret),
    }
    return client.post(url, content=body, headers=headers)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Repository full name, e.g. owner/repo")
    parser.add_argument("--pr", type=int, required=True, help="Pull-request number")
    parser.add_argument("--sha", required=True, help="Head commit SHA to review")
    parser.add_argument(
        "--installation-id",
        type=int,
        required=True,
        help="GitHub App installation ID for the repo",
    )
    parser.add_argument(
        "--action",
        default="opened",
        help="pull_request action (opened/reopened/synchronize/ready_for_review)",
    )
    parser.add_argument("--draft", action="store_true", help="Mark the PR as a draft")
    parser.add_argument(
        "--url", default=_DEFAULT_URL, help=f"Webhook URL (default: {_DEFAULT_URL})"
    )
    parser.add_argument(
        "--secret",
        default=os.environ.get("WEBHOOK_SECRET"),
        help="Webhook secret (default: $WEBHOOK_SECRET)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build, sign, and POST a replayed pull_request webhook.

    Returns a process exit code: 0 when the service accepts the replay (2xx),
    1 on a missing secret or a non-2xx response.
    """
    args = _parse_args(argv)
    if not args.secret:
        print("error: no webhook secret (pass --secret or set WEBHOOK_SECRET)", file=sys.stderr)
        return 1

    try:
        head_sha = resolve_commit_sha(args.sha)
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: could not resolve --sha {args.sha!r}: {exc}", file=sys.stderr)
        return 1

    payload = build_pull_request_payload(
        action=args.action,
        pr_number=args.pr,
        head_sha=head_sha,
        repo_full_name=args.repo,
        installation_id=args.installation_id,
        draft=args.draft,
    )
    with httpx.Client() as client:
        response = post_replay(url=args.url, secret=args.secret, payload=payload, client=client)

    print(f"{response.status_code} {response.reason_phrase}")
    if response.text:
        print(response.text)
    return 0 if response.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
