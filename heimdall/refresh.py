"""OAuth (subscription) token-refresher sidecar — the ``heimdall-refresh`` command.

The lens sandbox binds ``~/.claude`` **read-only** (``heimdall/lens.py``
``build_bwrap_prefix``) so a prompt-injected PR can never tamper with the worker's
credentials.  OAuth access tokens expire after hours, and the ``claude`` CLI refreshes
them by rewriting ``~/.claude/.credentials.json`` — a write the sandbox blocks by
design.  On a developer's box ordinary ``claude`` use keeps the token warm; on a
headless VPS nothing does, so subscription auth dies after the first expiry.

This sidecar closes that gap.  It runs **unsandboxed** beside the worker, sharing the
same ``~/.claude`` (a Compose named volume) **read-write**, and periodically fires a
throwaway ``claude -p`` ping.  Any successful call refreshes — and rotates — the token
as a side effect; the response itself is discarded.  Because the refresher runs **no
untrusted PR code**, write access to the credentials is safe, and the worker's lens
sandbox keeps binding the very same volume read-only — the security control is
unchanged (see the README "Docker deployment" → OAuth section).

It is opt-in: the default deployment authenticates via ``ANTHROPIC_API_KEY`` and needs
no refresher.  Deliberately decoupled from :class:`heimdall.config.Settings` — the
refresher runs no GitHub code, so its container is given **none** of the App secrets;
it reads only the few knobs it needs straight from the environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# The cheapest model: token refresh is the only goal, so the reply is thrown away.
DEFAULT_REFRESH_MODEL = "haiku"

# OAuth access tokens live for hours; pinging twice an hour keeps one comfortably warm
# while costing next to nothing.
DEFAULT_REFRESH_INTERVAL_SECONDS = 1800.0

# A refresh ping is one trivial turn; cap it so a hung/stalled call can never wedge the
# loop and silently stop refreshing.
_REFRESH_TIMEOUT_SECONDS = 120.0

# Non-zero code reported when the ping is killed for exceeding its timeout (mirrors the
# shell convention for a timeout-terminated process) so the loop logs a failure rather
# than crashing.
_TIMEOUT_RC = 124


def build_refresh_argv(*, claude_binary: str, model: str) -> list[str]:
    """Build the ``claude -p`` argv for a single throwaway token-refresh ping.

    Headless ``-p`` on the chosen model, capped at one turn — no tools, no workspace,
    no system prompt: the call exists only so the CLI refreshes its OAuth token, and
    its output is discarded.  argv is consumed by ``subprocess.run`` (no shell), so
    none of these strings are shell-interpreted.

    Args:
        claude_binary: Path or name of the ``claude`` executable.
        model: Model to ping; cheap models suffice since the reply is unused.

    Returns:
        The argument vector to pass to :func:`run_refresh_once`.
    """
    return [
        claude_binary,
        "-p",
        "ping",
        "--model",
        model,
        "--max-turns",
        "1",
    ]


def run_refresh_once(
    *, claude_binary: str, model: str, timeout_seconds: float = _REFRESH_TIMEOUT_SECONDS
) -> int:
    """Fire one refresh ping and return its exit code (never raise on failure).

    Runs unsandboxed, inheriting the full environment so the ``claude`` child sees the
    shared read-write ``~/.claude`` (via ``HOME``) and can persist a refreshed/rotated
    token.  Output is captured and discarded except that a non-zero exit logs a short
    stderr tail to make an expired/unseeded-credentials failure diagnosable.  A ping
    that overruns ``timeout_seconds`` is killed and reported as a failure so the loop
    keeps going.

    Args:
        claude_binary: Path or name of the ``claude`` executable.
        model: Model to ping.
        timeout_seconds: Wall-clock cap before the ping is killed.

    Returns:
        The ``claude`` exit code, or :data:`_TIMEOUT_RC` if it was killed for timing out.
    """
    argv = build_refresh_argv(claude_binary=claude_binary, model=model)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Refresh ping timed out after %.0fs", timeout_seconds)
        return _TIMEOUT_RC

    if completed.returncode == 0:
        logger.info("Refresh ping succeeded; OAuth token warm")
    else:
        tail = (completed.stderr or "").strip()[-500:]
        logger.warning(
            "Refresh ping failed (rc=%d); check the seeded credentials. stderr: %s",
            completed.returncode,
            tail,
        )
    return completed.returncode


def _refresh_config() -> tuple[str, str, float]:
    """Read the refresher's knobs from the environment, falling back to defaults.

    Read directly (not via :class:`~heimdall.config.Settings`) so the refresher
    container never has to carry the GitHub App secrets it does not use.

    Returns:
        ``(claude_binary, model, interval_seconds)``.
    """
    binary = os.environ.get("CLAUDE_BINARY", "claude")
    model = os.environ.get("CLAUDE_REFRESH_MODEL", DEFAULT_REFRESH_MODEL)
    interval = float(
        os.environ.get(
            "CLAUDE_REFRESH_INTERVAL_SECONDS", DEFAULT_REFRESH_INTERVAL_SECONDS
        )
    )
    return binary, model, interval


def main() -> None:
    """Console-script entrypoint (``heimdall-refresh``): the refresh loop.

    Pings once immediately to warm the token at startup, then sleeps the configured
    interval and repeats forever.  Logs at INFO to stdout (the image sets
    ``PYTHONUNBUFFERED`` so lines stream) and exits cleanly on SIGINT/SIGTERM.
    """
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    binary, model, interval = _refresh_config()
    logger.info(
        "Starting OAuth refresher: model=%s interval=%.0fs binary=%s",
        model,
        interval,
        binary,
    )
    try:
        while True:
            run_refresh_once(claude_binary=binary, model=model)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Refresher interrupted; shutting down")


if __name__ == "__main__":
    main()
