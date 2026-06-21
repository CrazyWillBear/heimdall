"""Tests for the OAuth token-refresher sidecar (``heimdall-refresh``).

The refresher runs **unsandboxed** and its only job is to keep subscription
(OAuth) credentials in ``~/.claude`` warm by periodically running a throwaway
``claude -p`` ping — the model refreshes (and rotates) the token as a side
effect of any call, which the read-only lens sandbox can never persist.  These
tests pin the ping argv and the single-shot runner; the real ``claude`` binary
is never spawned (the subprocess call is patched).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from heimdall.refresh import (
    DEFAULT_REFRESH_INTERVAL_SECONDS,
    DEFAULT_REFRESH_MODEL,
    build_refresh_argv,
    run_refresh_once,
)


def test_default_refresh_model_is_haiku() -> None:
    """The cheapest model is the default — the response is discarded anyway."""
    assert DEFAULT_REFRESH_MODEL == "haiku"


def test_default_interval_is_under_token_lifetime() -> None:
    """30 min comfortably beats the hours-long OAuth token lifetime."""
    assert DEFAULT_REFRESH_INTERVAL_SECONDS == 1800.0


def test_refresh_argv_is_a_single_turn_headless_ping() -> None:
    """The argv is a minimal one-turn headless ping on the chosen model."""
    argv = build_refresh_argv(claude_binary="claude", model="haiku")
    assert argv[0] == "claude"
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "ping"
    assert argv[argv.index("--model") + 1] == "haiku"
    assert argv[argv.index("--max-turns") + 1] == "1"


def test_refresh_argv_threads_the_model() -> None:
    """A custom model name reaches the argv verbatim (configurable knob)."""
    argv = build_refresh_argv(claude_binary="claude", model="claude-opus-4-8")
    assert argv[argv.index("--model") + 1] == "claude-opus-4-8"


def test_refresh_argv_honours_a_custom_binary_path() -> None:
    """An absolute/custom claude path is used as argv[0], not the bare name."""
    argv = build_refresh_argv(claude_binary="/opt/claude/bin/claude", model="haiku")
    assert argv[0] == "/opt/claude/bin/claude"


def test_refresh_argv_grants_no_tools_and_no_workspace() -> None:
    """The ping reasons over nothing — no tools, no --add-dir, no system prompt."""
    argv = build_refresh_argv(claude_binary="claude", model="haiku")
    for flag in ("--allowedTools", "--add-dir", "--append-system-prompt"):
        assert flag not in argv


def test_run_refresh_once_spawns_the_ping_and_returns_rc() -> None:
    """run_refresh_once invokes the built argv and surfaces the exit code."""
    completed = MagicMock(returncode=0)
    with patch("heimdall.refresh.subprocess.run", return_value=completed) as run:
        rc = run_refresh_once(claude_binary="claude", model="haiku", timeout_seconds=60.0)

    assert rc == 0
    argv = run.call_args.args[0]
    assert argv == build_refresh_argv(claude_binary="claude", model="haiku")
    # A timeout must be enforced so a hung ping can't wedge the refresh loop.
    assert run.call_args.kwargs["timeout"] == 60.0


def test_run_refresh_once_surfaces_nonzero_rc() -> None:
    """A failed ping (expired/unseeded creds) returns its non-zero code, not a raise."""
    completed = MagicMock(returncode=1)
    with patch("heimdall.refresh.subprocess.run", return_value=completed):
        rc = run_refresh_once(claude_binary="claude", model="haiku", timeout_seconds=60.0)
    assert rc == 1


def test_run_refresh_once_treats_timeout_as_failure() -> None:
    """A hung ping that times out is reported as a non-zero rc, never propagates."""
    import subprocess

    exc = subprocess.TimeoutExpired(cmd="claude", timeout=60.0)
    with patch("heimdall.refresh.subprocess.run", side_effect=exc):
        rc = run_refresh_once(claude_binary="claude", model="haiku", timeout_seconds=60.0)
    assert rc != 0


# ---------------------------------------------------------------------------
# Doc-completeness: the refresher's env knobs must stay documented (no drift).
# ---------------------------------------------------------------------------

_README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_documents_refresh_env_vars() -> None:
    """The refresher's configurable env vars appear in the README by name."""
    text = _README.read_text(encoding="utf-8")
    for var in ("CLAUDE_REFRESH_MODEL", "CLAUDE_REFRESH_INTERVAL_SECONDS"):
        assert var in text, f"README is missing refresher env var: {var}"
