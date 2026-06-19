"""Tests for the reusable lens runner: claude -p invocation, parsing, caps, verdict.

The claude subprocess is always mocked here via an injected ClaudeInvoker so no
real claude binary is ever spawned.  These tests pin the contract that issue #5
(two more lenses + synthesis) will call into.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from heimdall.lens import (
    SECURITY_LENS,
    ClaudeResult,
    Finding,
    LensResult,
    LensTimeoutError,
    LensTokenCapError,
    Severity,
    build_claude_argv,
    format_review_body,
    parse_findings,
    run_lens,
    verdict_for,
)

_WORKSPACE = "/tmp/heimdall-ws"


def _claude_result(findings: list[dict[str, Any]], *, tokens: int = 100) -> ClaudeResult:
    """Build a ClaudeResult whose stdout carries a JSON findings block."""
    payload = {"findings": findings}
    return ClaudeResult(stdout=json.dumps(payload), total_tokens=tokens, killed=False)


# ---------------------------------------------------------------------------
# Finding parsing
# ---------------------------------------------------------------------------


def test_parse_findings_extracts_severity_and_message() -> None:
    """A claude JSON block maps to Finding objects with parsed severity."""
    raw = json.dumps(
        {
            "findings": [
                {
                    "severity": "high",
                    "title": "SQL injection",
                    "message": "User input concatenated into query",
                    "location": "app/db.py:12",
                }
            ]
        }
    )
    findings = parse_findings(raw)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity is Severity.HIGH
    assert f.title == "SQL injection"
    assert "concatenated" in f.message
    assert f.location == "app/db.py:12"


def test_parse_findings_tolerates_prose_around_json() -> None:
    """Claude may wrap the JSON block in prose; parsing still finds it."""
    raw = (
        "Here is my review.\n\n"
        '{"findings": [{"severity": "critical", "title": "RCE", "message": "eval"}]}'
        "\n\nThat concludes my review."
    )
    findings = parse_findings(raw)
    assert len(findings) == 1
    assert findings[0].severity is Severity.CRITICAL


def test_parse_findings_empty_when_no_findings() -> None:
    """A clean PR yields an empty findings list, not an error."""
    assert parse_findings(json.dumps({"findings": []})) == []


def test_parse_findings_unknown_severity_falls_back_to_low() -> None:
    """An unrecognised severity string degrades to LOW rather than crashing."""
    raw = json.dumps({"findings": [{"severity": "spicy", "message": "weird"}]})
    findings = parse_findings(raw)
    assert findings[0].severity is Severity.LOW


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


def test_verdict_request_changes_on_high() -> None:
    """A high finding maps to REQUEST_CHANGES."""
    findings = [Finding(severity=Severity.HIGH, title="x", message="m", location=None)]
    assert verdict_for(findings) == "REQUEST_CHANGES"


def test_verdict_request_changes_on_critical() -> None:
    """A critical finding maps to REQUEST_CHANGES."""
    findings = [Finding(severity=Severity.CRITICAL, title="x", message="m", location=None)]
    assert verdict_for(findings) == "REQUEST_CHANGES"


def test_verdict_comment_when_only_medium_and_low() -> None:
    """Medium/low findings (no high/critical) map to COMMENT."""
    findings = [
        Finding(severity=Severity.MEDIUM, title="x", message="m", location=None),
        Finding(severity=Severity.LOW, title="y", message="n", location=None),
    ]
    assert verdict_for(findings) == "COMMENT"


def test_verdict_comment_when_no_findings() -> None:
    """No findings maps to COMMENT."""
    assert verdict_for([]) == "COMMENT"


# ---------------------------------------------------------------------------
# Review body formatting
# ---------------------------------------------------------------------------


def test_format_review_body_lists_findings_worst_first() -> None:
    """Findings render worst-first with severity, title, location, and message."""
    findings = [
        Finding(severity=Severity.LOW, title="Nit", message="minor", location="a.py:1"),
        Finding(severity=Severity.CRITICAL, title="RCE", message="eval", location="b.py:9"),
    ]
    body = format_review_body(findings)
    assert body.index("RCE") < body.index("Nit")
    assert "CRITICAL" in body
    assert "b.py:9" in body
    assert "eval" in body


def test_format_review_body_all_clear_when_empty() -> None:
    """An empty finding list yields a short all-clear message."""
    body = format_review_body([])
    assert "no security concerns" in body.lower()


# ---------------------------------------------------------------------------
# build_claude_argv: read-only tools + heimdall-context allowlist only
# ---------------------------------------------------------------------------


def test_argv_uses_print_and_opus_max_effort() -> None:
    """The invocation runs headless (-p) on opus at max effort with JSON output."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert argv[0] == "claude"
    assert "-p" in argv or "--print" in argv
    assert "--model" in argv
    model = argv[argv.index("--model") + 1]
    assert "opus" in model
    assert "--effort" in argv
    assert argv[argv.index("--effort") + 1] == "max"
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "json"


def test_argv_allows_only_readonly_tools_and_heimdall_context_bash() -> None:
    """Allowed tools are Read/Grep/Glob plus the heimdall-context wrapper only."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert "--allowedTools" in argv
    allowed = argv[argv.index("--allowedTools") + 1]
    assert "Read" in allowed
    assert "Grep" in allowed
    assert "Glob" in allowed
    assert "heimdall-context" in allowed


def test_argv_forbids_raw_bash_and_write_tools() -> None:
    """No bare Bash, and no mutating tools (Write/Edit) are permitted."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    allowed = argv[argv.index("--allowedTools") + 1]
    # A bare "Bash" with no command scope would allow arbitrary shell — forbid it.
    tools = [t.strip() for t in allowed.replace(",", " ").split()]
    assert "Bash" not in tools
    assert "Write" not in allowed
    assert "Edit" not in allowed
    # Bash is only present scoped to the wrapper.
    assert any("Bash(heimdall-context" in t for t in tools)


def test_argv_disallows_raw_bash_explicitly() -> None:
    """Bash and Write are explicitly disallowed as a defense-in-depth measure."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert "--disallowedTools" in argv
    disallowed = argv[argv.index("--disallowedTools") + 1]
    assert "Bash" in disallowed
    assert "Write" in disallowed
    assert "Edit" in disallowed


def test_argv_scopes_session_to_workspace() -> None:
    """The workspace directory is added so claude reads from the seed context."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert "--add-dir" in argv
    assert _WORKSPACE in argv


# ---------------------------------------------------------------------------
# run_lens: end-to-end with a mocked invoker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_lens_returns_planted_finding() -> None:
    """A planted (mocked) security finding surfaces in the LensResult."""
    planted = [
        {
            "severity": "high",
            "title": "Hardcoded secret",
            "message": "AWS key committed in config",
            "location": "config.py:3",
        }
    ]

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int
    ) -> ClaudeResult:
        return _claude_result(planted)

    result = await run_lens(
        lens=SECURITY_LENS,
        workspace_dir=_WORKSPACE,
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )
    assert isinstance(result, LensResult)
    assert result.lens_name == SECURITY_LENS.name
    assert len(result.findings) == 1
    assert result.findings[0].title == "Hardcoded secret"
    assert result.findings[0].severity is Severity.HIGH


@pytest.mark.asyncio
async def test_run_lens_passes_caps_to_invoker() -> None:
    """run_lens threads the token cap and timeout through to the invoker."""
    captured: dict[str, Any] = {}

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int
    ) -> ClaudeResult:
        captured["timeout"] = timeout_seconds
        captured["cap"] = token_cap
        captured["argv"] = argv
        return _claude_result([])

    await run_lens(
        lens=SECURITY_LENS,
        workspace_dir=_WORKSPACE,
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )
    assert captured["cap"] == 400_000
    assert captured["timeout"] == 900
    assert captured["argv"][0] == "claude"


@pytest.mark.asyncio
async def test_run_lens_propagates_timeout() -> None:
    """A timeout in the invoker surfaces as LensTimeoutError (subprocess killed)."""

    async def timing_out_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int
    ) -> ClaudeResult:
        raise LensTimeoutError("wall-clock exceeded; subprocess killed")

    with pytest.raises(LensTimeoutError):
        await run_lens(
            lens=SECURITY_LENS,
            workspace_dir=_WORKSPACE,
            claude_binary="claude",
            token_cap=400_000,
            timeout_seconds=1,
            invoker=timing_out_invoker,
        )


@pytest.mark.asyncio
async def test_run_lens_propagates_token_cap() -> None:
    """Exceeding the token cap in the invoker surfaces as LensTokenCapError."""

    async def capping_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int
    ) -> ClaudeResult:
        raise LensTokenCapError("token cap exceeded; subprocess killed")

    with pytest.raises(LensTokenCapError):
        await run_lens(
            lens=SECURITY_LENS,
            workspace_dir=_WORKSPACE,
            claude_binary="claude",
            token_cap=10,
            timeout_seconds=900,
            invoker=capping_invoker,
        )
