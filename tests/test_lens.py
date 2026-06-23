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
    CLEANLINESS_LENS,
    DESIGN_LENS,
    SECURITY_LENS,
    ClaudeResult,
    Finding,
    LensOutputError,
    LensResult,
    LensSpec,
    LensTimeoutError,
    LensTokenCapError,
    Severity,
    SuppressedFinding,
    SynthesisResult,
    _build_subprocess_env,
    build_claude_argv,
    format_review_body,
    parse_findings,
    render_comments_truncated_note,
    render_dropped_lenses_warning,
    render_suppressed_findings_section,
    run_lens,
    run_synthesis,
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


def test_argv_disallows_mutating_tools_without_subsuming_bash_wrapper() -> None:
    """Write/Edit are denied, but no unscoped Bash deny subsumes the wrapper.

    Deny rules take precedence over allow rules, and an unscoped ``Bash`` deny
    matches every Bash invocation — including ``Bash(heimdall-context *)``. Such
    a token would neuter the allowlisted wrapper at runtime, so the deny list must
    not contain a bare ``Bash``. Raw Bash stays blocked by default-deny anyway.
    """
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert "--disallowedTools" in argv
    disallowed = argv[argv.index("--disallowedTools") + 1]
    disallowed_tools = [t.strip() for t in disallowed.replace(",", " ").split()]
    # Mutating tools remain explicitly denied.
    assert "Write" in disallowed_tools
    assert "Edit" in disallowed_tools
    # No unscoped Bash deny — it would subsume the Bash(heimdall-context *) allow.
    assert "Bash" not in disallowed_tools
    assert not any(t.startswith("Bash") and "heimdall-context" in t for t in disallowed_tools)


def test_argv_allow_deny_lets_wrapper_through_but_blocks_raw_bash() -> None:
    """The allow/deny pair permits the wrapper while raw Bash stays denied.

    Pins the allow/deny interaction: ``Bash(heimdall-context *)`` is the only Bash
    form on the allow list, and the deny list carries no token that would also
    match it. Under default-deny, any other ``Bash(...)`` is rejected because it is
    simply not allowed — without an unscoped deny that would kill the wrapper too.
    """
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    allowed = [t.strip() for t in argv[argv.index("--allowedTools") + 1].replace(",", " ").split()]
    disallowed = [
        t.strip() for t in argv[argv.index("--disallowedTools") + 1].replace(",", " ").split()
    ]
    # The wrapper is allowed and nothing in deny subsumes it (no bare Bash, no
    # deny token that matches heimdall-context).
    assert "Bash(heimdall-context" in " ".join(allowed)
    assert "Bash" not in disallowed
    # Raw Bash is not on the allow list, so default-deny rejects it.
    assert not any(
        t == "Bash" or (t.startswith("Bash(") and "heimdall-context" not in t) for t in allowed
    )


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
# Lenses see comments as untrusted context via the heimdall-context wrapper.
# The payload is NOT embedded in the prompt (unlike synthesis); each lens reads
# it through the same allowlisted `heimdall-context comments` call in-sandbox.
# ---------------------------------------------------------------------------

_ALL_LENSES = (SECURITY_LENS, DESIGN_LENS, CLEANLINESS_LENS)


@pytest.mark.parametrize("lens", _ALL_LENSES, ids=lambda lens: lens.name)
def test_lens_default_prompt_directs_reading_full_discussion_as_untrusted(
    lens: LensSpec,
) -> None:
    """The default lens prompt directs all four discussion reads as untrusted context."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=lens,
    )
    prompt = argv[argv.index("-p") + 1]
    # The lens is pointed at the full PR discussion via the four wrapper subcommands...
    assert "heimdall-context comments /workspace" in prompt
    assert "heimdall-context review-threads /workspace" in prompt
    assert "heimdall-context review-summaries /workspace" in prompt
    assert "heimdall-context own-prior /workspace" in prompt
    # ...and they are framed as untrusted background context, not instructions.
    assert "UNTRUSTED" in prompt
    assert "never" in prompt and "instructions" in prompt


@pytest.mark.parametrize("lens", _ALL_LENSES, ids=lambda lens: lens.name)
def test_lens_default_prompt_references_all_four_discussion_subcommands(
    lens: LensSpec,
) -> None:
    """The default prompt names every discussion subcommand against /workspace.

    Each of the four reads (comments, review-threads, review-summaries, own-prior)
    must be directed explicitly so the lens consults the full PR discussion, all
    framed as untrusted background context rather than instructions.
    """
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=lens,
    )
    prompt = argv[argv.index("-p") + 1]
    for sub in ("comments", "review-threads", "review-summaries", "own-prior"):
        assert f"heimdall-context {sub} /workspace" in prompt
    # Untrusted-frame tokens are present so directives in any read stay data.
    assert "UNTRUSTED" in prompt
    assert "never" in prompt and "instructions" in prompt


@pytest.mark.parametrize("lens", _ALL_LENSES, ids=lambda lens: lens.name)
def test_lens_system_prompt_lists_comments_subcommand_as_untrusted(
    lens: LensSpec,
) -> None:
    """Every lens system prompt exposes the comments subcommand, flagged untrusted.

    Comments must be framed consistently with the existing diff/file/docs context,
    so the wrapper subcommand list includes `comments` and the prompt marks it
    UNTRUSTED third-party data.
    """
    system_prompt = lens.system_prompt
    assert "diff|pr|file|docs|comments" in system_prompt
    assert "UNTRUSTED" in system_prompt


def test_lens_comments_payload_not_embedded_in_argv() -> None:
    """Lenses read comments via the wrapper, so no payload is baked into the argv.

    Unlike synthesis (which is tool-less and must embed the payload), a lens has the
    allowlisted wrapper and reads `heimdall-context comments` in-sandbox; baking the
    payload into the prompt would defeat the central mechanism and is not done.
    """
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
    )
    prompt = argv[argv.index("-p") + 1]
    # The prompt points AT the wrapper but carries no serialized comments JSON.
    assert '"comments":' not in prompt


def test_lens_comments_visibility_grants_no_new_tool() -> None:
    """Surfacing comments to the lenses leaves the tool allowlist unchanged.

    The sandbox posture is fixed: read-only Read/Grep/Glob plus the single
    `heimdall-context` Bash wrapper — comments ride that existing wrapper, so no
    new tool appears on the allow list.
    """
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
    )
    allowed = argv[argv.index("--allowedTools") + 1]
    assert allowed == "Read Grep Glob Bash(heimdall-context *)"


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
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
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
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
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


# ---------------------------------------------------------------------------
# Subprocess hardening: env allowlist + workspace cwd
# ---------------------------------------------------------------------------


def test_build_subprocess_env_strips_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claude child env keeps the base allowlist + passthrough, drops secrets."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/heimdall")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "-----BEGIN RSA-----")
    monkeypatch.setenv("WEBHOOK_SECRET", "hush")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")

    env = _build_subprocess_env(["HTTPS_PROXY"])

    assert env == {
        "PATH": "/usr/bin",
        "HOME": "/home/heimdall",
        "ANTHROPIC_API_KEY": "sk-test",
        "HTTPS_PROXY": "http://proxy:8080",
    }
    assert "GITHUB_APP_PRIVATE_KEY" not in env
    assert "WEBHOOK_SECRET" not in env


def test_build_subprocess_env_omits_absent_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowlisted keys missing from the parent env are simply not forwarded."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _build_subprocess_env(["DOES_NOT_EXIST"])
    assert "ANTHROPIC_API_KEY" not in env
    assert "DOES_NOT_EXIST" not in env
    assert env["PATH"] == "/usr/bin"


@pytest.mark.asyncio
async def test_run_lens_runs_claude_in_workspace_cwd() -> None:
    """run_lens scopes the subprocess cwd to the workspace and forwards passthrough."""
    captured: dict[str, Any] = {}

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **kwargs: object
    ) -> ClaudeResult:
        captured.update(kwargs)
        return _claude_result([])

    await run_lens(
        lens=SECURITY_LENS,
        workspace_dir=_WORKSPACE,
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        env_passthrough=["HTTPS_PROXY"],
        invoker=fake_invoker,
    )
    assert captured["cwd"] == _WORKSPACE
    assert captured["env_passthrough"] == ["HTTPS_PROXY"]


@pytest.mark.asyncio
async def test_run_lens_propagates_timeout() -> None:
    """A timeout in the invoker surfaces as LensTimeoutError (subprocess killed)."""

    async def timing_out_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
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
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
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


# ---------------------------------------------------------------------------
# Loud-fail: a run that never actually reviewed (0 tokens / no parseable JSON)
# must raise rather than masquerade as a clean "no findings" review.
# ---------------------------------------------------------------------------


def _raw_result(stdout: str, *, tokens: int) -> ClaudeResult:
    """A ClaudeResult carrying arbitrary (possibly non-JSON) stdout."""
    return ClaudeResult(stdout=stdout, total_tokens=tokens, killed=False)


@pytest.mark.asyncio
async def test_run_lens_raises_on_zero_tokens() -> None:
    """A 0-token run (e.g. a 401 before any API call) is a failed run, not clean."""

    async def zero_token_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _raw_result("", tokens=0)

    with pytest.raises(LensOutputError):
        await run_lens(
            lens=SECURITY_LENS,
            workspace_dir=_WORKSPACE,
            invoker=zero_token_invoker,
        )


@pytest.mark.asyncio
async def test_run_lens_raises_on_no_findings_json() -> None:
    """Tokens burned but no parseable findings JSON (garbage/error text) -> raise."""

    async def garbage_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _raw_result("401 Invalid authentication credentials", tokens=50)

    with pytest.raises(LensOutputError):
        await run_lens(
            lens=SECURITY_LENS,
            workspace_dir=_WORKSPACE,
            invoker=garbage_invoker,
        )


@pytest.mark.asyncio
async def test_run_lens_clean_pr_returns_empty_findings() -> None:
    """A genuine clean run (tokens > 0, parseable empty findings) still succeeds."""

    async def clean_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _claude_result([], tokens=120)

    result = await run_lens(
        lens=SECURITY_LENS,
        workspace_dir=_WORKSPACE,
        invoker=clean_invoker,
    )
    assert isinstance(result, LensResult)
    assert result.findings == []


@pytest.mark.asyncio
async def test_run_synthesis_raises_on_zero_tokens() -> None:
    """Synthesis that reports 0 tokens never ran -> raise (it builds the posted body)."""

    async def zero_token_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _raw_result("", tokens=0)

    with pytest.raises(LensOutputError):
        await run_synthesis(
            lens_results=[_lens_result_for_synthesis()],
            invoker=zero_token_invoker,
        )


@pytest.mark.asyncio
async def test_run_synthesis_raises_on_no_findings_json() -> None:
    """Synthesis output with no parseable findings JSON -> raise, not clean review."""

    async def garbage_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _raw_result("error: could not complete", tokens=80)

    with pytest.raises(LensOutputError):
        await run_synthesis(
            lens_results=[_lens_result_for_synthesis()],
            invoker=garbage_invoker,
        )


@pytest.mark.asyncio
async def test_run_synthesis_clean_returns_empty_comment() -> None:
    """Genuine clean synthesis (tokens > 0, empty findings) -> COMMENT, no survivors."""

    async def clean_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return _claude_result([], tokens=200)

    result = await run_synthesis(
        lens_results=[_lens_result_for_synthesis()],
        invoker=clean_invoker,
    )
    assert isinstance(result, SynthesisResult)
    assert result.tagged_findings == []
    assert result.verdict == "COMMENT"


def _lens_result_for_synthesis() -> LensResult:
    """A minimal LensResult to feed run_synthesis in the loud-fail tests."""
    return LensResult(
        lens_name="security",
        findings=[Finding(Severity.HIGH, "SecretLeak", "m", None)],
    )


# ---------------------------------------------------------------------------
# Dropped-lens warning banner (partial-failure surfacing)
# ---------------------------------------------------------------------------


def test_render_dropped_lenses_warning_empty_when_none() -> None:
    assert render_dropped_lenses_warning([]) == ""


def test_render_dropped_lenses_warning_names_single_lens() -> None:
    banner = render_dropped_lenses_warning(["security"])
    assert "security" in banner
    assert "1 review lens" in banner
    assert "was skipped" in banner


def test_render_dropped_lenses_warning_names_multiple_lenses() -> None:
    banner = render_dropped_lenses_warning(["security", "design"])
    assert "security" in banner
    assert "design" in banner
    assert "2 review lenses" in banner
    assert "were skipped" in banner


def test_render_comments_truncated_note_empty_when_not_truncated() -> None:
    assert render_comments_truncated_note(False) == ""


def test_render_comments_truncated_note_present_when_truncated() -> None:
    note = render_comments_truncated_note(True)
    assert note != ""
    assert "omitted" in note


# ---------------------------------------------------------------------------
# Suppressed-findings section (synthesis-judgment surfacing, #66)
# ---------------------------------------------------------------------------


def test_render_suppressed_findings_section_empty_when_none() -> None:
    assert render_suppressed_findings_section(()) == ""


def test_render_suppressed_findings_section_lists_title_and_reason() -> None:
    section = render_suppressed_findings_section(
        (
            SuppressedFinding(title="SQL injection", reason="resolved thread"),
            SuppressedFinding(
                title="XSS", reason="maintainer marked intentional"
            ),
        )
    )
    assert section != ""
    # Each suppressed finding's title and reason is surfaced.
    assert "SQL injection" in section
    assert "resolved thread" in section
    assert "XSS" in section
    assert "maintainer marked intentional" in section
    # Clearly labeled so a maintainer sees Heimdall made a judgment.
    assert "suppressed" in section.lower()
