"""Tests for the three-lens fan-out helpers and the synthesis pass.

The claude subprocess is always mocked via an injected ClaudeInvoker, so no real
claude binary is spawned.  These tests pin issue #5's contract: per-lens model and
effort wiring, the Design-fit and Cleanliness lens specs, and the synthesis pass
that dedups overlapping findings, ranks by severity, writes the verdict, and renders
a severity-grouped, lens-tagged review body.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from heimdall.lens import (
    CLEANLINESS_LENS,
    DESIGN_LENS,
    SECURITY_LENS,
    SYNTHESIS_LENS,
    ClaudeResult,
    Finding,
    LensResult,
    Severity,
    SynthesisResult,
    TaggedFinding,
    build_claude_argv,
    build_synthesis_argv,
    format_synthesis_body,
    run_synthesis,
    verdict_for_tagged,
)

_WORKSPACE = "/tmp/heimdall-ws"


# ---------------------------------------------------------------------------
# Per-lens model + effort wiring
# ---------------------------------------------------------------------------


def test_security_lens_pins_opus_max() -> None:
    """The Security lens keeps the opus/max model+effort contract."""
    assert SECURITY_LENS.model == "opus"
    assert SECURITY_LENS.effort == "max"


def test_design_and_cleanliness_lenses_use_sonnet_high() -> None:
    """The two new lenses run on sonnet at high effort."""
    for lens in (DESIGN_LENS, CLEANLINESS_LENS):
        assert lens.model == "sonnet"
        assert lens.effort == "high"


def test_argv_threads_lens_model_and_effort() -> None:
    """build_claude_argv reads model+effort from the lens, not a hardcoded pair."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=DESIGN_LENS,
        prompt="review",
    )
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "high"


def test_argv_security_lens_still_opus_max() -> None:
    """The Security lens argv still pins opus/max after the model+effort refactor."""
    argv = build_claude_argv(
        claude_binary="claude",
        workspace_dir=_WORKSPACE,
        lens=SECURITY_LENS,
        prompt="review",
    )
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "max"


# ---------------------------------------------------------------------------
# Synthesis argv: no workspace, no tools (it only reasons over the prompt JSON)
# ---------------------------------------------------------------------------


def test_synthesis_argv_pins_synthesis_lens_model_effort_and_prompt() -> None:
    """The synthesis argv runs headless on the synthesis lens with JSON output."""
    argv = build_synthesis_argv(claude_binary="claude", prompt="synthesize")
    assert argv[0] == "claude"
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "synthesize"
    assert argv[argv.index("--model") + 1] == SYNTHESIS_LENS.model
    assert argv[argv.index("--effort") + 1] == SYNTHESIS_LENS.effort
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--append-system-prompt") + 1] == SYNTHESIS_LENS.system_prompt


def test_synthesis_argv_has_no_workspace() -> None:
    """Synthesis is handed the findings JSON; it gets no --add-dir workspace."""
    argv = build_synthesis_argv(claude_binary="claude", prompt="synthesize")
    assert "--add-dir" not in argv


def test_synthesis_argv_grants_no_tools() -> None:
    """Synthesis gets no Read/Grep/Glob/Bash tools — it cannot explore the seed."""
    argv = build_synthesis_argv(claude_binary="claude", prompt="synthesize")
    allowed = argv[argv.index("--allowedTools") + 1] if "--allowedTools" in argv else ""
    for tool in ("Read", "Grep", "Glob", "Bash"):
        assert tool not in allowed


# ---------------------------------------------------------------------------
# Tagged findings + verdict over the surviving set
# ---------------------------------------------------------------------------


def test_verdict_for_tagged_reflects_highest_severity() -> None:
    """The verdict over tagged findings escalates on a high/critical survivor."""
    tagged = [
        TaggedFinding(lens="cleanliness", finding=_finding(Severity.LOW)),
        TaggedFinding(lens="security", finding=_finding(Severity.HIGH)),
    ]
    assert verdict_for_tagged(tagged) == "REQUEST_CHANGES"


def test_verdict_for_tagged_comment_without_blocking() -> None:
    """Only medium/low survivors produce a COMMENT verdict."""
    tagged = [
        TaggedFinding(lens="design", finding=_finding(Severity.MEDIUM)),
        TaggedFinding(lens="cleanliness", finding=_finding(Severity.LOW)),
    ]
    assert verdict_for_tagged(tagged) == "COMMENT"


# ---------------------------------------------------------------------------
# Severity-grouped, lens-tagged review body
# ---------------------------------------------------------------------------


def test_synthesis_body_groups_by_severity_and_tags_lens() -> None:
    """The body groups findings under severity headers, each tagged by its lens."""
    tagged = [
        TaggedFinding(
            lens="cleanliness",
            finding=Finding(
                severity=Severity.LOW, title="Naming nit", message="rename x", location="a.py:1"
            ),
        ),
        TaggedFinding(
            lens="security",
            finding=Finding(
                severity=Severity.CRITICAL, title="RCE", message="eval", location="b.py:9"
            ),
        ),
    ]
    body = format_synthesis_body(tagged)
    # Critical group precedes the low group (worst-first).
    assert body.index("CRITICAL") < body.index("LOW")
    # Each finding carries its originating lens tag.
    assert "security" in body.lower()
    assert "cleanliness" in body.lower()
    # The body still surfaces the finding details.
    assert "RCE" in body
    assert "Naming nit" in body


def test_synthesis_body_all_clear_when_empty() -> None:
    """An empty surviving set yields a short all-clear message."""
    assert "no" in format_synthesis_body([]).lower()


# ---------------------------------------------------------------------------
# run_synthesis: combines lens findings, dedups, ranks, tags
# ---------------------------------------------------------------------------


def _finding(severity: Severity, title: str = "t", message: str = "m") -> Finding:
    return Finding(severity=severity, title=title, message=message, location=None)


def _lens_result(name: str, findings: list[Finding]) -> LensResult:
    return LensResult(lens_name=name, findings=findings)


@pytest.mark.asyncio
async def test_run_synthesis_passes_all_lens_findings_to_claude() -> None:
    """Every lens's findings reach the synthesis prompt (all three lenses)."""
    captured: dict[str, Any] = {}

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        captured["prompt"] = argv[argv.index("-p") + 1]
        # Echo a single deduped survivor.
        survivor = {"severity": "high", "title": "dup", "message": "m", "lens": "security"}
        return ClaudeResult(
            stdout=json.dumps({"findings": [survivor]}),
            total_tokens=10,
        )

    lens_results = [
        _lens_result("security", [_finding(Severity.HIGH, "SecretLeak")]),
        _lens_result("design", [_finding(Severity.MEDIUM, "GodObject")]),
        _lens_result("cleanliness", [_finding(Severity.LOW, "DeadCode")]),
    ]

    await run_synthesis(
        lens_results=lens_results,
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    prompt = captured["prompt"]
    # Findings from all three lenses must be present in the synthesis input.
    assert "SecretLeak" in prompt
    assert "GodObject" in prompt
    assert "DeadCode" in prompt
    # Each lens name is present so synthesis can tag survivors by origin.
    assert "security" in prompt
    assert "design" in prompt
    assert "cleanliness" in prompt


@pytest.mark.asyncio
async def test_run_synthesis_returns_deduped_ranked_tagged_findings() -> None:
    """Synthesis output parses into ranked, lens-tagged surviving findings + verdict."""
    synthesized = {
        "findings": [
            {"severity": "low", "title": "Nit", "message": "x", "lens": "cleanliness"},
            {"severity": "critical", "title": "RCE", "message": "eval", "lens": "security"},
        ]
    }

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return ClaudeResult(stdout=json.dumps(synthesized), total_tokens=10)

    result = await run_synthesis(
        lens_results=[
            _lens_result("security", [_finding(Severity.CRITICAL, "RCE")]),
            _lens_result("cleanliness", [_finding(Severity.LOW, "Nit")]),
        ],
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    assert isinstance(result, SynthesisResult)
    # Ranked worst-first.
    assert [t.finding.severity for t in result.tagged_findings] == [
        Severity.CRITICAL,
        Severity.LOW,
    ]
    # Each survivor is tagged with its originating lens.
    assert result.tagged_findings[0].lens == "security"
    assert result.tagged_findings[1].lens == "cleanliness"
    # Verdict reflects the highest surviving severity.
    assert result.verdict == "REQUEST_CHANGES"
    # The body is severity-grouped and lens-tagged.
    assert "CRITICAL" in result.body
    assert "security" in result.body.lower()


@pytest.mark.asyncio
async def test_run_synthesis_lens_tags_survive_malformed_entry() -> None:
    """A non-dict entry in findings must not misalign surviving findings' lens tags.

    The synthesizer can emit a malformed (non-dict) entry anywhere in the array.
    Each surviving finding must still carry the lens named on its own raw object,
    not a tag shifted in from a neighbour (the pre-fix positional-zip bug).
    """
    synthesized = {
        "findings": [
            "garbage-non-dict-entry",
            {"severity": "high", "title": "RealBug", "message": "m", "lens": "security"},
            {"severity": "low", "title": "Nit", "message": "x", "lens": "cleanliness"},
        ]
    }

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return ClaudeResult(stdout=json.dumps(synthesized), total_tokens=10)

    result = await run_synthesis(
        lens_results=[_lens_result("security", [_finding(Severity.HIGH, "RealBug")])],
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    lens_by_title = {t.finding.title: t.lens for t in result.tagged_findings}
    assert lens_by_title == {"RealBug": "security", "Nit": "cleanliness"}


@pytest.mark.asyncio
async def test_run_synthesis_verdict_reflects_dedup_survivors_only() -> None:
    """When synthesis drops the only high finding, the verdict downgrades to COMMENT."""
    # Two lenses each report a HIGH duplicate; synthesis collapses them into one
    # surviving LOW after deciding the real impact is minor.
    synthesized = {
        "findings": [
            {"severity": "low", "title": "Merged", "message": "m", "lens": "design"},
        ]
    }

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        return ClaudeResult(stdout=json.dumps(synthesized), total_tokens=10)

    result = await run_synthesis(
        lens_results=[
            _lens_result("security", [_finding(Severity.HIGH, "Dup")]),
            _lens_result("design", [_finding(Severity.HIGH, "Dup")]),
        ],
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    assert result.verdict == "COMMENT"
    assert len(result.tagged_findings) == 1


@pytest.mark.asyncio
async def test_run_synthesis_uses_synthesis_lens_spec() -> None:
    """run_synthesis invokes claude with the synthesis lens (its system prompt)."""
    captured: dict[str, Any] = {}

    async def fake_invoker(
        argv: list[str], *, timeout_seconds: float, token_cap: int, **_kwargs: object
    ) -> ClaudeResult:
        captured["argv"] = argv
        return ClaudeResult(stdout=json.dumps({"findings": []}), total_tokens=0)

    await run_synthesis(
        lens_results=[_lens_result("security", [])],
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    argv = captured["argv"]
    system_prompt = argv[argv.index("--append-system-prompt") + 1]
    assert system_prompt == SYNTHESIS_LENS.system_prompt


@pytest.mark.asyncio
async def test_run_synthesis_does_not_scope_to_a_workspace() -> None:
    """The synthesis call gets no workspace: no --add-dir and no seed cwd."""
    captured: dict[str, Any] = {}

    async def fake_invoker(
        argv: list[str],
        *,
        timeout_seconds: float,
        token_cap: int,
        cwd: str | None = None,
        **_kwargs: object,
    ) -> ClaudeResult:
        captured["argv"] = argv
        captured["cwd"] = cwd
        return ClaudeResult(stdout=json.dumps({"findings": []}), total_tokens=0)

    await run_synthesis(
        lens_results=[_lens_result("security", [])],
        claude_binary="claude",
        token_cap=400_000,
        timeout_seconds=900,
        invoker=fake_invoker,
    )

    assert "--add-dir" not in captured["argv"]
    assert captured["cwd"] is None


def test_synthesis_lens_is_opus_max() -> None:
    """Synthesis runs on opus/max — it must reason over all lens output carefully."""
    assert SYNTHESIS_LENS.model == "opus"
    assert SYNTHESIS_LENS.effort == "max"
