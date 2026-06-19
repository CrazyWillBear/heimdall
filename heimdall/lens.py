"""Reusable single-lens runner: drive ``claude -p`` over a PR seed workspace.

A *lens* is one read-only Claude review pass over the materialized seed context
assembled by :mod:`heimdall.context`.  This module exposes the pieces #5 reuses
to add the Design-fit and Cleanliness lenses plus a synthesis pass:

    run_lens(lens, workspace_dir, ...)  -> LensResult   (the reusable seam)
    verdict_for(findings)               -> "REQUEST_CHANGES" | "COMMENT"

The Claude subprocess is invoked through an injected ``ClaudeInvoker`` so tests
feed canned output and never spawn the real binary.  The default invoker,
:func:`run_claude_subprocess`, enforces a per-agent cumulative-token cap and a
wall-clock timeout — exceeding either kills the subprocess and raises.

Security posture of the invocation (see :func:`build_claude_argv`):
  * model opus at max effort, headless (``-p``), JSON output;
  * the subprocess is spawned via ``create_subprocess_exec`` (no shell, so no
    shell-injection surface);
  * allowed tools are the read-only Read/Grep/Glob plus the single allowlisted
    ``heimdall-context`` Bash wrapper — never raw Bash, Write, or Edit;
  * Bash/Write/Edit are also explicitly disallowed as defense in depth.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Per-agent cumulative-token cap (issue requirement: 400k).
DEFAULT_TOKEN_CAP = 400_000
# Generous wall-clock timeout; a lens that runs longer is killed.
DEFAULT_TIMEOUT_SECONDS = 1_800.0


class Severity(Enum):
    """Severity of a single finding, ordered low-to-high by review impact."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Severities that escalate the PR verdict to REQUEST_CHANGES.
_BLOCKING_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})


@dataclass(frozen=True)
class Finding:
    """A single issue reported by a lens.

    Attributes:
        severity: One of :class:`Severity`; drives verdict and synthesis.
        title: Short headline for the finding.
        message: Full explanation of the problem and (ideally) a fix.
        location: ``path:line`` of the issue, or None when not localised.
    """

    severity: Severity
    title: str
    message: str
    location: str | None


@dataclass(frozen=True)
class LensResult:
    """The outcome of one lens pass over the seed context.

    Attributes:
        lens_name: Identifier of the lens that produced these findings.
        findings: All findings reported by the lens (possibly empty).
    """

    lens_name: str
    findings: list[Finding]


@dataclass(frozen=True)
class LensSpec:
    """Static definition of a lens: its name and review instructions.

    #5 defines two more of these (Design-fit, Cleanliness) and passes them to
    the same :func:`run_lens`.

    Attributes:
        name: Stable identifier (also used in the posted review body).
        system_prompt: Appended to Claude's system prompt to focus the pass.
    """

    name: str
    system_prompt: str


@dataclass
class ClaudeResult:
    """Raw result of one claude invocation.

    Attributes:
        stdout: The lens's textual output (claude's ``result`` field when the
            real invoker decodes the JSON envelope, or a bare findings JSON in
            tests).
        total_tokens: Cumulative tokens reported by claude for the run.
        killed: True if the subprocess was killed (timeout or cap breach).
    """

    stdout: str
    total_tokens: int
    killed: bool = False


class LensError(Exception):
    """Base class for lens-run failures."""


class LensTimeoutError(LensError):
    """Raised when a lens run exceeds its wall-clock timeout and is killed."""


class LensTokenCapError(LensError):
    """Raised when a lens run exceeds the cumulative-token cap and is killed."""


# An invoker takes the fully-built argv plus the caps and returns a ClaudeResult.
# Injected in tests; defaults to run_claude_subprocess in production.
ClaudeInvoker = Callable[..., Awaitable[ClaudeResult]]


_SECURITY_SYSTEM_PROMPT = (
    "You are Heimdall's Security review lens. Review ONLY the security posture of "
    "this pull request using the materialized seed context in the workspace. Use the "
    "heimdall-context wrapper (diff|pr|file|conventions) and the read-only Read/Grep/"
    "Glob tools to inspect changes. Do not modify anything. Report findings as a single "
    'JSON object on its own line: {"findings": [{"severity": "critical|high|medium|low", '
    '"title": "...", "message": "...", "location": "path:line"}]}. '
    "Emit an empty findings list when the PR introduces no security concern."
)

SECURITY_LENS = LensSpec(name="security", system_prompt=_SECURITY_SYSTEM_PROMPT)

_DEFAULT_PROMPT = (
    "Review this pull request through your assigned lens and report findings as the "
    "specified JSON object."
)

# Read-only tools plus the single allowlisted Bash wrapper. A bare "Bash" is
# never allowed; Bash is scoped to the heimdall-context command only.
_ALLOWED_TOOLS = "Read Grep Glob Bash(heimdall-context *)"
# Defense in depth: explicitly deny raw Bash and any mutation.
_DISALLOWED_TOOLS = "Bash Write Edit"


def build_claude_argv(
    *,
    claude_binary: str,
    workspace_dir: str,
    lens: LensSpec,
    prompt: str = _DEFAULT_PROMPT,
) -> list[str]:
    """Build the ``claude -p`` argv for a read-only lens run over a workspace.

    The invocation pins opus at max effort with JSON output, restricts tools to
    read-only Read/Grep/Glob plus the allowlisted ``heimdall-context`` Bash
    wrapper, and explicitly disallows raw Bash/Write/Edit.  argv is consumed by
    ``create_subprocess_exec`` (no shell), so none of these strings are
    shell-interpreted.

    Args:
        claude_binary: Path or name of the claude executable.
        workspace_dir: Materialized seed-context directory to scope the session to.
        lens: The lens whose system prompt focuses the review.
        prompt: The user prompt; defaults to a generic "review and report" prompt.

    Returns:
        The argument vector to pass to the subprocess invoker.
    """
    return [
        claude_binary,
        "-p",
        prompt,
        "--model",
        "opus",
        "--effort",
        "max",
        "--output-format",
        "json",
        "--add-dir",
        workspace_dir,
        "--allowedTools",
        _ALLOWED_TOOLS,
        "--disallowedTools",
        _DISALLOWED_TOOLS,
        "--append-system-prompt",
        lens.system_prompt,
    ]


def _coerce_severity(value: object) -> Severity:
    """Map a raw severity string to a Severity, defaulting unknowns to LOW."""
    if isinstance(value, str):
        try:
            return Severity(value.strip().lower())
        except ValueError:
            logger.warning("Unknown severity %r from lens; defaulting to LOW", value)
    return Severity.LOW


def _extract_findings_json(text: str) -> dict[str, object] | None:
    """Return the first ``{...}`` object in ``text`` that parses to a findings dict.

    Claude may wrap the findings JSON in prose, so we scan for opening braces and
    try to decode each candidate.  Returns None when no findings object exists.
    """
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        brace = text.find("{", index)
        if brace == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(obj, dict) and "findings" in obj:
            return obj
        index = brace + 1
    return None


def parse_findings(text: str) -> list[Finding]:
    """Parse a lens's textual output into a list of :class:`Finding`.

    Tolerates prose around the JSON block.  Each finding's severity is coerced
    via :func:`_coerce_severity`; missing title/message degrade to empty strings.

    Args:
        text: The lens output (claude's ``result`` text or a bare JSON block).

    Returns:
        The parsed findings, empty when none are present.
    """
    obj = _extract_findings_json(text)
    if obj is None:
        logger.warning("No findings JSON in lens output; treating as no findings")
        return []
    raw_findings = obj.get("findings", [])
    if not isinstance(raw_findings, list):
        return []

    findings: list[Finding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        location = item.get("location")
        findings.append(
            Finding(
                severity=_coerce_severity(item.get("severity")),
                title=str(item.get("title", "")),
                message=str(item.get("message", "")),
                location=str(location) if location is not None else None,
            )
        )
    return findings


_NO_FINDINGS_BODY = "Heimdall security review: no security concerns found."


def format_review_body(findings: list[Finding]) -> str:
    """Render findings into a Markdown PR-review body.

    Findings are listed worst-first (critical -> low) so the most important
    issues lead.  An empty list yields a short all-clear message.

    Args:
        findings: The findings to render.

    Returns:
        A Markdown string suitable as a GitHub review body.
    """
    if not findings:
        return _NO_FINDINGS_BODY

    order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    ranked = sorted(findings, key=lambda f: order[f.severity])

    lines = ["## Heimdall security review", ""]
    for finding in ranked:
        location = f" (`{finding.location}`)" if finding.location else ""
        lines.append(f"- **[{finding.severity.value.upper()}] {finding.title}**{location}")
        if finding.message:
            lines.append(f"  {finding.message}")
    return "\n".join(lines)


def verdict_for(findings: list[Finding]) -> str:
    """Map findings to a PR review event.

    Any high or critical finding requests changes; otherwise the review is a
    plain comment.

    Args:
        findings: All findings across the run.

    Returns:
        "REQUEST_CHANGES" if any blocking finding exists, else "COMMENT".
    """
    if any(f.severity in _BLOCKING_SEVERITIES for f in findings):
        return "REQUEST_CHANGES"
    return "COMMENT"


def _sum_tokens(envelope: dict[str, object]) -> int:
    """Sum input+output tokens from a claude JSON-output usage block."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = 0
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total += value
    return total


def _parse_envelope(stdout_bytes: bytes) -> dict[str, object]:
    """Decode claude's ``--output-format json`` envelope; tolerate non-JSON stdout."""
    text = stdout_bytes.decode("utf-8", errors="replace")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to treating raw stdout as the result text with unknown usage.
        return {"result": text}
    if isinstance(obj, dict):
        return obj
    return {"result": text}


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and reap it, ignoring an already-exited process."""
    try:
        proc.kill()
    except ProcessLookupError:
        return
    with contextlib.suppress(Exception):
        await proc.wait()


async def run_claude_subprocess(
    argv: list[str],
    *,
    timeout_seconds: float,
    token_cap: int,
) -> ClaudeResult:
    """Default invoker: spawn claude, enforce the timeout and token cap, parse output.

    The subprocess is spawned with ``create_subprocess_exec`` (no shell) and is
    killed (and the failure raised) when the wall-clock timeout elapses or when
    claude's reported cumulative usage exceeds the cap.

    Args:
        argv: The argument vector from :func:`build_claude_argv`.
        timeout_seconds: Wall-clock limit; the process is killed past it.
        token_cap: Cumulative-token ceiling; a run reporting more is rejected.

    Returns:
        A :class:`ClaudeResult` with stdout (claude's ``result`` text) and tokens.

    Raises:
        LensTimeoutError: The run exceeded ``timeout_seconds`` (subprocess killed).
        LensTokenCapError: The run exceeded ``token_cap`` (subprocess killed).
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except TimeoutError as exc:
        await _kill(proc)
        raise LensTimeoutError(
            f"Lens run exceeded {timeout_seconds}s wall-clock; subprocess killed"
        ) from exc

    envelope = _parse_envelope(stdout_bytes)
    total_tokens = _sum_tokens(envelope)
    if total_tokens > token_cap:
        await _kill(proc)
        raise LensTokenCapError(
            f"Lens run used {total_tokens} tokens, exceeding cap {token_cap}; "
            "subprocess killed"
        )

    result_text = envelope.get("result")
    stdout_text = (
        result_text
        if isinstance(result_text, str)
        else stdout_bytes.decode("utf-8", errors="replace")
    )
    return ClaudeResult(stdout=stdout_text, total_tokens=total_tokens, killed=False)


async def run_lens(
    *,
    lens: LensSpec,
    workspace_dir: str,
    claude_binary: str = "claude",
    token_cap: int = DEFAULT_TOKEN_CAP,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    invoker: ClaudeInvoker = run_claude_subprocess,
) -> LensResult:
    """Run one lens over a materialized seed workspace and parse its findings.

    This is the reusable seam #5 calls once per lens.  It builds the read-only
    ``claude -p`` argv, delegates execution (and cap/timeout enforcement) to the
    injected ``invoker``, then parses structured findings.

    Args:
        lens: The lens to run (name + system prompt).
        workspace_dir: Materialized seed-context directory from assemble_pr_context.
        claude_binary: Path or name of the claude executable.
        token_cap: Per-agent cumulative-token ceiling.
        timeout_seconds: Wall-clock limit for the run.
        invoker: Coroutine that runs the subprocess; injected in tests.

    Returns:
        A :class:`LensResult` with the lens name and parsed findings.

    Raises:
        LensTimeoutError / LensTokenCapError: Propagated from the invoker when
            the run is aborted; callers handle these as a failed lens.
    """
    argv = build_claude_argv(
        claude_binary=claude_binary,
        workspace_dir=workspace_dir,
        lens=lens,
    )
    logger.info("Running lens %s over %s", lens.name, workspace_dir)
    result = await invoker(argv, timeout_seconds=timeout_seconds, token_cap=token_cap)
    findings = parse_findings(result.stdout)
    logger.info(
        "Lens %s produced %d findings (%d tokens)",
        lens.name,
        len(findings),
        result.total_tokens,
    )
    return LensResult(lens_name=lens.name, findings=findings)
