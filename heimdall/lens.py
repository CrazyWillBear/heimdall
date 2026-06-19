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
  * Write/Edit are explicitly disallowed; raw Bash needs no deny rule because
    default-deny already blocks anything off the allowlist, and an unscoped Bash
    deny would take precedence over (and neuter) the wrapper's allow rule.

Filesystem-read confinement is **incomplete** and only partially mitigated here.
``--add-dir`` *adds* the workspace to the allowed set; it does **not** restrict
Read/Grep/Glob to it, so a prompt-injected PR can still read files elsewhere on the
worker by absolute path.  The interim mitigations are: the child env is reduced to a
strict allowlist (see :func:`run_claude_subprocess`), so the App private key and
webhook secret are not visible to the model; and the subprocess ``cwd`` is set to the
workspace, which keeps default-scope (no-path) Grep/Glob inside it.  Neither closes
absolute-path reads — the real boundary is an OS-level filesystem sandbox
(Landlock/bwrap/container restricting the read syscall to the workspace), which is NOT
yet implemented.  PR code is still never *executed* (Bash off the allowlist is denied).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
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

# Worst-first sort key shared by every review-body renderer.
_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


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
    """Static definition of a lens: its name, instructions, and Claude model knobs.

    The Security lens runs on opus/max; the Design-fit and Cleanliness lenses run
    on sonnet/high; the synthesis pass runs on opus/max.  Model and effort live on
    the spec so :func:`build_claude_argv` is the same code path for every lens.

    Attributes:
        name: Stable identifier (also used in the posted review body).
        system_prompt: Appended to Claude's system prompt to focus the pass.
        model: Claude model to run the pass on (e.g. "opus", "sonnet").
        effort: Reasoning-effort level for the pass (e.g. "max", "high").
    """

    name: str
    system_prompt: str
    model: str = "opus"
    effort: str = "max"


@dataclass(frozen=True)
class TaggedFinding:
    """A finding tagged with the lens that originated it.

    Synthesis survivors carry their source lens so the rendered review can show,
    per finding, which lens raised it.

    Attributes:
        lens: Name of the originating lens (e.g. "security", "design").
        finding: The underlying :class:`Finding`.
    """

    lens: str
    finding: Finding


@dataclass(frozen=True)
class SynthesisResult:
    """The outcome of the synthesis pass over all lenses' findings.

    Attributes:
        tagged_findings: Deduped, severity-ranked survivors, each lens-tagged.
        verdict: "REQUEST_CHANGES" or "COMMENT" over the surviving set.
        body: The rendered Markdown review body (severity-grouped, lens-tagged).
    """

    tagged_findings: list[TaggedFinding]
    verdict: str
    body: str


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

SECURITY_LENS = LensSpec(
    name="security",
    system_prompt=_SECURITY_SYSTEM_PROMPT,
    model="opus",
    effort="max",
)

_FINDINGS_JSON_CONTRACT = (
    'Report findings as a single JSON object on its own line: {"findings": '
    '[{"severity": "critical|high|medium|low", "title": "...", "message": "...", '
    '"location": "path:line"}]}. Emit an empty findings list when the PR is clean '
    "through your lens."
)

_DESIGN_SYSTEM_PROMPT = (
    "You are Heimdall's Design-fit / architecture review lens. Review ONLY whether "
    "this pull request fits the existing design and architecture: module boundaries, "
    "coupling and cohesion, abstraction level, layering, naming of public surfaces, "
    "and consistency with established patterns and conventions. Use the heimdall-context "
    "wrapper (diff|pr|file|conventions) and the read-only Read/Grep/Glob tools to inspect "
    "changes. Do not modify anything. " + _FINDINGS_JSON_CONTRACT
)

DESIGN_LENS = LensSpec(
    name="design",
    system_prompt=_DESIGN_SYSTEM_PROMPT,
    model="sonnet",
    effort="high",
)

_CLEANLINESS_SYSTEM_PROMPT = (
    "You are Heimdall's Cleanliness review lens. Review ONLY the cleanliness of this "
    "pull request: readability, dead or duplicated code, unclear names, missing or "
    "misleading docs, error-handling hygiene, and adherence to the repo style guide. "
    "Use the heimdall-context wrapper (diff|pr|file|conventions) and the read-only "
    "Read/Grep/Glob tools to inspect changes. Do not modify anything. "
    + _FINDINGS_JSON_CONTRACT
)

CLEANLINESS_LENS = LensSpec(
    name="cleanliness",
    system_prompt=_CLEANLINESS_SYSTEM_PROMPT,
    model="sonnet",
    effort="high",
)

# Synthesis runs on opus/max because it must reason over every lens's output,
# dedup overlaps, and decide the surviving severity that drives the verdict.
_SYNTHESIS_SYSTEM_PROMPT = (
    "You are Heimdall's review synthesizer. You receive the combined findings of "
    "three independent review lenses (security, design, cleanliness) as JSON in the "
    "prompt. Produce the final set of findings by: (1) DEDUPING overlapping findings "
    "that describe the same underlying issue across lenses into a single finding; "
    "(2) keeping the most accurate severity for each surviving finding; (3) attributing "
    "each survivor to the lens that originated it. Do not invent new findings beyond "
    "what the lenses reported. Report the surviving findings as a single JSON object on "
    'its own line: {"findings": [{"severity": "critical|high|medium|low", "title": '
    '"...", "message": "...", "location": "path:line", "lens": '
    '"security|design|cleanliness"}]}. Emit an empty findings list when nothing survives.'
)

SYNTHESIS_LENS = LensSpec(
    name="synthesis",
    system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
    model="opus",
    effort="max",
)

_DEFAULT_PROMPT = (
    "Review this pull request through your assigned lens and report findings as the "
    "specified JSON object."
)

# Read-only tools plus the single allowlisted Bash wrapper. A bare "Bash" is
# never allowed; Bash is scoped to the heimdall-context command only.
_ALLOWED_TOOLS = "Read Grep Glob Bash(heimdall-context *)"
# Deny mutating tools only. An unscoped "Bash" deny would take precedence over the
# Bash(heimdall-context *) allow rule (deny wins) and neuter the wrapper, so it is
# intentionally absent: under default-deny, raw Bash is already blocked by not
# being allowed.
_DISALLOWED_TOOLS = "Write Edit"


def _base_claude_argv(*, claude_binary: str, lens: LensSpec, prompt: str) -> list[str]:
    """Shared head of every ``claude -p`` argv: binary, prompt, model/effort, JSON.

    Holds the parts common to a read-only lens run and the no-tools synthesis pass —
    headless ``-p`` on the lens's own model+effort with JSON output and the lens's
    appended system prompt.  Callers append (or omit) workspace/tool flags.  argv is
    consumed by ``create_subprocess_exec`` (no shell), so none of these strings are
    shell-interpreted.
    """
    return [
        claude_binary,
        "-p",
        prompt,
        "--model",
        lens.model,
        "--effort",
        lens.effort,
        "--output-format",
        "json",
        "--append-system-prompt",
        lens.system_prompt,
    ]


def build_claude_argv(
    *,
    claude_binary: str,
    workspace_dir: str,
    lens: LensSpec,
    prompt: str = _DEFAULT_PROMPT,
) -> list[str]:
    """Build the ``claude -p`` argv for a read-only lens run over a workspace.

    The invocation pins the lens's own model and effort with JSON output, restricts
    tools to read-only Read/Grep/Glob plus the allowlisted ``heimdall-context`` Bash
    wrapper, and disallows Write/Edit.  Raw Bash carries no deny rule (an unscoped
    Bash deny would override the wrapper's allow rule); default-deny blocks it.
    argv is consumed by ``create_subprocess_exec`` (no shell), so none of these
    strings are shell-interpreted.

    Args:
        claude_binary: Path or name of the claude executable.
        workspace_dir: Materialized seed-context directory to scope the session to.
        lens: The lens whose system prompt, model, and effort focus the review.
        prompt: The user prompt; defaults to a generic "review and report" prompt.

    Returns:
        The argument vector to pass to the subprocess invoker.
    """
    return [
        *_base_claude_argv(claude_binary=claude_binary, lens=lens, prompt=prompt),
        "--add-dir",
        workspace_dir,
        "--allowedTools",
        _ALLOWED_TOOLS,
        "--disallowedTools",
        _DISALLOWED_TOOLS,
    ]


def build_synthesis_argv(*, claude_binary: str, prompt: str) -> list[str]:
    """Build the ``claude -p`` argv for the no-tools, no-workspace synthesis pass.

    Synthesis only dedups/ranks/tags the three lenses' findings JSON it is handed in
    the prompt, so it gets neither a workspace (no ``--add-dir``, and the caller does
    not set ``cwd``) nor any tools (no Read/Grep/Glob and no ``heimdall-context`` Bash
    wrapper).  Because it cannot read files at all, it has nothing to confine and is
    not sandboxed — which is what makes it correct to sandbox only the three lenses.
    It still runs headless on the synthesis lens's model+effort with JSON output.

    Args:
        claude_binary: Path or name of the claude executable.
        prompt: The synthesis user prompt carrying the per-lens findings JSON.

    Returns:
        The argument vector to pass to the subprocess invoker.
    """
    return _base_claude_argv(
        claude_binary=claude_binary, lens=SYNTHESIS_LENS, prompt=prompt
    )


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


def _finding_from_raw(item: dict[str, object]) -> Finding:
    """Coerce one raw findings dict into a :class:`Finding`.

    Shared by :func:`parse_findings` and :func:`parse_tagged_findings` so the
    severity/title/message/location coercion lives in exactly one place.  Severity
    is coerced via :func:`_coerce_severity`; missing title/message degrade to empty
    strings, and a missing location stays None.
    """
    location = item.get("location")
    return Finding(
        severity=_coerce_severity(item.get("severity")),
        title=str(item.get("title", "")),
        message=str(item.get("message", "")),
        location=str(location) if location is not None else None,
    )


def _raw_findings(text: str) -> list[object]:
    """Return the raw ``findings`` list from a lens's output, empty when absent.

    Tolerates prose around the JSON block and a missing or non-list ``findings``
    field.  Callers filter the list for dict entries themselves.
    """
    obj = _extract_findings_json(text)
    if obj is None:
        logger.warning("No findings JSON in lens output; treating as no findings")
        return []
    raw_findings = obj.get("findings", [])
    if not isinstance(raw_findings, list):
        return []
    return raw_findings


def parse_findings(text: str) -> list[Finding]:
    """Parse a lens's textual output into a list of :class:`Finding`.

    Tolerates prose around the JSON block.  Each finding's severity is coerced
    via :func:`_coerce_severity`; missing title/message degrade to empty strings.

    Args:
        text: The lens output (claude's ``result`` text or a bare JSON block).

    Returns:
        The parsed findings, empty when none are present.
    """
    return [_finding_from_raw(item) for item in _raw_findings(text) if isinstance(item, dict)]


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

    ranked = sorted(findings, key=lambda f: _SEVERITY_ORDER[f.severity])

    lines = ["## Heimdall security review", ""]
    for finding in ranked:
        location = f" (`{finding.location}`)" if finding.location else ""
        lines.append(f"- **[{finding.severity.value.upper()}] {finding.title}**{location}")
        if finding.message:
            lines.append(f"  {finding.message}")
    return "\n".join(lines)


def verdict_for(
    findings: list[Finding],
    *,
    blocking: frozenset[Severity] = _BLOCKING_SEVERITIES,
) -> str:
    """Map findings to a PR review event.

    Any finding whose severity is in ``blocking`` requests changes; otherwise the
    review is a plain comment.  The default blocking set (high/critical) is
    overridden by the repo config's severity threshold.

    Args:
        findings: All findings across the run.
        blocking: The severities that escalate to REQUEST_CHANGES; defaults to
            high/critical.

    Returns:
        "REQUEST_CHANGES" if any blocking finding exists, else "COMMENT".
    """
    if any(f.severity in blocking for f in findings):
        return "REQUEST_CHANGES"
    return "COMMENT"


def verdict_for_tagged(
    tagged: list[TaggedFinding],
    *,
    blocking: frozenset[Severity] = _BLOCKING_SEVERITIES,
) -> str:
    """Map the surviving tagged findings to a PR review event.

    Reuses :func:`verdict_for` over the underlying findings so the verdict reflects
    the highest-severity finding that survived synthesis dedup.

    Args:
        tagged: The synthesis survivors.
        blocking: The severities that escalate to REQUEST_CHANGES; defaults to
            high/critical.

    Returns:
        "REQUEST_CHANGES" if any survivor is blocking, else "COMMENT".
    """
    return verdict_for([t.finding for t in tagged], blocking=blocking)


_NO_SYNTHESIS_FINDINGS_BODY = "Heimdall review: no concerns found across any lens."

_SEVERITY_HEADERS = (
    (Severity.CRITICAL, "Critical"),
    (Severity.HIGH, "High"),
    (Severity.MEDIUM, "Medium"),
    (Severity.LOW, "Low"),
)


def format_synthesis_body(tagged: list[TaggedFinding]) -> str:
    """Render synthesis survivors grouped by severity, each tagged by its lens.

    Findings are grouped under worst-first severity headers (Critical -> Low); each
    bullet carries a ``[lens]`` tag naming the originating lens.  An empty list
    yields a short all-clear message.

    Args:
        tagged: The deduped, lens-tagged survivors from synthesis.

    Returns:
        A Markdown string suitable as a GitHub review body.
    """
    if not tagged:
        return _NO_SYNTHESIS_FINDINGS_BODY

    by_severity: dict[Severity, list[TaggedFinding]] = {}
    for item in tagged:
        by_severity.setdefault(item.finding.severity, []).append(item)

    lines = ["## Heimdall review", ""]
    for severity, header in _SEVERITY_HEADERS:
        group = by_severity.get(severity)
        if not group:
            continue
        lines.append(f"### {severity.value.upper()} — {header}")
        for item in group:
            finding = item.finding
            location = f" (`{finding.location}`)" if finding.location else ""
            lines.append(f"- **[{item.lens}] {finding.title}**{location}")
            if finding.message:
                lines.append(f"  {finding.message}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_tagged_findings(text: str) -> list[TaggedFinding]:
    """Parse synthesis output into lens-tagged findings, ranked worst-first.

    Derives each survivor's finding fields and its ``lens`` tag from the SAME raw
    dict in one filtered pass, so a malformed (non-dict) entry anywhere in the array
    cannot misalign a finding with a neighbour's lens tag.  Reuses
    :func:`_finding_from_raw` for the severity/title/message/location coercion.  The
    result is sorted worst-first so the verdict and body see a stable ranking.

    Args:
        text: The synthesis lens output (claude's ``result`` text or bare JSON).

    Returns:
        The deduped survivors, lens-tagged and severity-ranked. Empty when none.
    """
    tagged = [
        TaggedFinding(lens=_lens_tag(item), finding=_finding_from_raw(item))
        for item in _raw_findings(text)
        if isinstance(item, dict)
    ]
    return sorted(tagged, key=lambda t: _SEVERITY_ORDER[t.finding.severity])


def _lens_tag(item: dict[str, object]) -> str:
    """Read the ``lens`` tag off a raw findings dict, defaulting to ""."""
    lens_value = item.get("lens")
    return str(lens_value) if lens_value is not None else ""


def _render_lens_findings_json(lens_results: list[LensResult]) -> str:
    """Serialize every lens's findings into the JSON the synthesizer reads.

    Each finding carries its originating lens name so the synthesizer can dedup
    across lenses and attribute survivors.
    """
    payload = {
        "lenses": [
            {
                "lens": result.lens_name,
                "findings": [
                    {
                        "severity": f.severity.value,
                        "title": f.title,
                        "message": f.message,
                        "location": f.location,
                    }
                    for f in result.findings
                ],
            }
            for result in lens_results
        ]
    }
    return json.dumps(payload, indent=2)


def _build_synthesis_prompt(lens_results: list[LensResult]) -> str:
    """Build the user prompt feeding all lens findings into the synthesis pass."""
    return (
        "Synthesize the final review from these per-lens findings. Dedup overlaps "
        "across lenses, keep the most accurate severity, attribute each survivor to "
        "its lens, and emit the findings JSON described in your instructions.\n\n"
        + _render_lens_findings_json(lens_results)
    )


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


# Env vars the claude child always needs: PATH to find node/claude, HOME for its
# ~/.claude config, ANTHROPIC_API_KEY to authenticate.  Everything else (incl. the
# App private key and webhook secret) is stripped so a prompt-injected PR cannot read
# a secret out of the child's environment.
_BASE_ENV_KEYS = ("PATH", "HOME", "ANTHROPIC_API_KEY")


def _build_subprocess_env(passthrough: Sequence[str] = ()) -> dict[str, str]:
    """Build a minimal env for the claude child: a strict allowlist of the parent's.

    Only ``PATH``/``HOME``/``ANTHROPIC_API_KEY`` plus any caller-supplied
    ``passthrough`` keys (e.g. ``HTTPS_PROXY``/``NODE_EXTRA_CA_CERTS`` for proxied or
    custom-CA deployments) are forwarded; every other variable — notably
    ``GITHUB_APP_PRIVATE_KEY`` and ``WEBHOOK_SECRET`` — is dropped.  Keys absent from
    the parent env are simply omitted.
    """
    keep = [*_BASE_ENV_KEYS, *passthrough]
    return {key: os.environ[key] for key in keep if key in os.environ}


async def run_claude_subprocess(
    argv: list[str],
    *,
    timeout_seconds: float,
    token_cap: int,
    cwd: str | None = None,
    env_passthrough: Sequence[str] = (),
) -> ClaudeResult:
    """Default invoker: spawn claude, enforce the timeout and token cap, parse output.

    The subprocess is spawned with ``create_subprocess_exec`` (no shell) and is
    killed (and the failure raised) when the wall-clock timeout elapses or when
    claude's reported cumulative usage exceeds the cap.  It runs with a strict
    allowlisted env (see :func:`_build_subprocess_env`) so secrets in the worker's
    environment are not exposed to the model, and with ``cwd`` set to the workspace so
    default-scope Grep/Glob stay inside it (absolute-path reads are not bounded here —
    see the module docstring).

    Args:
        argv: The argument vector from :func:`build_claude_argv`.
        timeout_seconds: Wall-clock limit; the process is killed past it.
        token_cap: Cumulative-token ceiling; a run reporting more is rejected.
        cwd: Working directory for the subprocess; should be the seed workspace.
        env_passthrough: Extra parent-env keys to forward beyond the base allowlist.

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
        cwd=cwd,
        env=_build_subprocess_env(env_passthrough),
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
    env_passthrough: Sequence[str] = (),
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
        env_passthrough: Extra parent-env keys forwarded to the claude child.
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
    result = await invoker(
        argv,
        timeout_seconds=timeout_seconds,
        token_cap=token_cap,
        cwd=workspace_dir,
        env_passthrough=env_passthrough,
    )
    findings = parse_findings(result.stdout)
    logger.info(
        "Lens %s produced %d findings (%d tokens)",
        lens.name,
        len(findings),
        result.total_tokens,
    )
    return LensResult(lens_name=lens.name, findings=findings)


async def run_synthesis(
    *,
    lens_results: list[LensResult],
    claude_binary: str = "claude",
    token_cap: int = DEFAULT_TOKEN_CAP,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    env_passthrough: Sequence[str] = (),
    invoker: ClaudeInvoker = run_claude_subprocess,
    blocking: frozenset[Severity] = _BLOCKING_SEVERITIES,
) -> SynthesisResult:
    """Run the 4th synthesis ``claude -p`` pass over all lenses' findings.

    Feeds the combined per-lens findings into a bounded synthesis call that dedups
    overlapping findings, ranks by severity, and tags each survivor with its lens.
    The verdict reflects the highest-severity surviving finding and the body groups
    survivors by severity with each tagged by its originating lens.

    This is a pure reasoning pass over the findings JSON in the prompt: it is given
    no workspace and no tools (see :func:`build_synthesis_argv`), so it cannot read
    the seed and ``cwd`` is left unset.  Having nothing to confine, it is not
    sandboxed — only the three lenses are.

    Args:
        lens_results: Results of every lens that ran (Security, Design, Cleanliness).
        claude_binary: Path or name of the claude executable.
        token_cap: Per-agent cumulative-token ceiling (bounds the synthesis call).
        timeout_seconds: Wall-clock limit for the synthesis run.
        env_passthrough: Extra parent-env keys forwarded to the claude child.
        invoker: Coroutine that runs the subprocess; injected in tests.
        blocking: The severities that escalate the verdict to REQUEST_CHANGES;
            defaults to high/critical, overridden by the repo config threshold.

    Returns:
        A :class:`SynthesisResult` with the tagged survivors, verdict, and body.

    Raises:
        LensTimeoutError / LensTokenCapError: Propagated from the invoker when the
            synthesis run is aborted; the caller handles these as a failed pass.
    """
    argv = build_synthesis_argv(
        claude_binary=claude_binary,
        prompt=_build_synthesis_prompt(lens_results),
    )
    total_lens_findings = sum(len(r.findings) for r in lens_results)
    logger.info(
        "Running synthesis over %d lenses (%d findings)",
        len(lens_results),
        total_lens_findings,
    )
    result = await invoker(
        argv,
        timeout_seconds=timeout_seconds,
        token_cap=token_cap,
        cwd=None,
        env_passthrough=env_passthrough,
    )
    tagged = parse_tagged_findings(result.stdout)
    logger.info(
        "Synthesis kept %d of %d findings (%d tokens)",
        len(tagged),
        total_lens_findings,
        result.total_tokens,
    )
    return SynthesisResult(
        tagged_findings=tagged,
        verdict=verdict_for_tagged(tagged, blocking=blocking),
        body=format_synthesis_body(tagged),
    )
