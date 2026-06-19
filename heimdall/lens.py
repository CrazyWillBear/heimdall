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

Filesystem-read confinement is enforced at the OS level by a **bubblewrap (bwrap)
sandbox** wrapped around the claude subprocess (see :func:`build_bwrap_prefix`).  The
seed workspace is bound **read-only** at the fixed in-sandbox path ``/workspace`` and
nothing else of the worker's filesystem is reachable: the worker project dir (its
``.env`` / ``heimdall.db``) is never bound in, ``/tmp`` is a private tmpfs, ``~/.claude``
and the OS/CA/DNS/runtime paths are read-only.  ``--add-dir`` then *adds* ``/workspace``
to claude's allowed set; even an absolute-path Read/Grep/Glob from a prompt-injected PR
hits a filesystem where nothing sensitive exists.  Defence in depth still holds beneath
the sandbox: the child env is reduced to a strict allowlist (see
:func:`run_claude_subprocess`) so secrets are not in its environment, and PR code is
never *executed* (Bash off the allowlist is denied).

The sandbox is **fail-closed**: if the bwrap wrap cannot be built or run the lens errors
and is dropped (like a timeout) — it never falls back to an unsandboxed spawn.  The code
is mode-agnostic (works with either setuid or unprivileged-userns bwrap).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
    "heimdall-context wrapper (diff|pr|file|docs) and the read-only Read/Grep/"
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
    "wrapper (diff|pr|file|docs) and the read-only Read/Grep/Glob tools to inspect "
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
    "Use the heimdall-context wrapper (diff|pr|file|docs) and the read-only "
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


class SandboxError(LensError):
    """Raised when the bwrap sandbox cannot be built (fail-closed at spawn).

    A lens that hits this never runs unsandboxed; the caller drops it like a
    timeout so a prompt-injected PR can never escape onto the host filesystem.
    """


# Fixed in-sandbox mount for the read-only seed.  The lens cwd, ``--add-dir``,
# and every ``heimdall-context`` invocation reference this path (not the host
# path), so the worker's real directory layout is never exposed to the model.
SANDBOX_WORKSPACE_PATH = "/workspace"

# Default bwrap executable name; resolved on PATH unless an explicit path is given.
DEFAULT_BWRAP_BINARY = "bwrap"

# Read-only OS / CA / DNS paths the claude+node runtime needs inside the sandbox.
# Bound with --ro-bind-try so a path missing on this distro (e.g. /lib64 on merged-usr
# systems, /etc/pki vs /etc/ssl) is skipped rather than aborting the build.  The worker
# project dir is deliberately absent — that is what keeps .env / heimdall.db unreadable.
_SANDBOX_BASE_READ_ONLY = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
    "/etc/resolv.conf",
    "/etc/hosts",
    "/etc/nsswitch.conf",
)


def _resolve_bwrap(bwrap_binary: str = DEFAULT_BWRAP_BINARY) -> str | None:
    """Resolve the bwrap executable to an absolute path, or None when absent.

    An explicit absolute path is returned only when it exists and is executable;
    a bare name is looked up on ``PATH`` via :func:`shutil.which`.  Returning None
    drives the fail-closed behaviour in :func:`build_bwrap_prefix`.
    """
    if os.path.isabs(bwrap_binary):
        return bwrap_binary if os.access(bwrap_binary, os.X_OK) else None
    return shutil.which(bwrap_binary)


def _runtime_read_only_paths(argv_binary: str) -> list[str]:
    """Resolve the real claude+node binaries and the venv to read-only bind paths.

    The lens claude argv may reference ``claude`` by bare name; the sandbox needs the
    *resolved* executable plus its directory (claude is typically a launcher that runs
    ``node``) and the running Python prefix (so the ``heimdall-context`` console script
    in the venv resolves).  Paths missing on this host are skipped by ``--ro-bind-try``.
    """
    paths: list[str] = []
    resolved = shutil.which(argv_binary) if not os.path.isabs(argv_binary) else argv_binary
    if resolved:
        real = os.path.realpath(resolved)
        paths.append(real)
        paths.append(os.path.dirname(real))
    node = shutil.which("node")
    if node:
        paths.append(os.path.realpath(node))
    # The venv hosting heimdall-context (and its Python runtime).
    paths.append(sys.prefix)
    if sys.base_prefix != sys.prefix:
        paths.append(sys.base_prefix)
    return paths


def _claude_home() -> str:
    """Return the read-only ``~/.claude`` config dir path for the sandbox."""
    return str(Path(os.environ.get("HOME", str(Path.home()))) / ".claude")


def _dedup(*paths: str) -> list[str]:
    """De-duplicate non-empty paths, preserving first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            kept.append(path)
    return kept


def build_bwrap_prefix(
    *,
    workspace_dir: str,
    claude_binary: str = "claude",
    bwrap_binary: str = DEFAULT_BWRAP_BINARY,
    extra_read_only_binds: Sequence[str] = (),
) -> list[str]:
    """Build the bwrap argv prefix that confines a lens claude subprocess.

    The returned list is prepended to the claude argv; together they form the argv
    handed to the no-shell subprocess spawn.  The sandbox binds the seed **read-only**
    at the fixed :data:`SANDBOX_WORKSPACE_PATH`, gives a private ``/tmp`` tmpfs, mounts
    ``~/.claude`` and the operator's extra binds read-only (hard ``--ro-bind``), binds the
    optional OS/CA/DNS/runtime paths with ``--ro-bind-try`` (skipped when absent so the
    wrap stays mode-agnostic across distros), unshares PID/IPC, and keeps the network
    (``--share-net``).  The worker project dir is **never** bound, so its ``.env`` /
    ``heimdall.db`` stay unreadable — this is what closes the absolute-path read hole.
    Works with either setuid or unprivileged-userns bwrap.

    Args:
        workspace_dir: Host seed dir to bind read-only at ``/workspace``.
        claude_binary: Path/name of the claude CLI, resolved to bind its real binary.
        bwrap_binary: Path/name of the bwrap executable (default: found on PATH).
        extra_read_only_binds: Extra host paths to bind read-only (for nonstandard
            claude/node/CA installs); each surfaces as a ``--ro-bind`` flag.

    Returns:
        The bwrap argv prefix (``[bwrap, --ro-bind, …, --tmpfs, /tmp, …]``).

    Raises:
        SandboxError: bwrap could not be resolved — fail closed, never spawn unsandboxed.
    """
    bwrap = _resolve_bwrap(bwrap_binary)
    if bwrap is None:
        raise SandboxError(
            f"bwrap executable {bwrap_binary!r} not found; refusing to run a lens "
            "unsandboxed"
        )

    prefix = [bwrap, "--ro-bind", workspace_dir, SANDBOX_WORKSPACE_PATH]

    # Optional OS / CA / DNS / runtime paths vary by distro and install layout, so
    # they are bound with --ro-bind-try: bwrap silently skips a missing source rather
    # than aborting the (fail-closed) build.  This keeps the wrap mode-agnostic.
    for path in _dedup(*_SANDBOX_BASE_READ_ONLY, *_runtime_read_only_paths(claude_binary)):
        prefix += ["--ro-bind-try", path, path]

    # ~/.claude and the operator-chosen extra binds are bound hard (--ro-bind): they
    # are required and surface as explicit flags; a wrong path fails closed at spawn.
    for path in _dedup(_claude_home(), *extra_read_only_binds):
        prefix += ["--ro-bind", path, path]

    prefix += [
        "--tmpfs",
        "/tmp",  # noqa: S108 - in-sandbox tmpfs, not a host temp path
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--unshare-pid",
        "--unshare-ipc",
        "--share-net",
        "--chdir",
        SANDBOX_WORKSPACE_PATH,
    ]
    return prefix


# Wall-clock ceiling for the startup exec-probe.  The probe runs ``true`` inside the
# sandbox (milliseconds); a few seconds is generous and keeps a wedged bwrap from
# hanging worker boot indefinitely.
_SANDBOX_PROBE_TIMEOUT_SECONDS = 10.0


def build_sandbox_probe_argv(bwrap_binary: str = DEFAULT_BWRAP_BINARY) -> list[str]:
    """Build the trivial bwrap exec-probe argv used at worker startup.

    Exercises the SAME unshare / share-net machinery a real lens wrap uses
    (:func:`build_bwrap_prefix`) so a broken sandbox — bwrap missing, unprivileged
    userns or seccomp blocked, setuid defeated by ``no-new-privileges`` — is caught
    here rather than failing closed at lens-spawn time.  The probe binds ``/`` read-only,
    unshares every namespace, keeps the network, and runs ``true``.  argv is consumed by
    ``create_subprocess_exec`` (no shell), so none of these strings are shell-interpreted.

    Args:
        bwrap_binary: Path/name of the bwrap executable (default: found on PATH).

    Returns:
        The full probe argv (``[bwrap, --ro-bind, /, /, --unshare-all, --share-net,
        --, true]``).

    Raises:
        SandboxError: bwrap could not be resolved — fail closed, never spawn unsandboxed.
    """
    bwrap = _resolve_bwrap(bwrap_binary)
    if bwrap is None:
        raise SandboxError(
            f"bwrap executable {bwrap_binary!r} not found; refusing to start the worker "
            "with a sandbox that cannot run"
        )
    return [bwrap, "--ro-bind", "/", "/", "--unshare-all", "--share-net", "--", "true"]


async def sandbox_exec_probe(bwrap_binary: str = DEFAULT_BWRAP_BINARY) -> None:
    """Run a trivial bwrap exec-probe, raising :class:`SandboxError` if it fails.

    Spawns the :func:`build_sandbox_probe_argv` command once (no shell) and waits for it
    to exit.  A missing bwrap, a non-zero exit (unprivileged userns / seccomp blocked,
    setuid defeated by ``--security-opt no-new-privileges``), or a hang past the short
    timeout all raise :class:`SandboxError` so the worker refuses to boot on a host where
    the sandbox cannot actually run — instead of every review silently failing closed at
    lens-spawn time.

    Args:
        bwrap_binary: Path/name of the bwrap executable (default: found on PATH).

    Raises:
        SandboxError: bwrap is missing, the probe exited non-zero, or it timed out.
    """
    argv = build_sandbox_probe_argv(bwrap_binary)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=_SANDBOX_PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        await _kill(proc)
        raise SandboxError(
            f"sandbox exec-probe timed out after {_SANDBOX_PROBE_TIMEOUT_SECONDS}s; "
            "refusing to start the worker with a sandbox that cannot run"
        ) from exc
    if proc.returncode != 0:
        detail = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise SandboxError(
            f"sandbox exec-probe failed (exit {proc.returncode}): {detail}; refusing "
            "to start the worker — bwrap cannot run here (unprivileged userns/seccomp "
            "blocked, or setuid defeated by no-new-privileges)"
        )


async def run_claude_subprocess(
    argv: list[str],
    *,
    timeout_seconds: float,
    token_cap: int,
    cwd: str | None = None,
    env_passthrough: Sequence[str] = (),
    bwrap_binary: str = DEFAULT_BWRAP_BINARY,
    sandbox_extra_read_only_binds: Sequence[str] = (),
) -> ClaudeResult:
    """Default invoker: spawn claude in a bwrap sandbox, enforce timeout + token cap.

    The claude argv is wrapped in a bubblewrap (bwrap) OS sandbox (see
    :func:`build_bwrap_prefix`) so a prompt-injected PR cannot read host files by
    absolute path: the host ``cwd`` (the seed workspace) is bound **read-only** at the
    fixed :data:`SANDBOX_WORKSPACE_PATH`, the worker project dir is never bound, ``/tmp``
    is a private tmpfs, and the spawn cwd inside the sandbox is ``/workspace``.  The wrap
    is **fail-closed**: if it cannot be built the call raises :class:`SandboxError` and
    no claude process is ever spawned unsandboxed.

    The subprocess is spawned with no shell and is killed (and the failure raised) when
    the wall-clock timeout elapses or when claude's reported cumulative usage exceeds the
    cap.  Defence in depth holds beneath the sandbox: a strict allowlisted env (see
    :func:`_build_subprocess_env`) keeps secrets out of the child.

    Args:
        argv: The argument vector from :func:`build_claude_argv` (claude binary first;
            its ``--add-dir``/cwd already reference ``/workspace``).
        timeout_seconds: Wall-clock limit; the process is killed past it.
        token_cap: Cumulative-token ceiling; a run reporting more is rejected.
        cwd: Host seed workspace; bound read-only at ``/workspace`` in the sandbox.
        env_passthrough: Extra parent-env keys to forward beyond the base allowlist.
        bwrap_binary: Path/name of the bwrap executable (default: found on PATH).
        sandbox_extra_read_only_binds: Extra host paths to bind read-only (nonstandard
            claude/node/CA installs); each surfaces as a ``--ro-bind`` flag.

    Returns:
        A :class:`ClaudeResult` with stdout (claude's ``result`` text) and tokens.

    Raises:
        SandboxError: The bwrap wrap could not be built (fail-closed; never spawned).
        LensTimeoutError: The run exceeded ``timeout_seconds`` (subprocess killed).
        LensTokenCapError: The run exceeded ``token_cap`` (subprocess killed).
    """
    if cwd is None:
        raise SandboxError("a seed workspace cwd is required to build the bwrap sandbox")
    prefix = build_bwrap_prefix(
        workspace_dir=cwd,
        claude_binary=argv[0],
        bwrap_binary=bwrap_binary,
        extra_read_only_binds=sandbox_extra_read_only_binds,
    )
    proc = await asyncio.create_subprocess_exec(
        *prefix,
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=SANDBOX_WORKSPACE_PATH,
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
    bwrap_binary: str = DEFAULT_BWRAP_BINARY,
    sandbox_extra_read_only_binds: Sequence[str] = (),
    invoker: ClaudeInvoker = run_claude_subprocess,
) -> LensResult:
    """Run one lens over a materialized seed workspace and parse its findings.

    This is the reusable seam #5 calls once per lens.  It builds the read-only
    ``claude -p`` argv (scoped to the in-sandbox ``/workspace`` path), delegates
    execution (sandbox wrap + cap/timeout enforcement) to the injected ``invoker``
    — which binds the host ``workspace_dir`` read-only at ``/workspace`` — then parses
    structured findings.

    Args:
        lens: The lens to run (name + system prompt).
        workspace_dir: Materialized seed-context directory from assemble_pr_context;
            bound read-only at ``/workspace`` inside the sandbox.
        claude_binary: Path or name of the claude executable.
        token_cap: Per-agent cumulative-token ceiling.
        timeout_seconds: Wall-clock limit for the run.
        env_passthrough: Extra parent-env keys forwarded to the claude child.
        bwrap_binary: Path/name of the bwrap executable (default: found on PATH).
        sandbox_extra_read_only_binds: Extra host paths bound read-only in the sandbox.
        invoker: Coroutine that runs the subprocess; injected in tests.

    Returns:
        A :class:`LensResult` with the lens name and parsed findings.

    Raises:
        SandboxError / LensTimeoutError / LensTokenCapError: Propagated from the
            invoker when the run is aborted; callers handle these as a failed lens.
    """
    argv = build_claude_argv(
        claude_binary=claude_binary,
        workspace_dir=SANDBOX_WORKSPACE_PATH,
        lens=lens,
    )
    logger.info("Running lens %s over %s", lens.name, workspace_dir)
    result = await invoker(
        argv,
        timeout_seconds=timeout_seconds,
        token_cap=token_cap,
        cwd=workspace_dir,
        env_passthrough=env_passthrough,
        bwrap_binary=bwrap_binary,
        sandbox_extra_read_only_binds=sandbox_extra_read_only_binds,
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
