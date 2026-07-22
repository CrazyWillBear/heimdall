"""Per-repo Heimdall configuration loaded from ``.github/heimdall.yml``.

This is the REPO config — distinct from the env-based service config in
:mod:`heimdall.config`.  It is opt-in: a repo with no ``.github/heimdall.yml``
is never reviewed (see :func:`load_repo_config` returning ``None``).

Trust / fork safety (a SECURITY property): config is read from the BASE branch
ref by default.  A fork PR must NEVER have its head's config honored (a malicious
head could disable the security lens or widen scope), so only same-repo or
collaborator/member/owner PRs may read config from the head ref — see
:func:`config_ref_for_pr`.

The config tunes the review pipeline:
  * per-lens ``{model, effort, enabled}`` — which lenses run, and with which
    Claude model/effort (see :func:`tuned_lenses`);
  * a ``severity_threshold`` that decides which severity blocks (REQUEST_CHANGES)
    versus comments (see :func:`blocking_severities`);
  * scope filters that skip a PR entirely (base-branch allowlist, path globs,
    drafts, bot authors, opt-out label) — see :func:`skip_reason`.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from heimdall.lens import (
    CLEANLINESS_LENS,
    DESIGN_LENS,
    SECURITY_LENS,
    SYNTHESIS_LENS,
    LensSpec,
    Severity,
)

logger = logging.getLogger(__name__)

# Path read (from the base ref) to decide opt-in and load the config.
CONFIG_PATH = ".github/heimdall.yml"

# Author associations GitHub reports for users trusted to push to the base repo.
# A PR from one of these (and from the same repo) may have its head config honored;
# everything else (notably forks) is forced to read config from the base ref.
_TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# The built-in lenses, keyed by name so the per-lens config can address them.
_BUILTIN_LENSES: dict[str, LensSpec] = {
    SECURITY_LENS.name: SECURITY_LENS,
    DESIGN_LENS.name: DESIGN_LENS,
    CLEANLINESS_LENS.name: CLEANLINESS_LENS,
}

# Repo docs fetched into every PR seed by default when ``docs`` is absent.  This is
# the single source of truth for the default list (the materializer threads it through
# assemble_pr_context); setting ``docs`` in heimdall.yml fully replaces these.
_DEFAULT_DOCS = ("CLAUDE.md", "README.md", "AGENTS.md", "STYLEGUIDE.md")

# Safe, non-unbounded default ceiling on how many comments (inline threads +
# conversation comments combined) enter the seed.  Mirrors the guardrail-caps
# convention that every cap has a sensible default; lives here (consumed by
# context.assemble_pr_context) as the single source of truth so the comment-
# incorporation config block can reference it without a circular import.
DEFAULT_MAX_COMMENTS = 50


class LensConfig(BaseModel):
    """Per-lens override of the built-in :class:`~heimdall.lens.LensSpec` knobs.

    Attributes:
        enabled: When False the lens does not run and its findings never reach
            synthesis.
        model: Overrides the lens's default Claude model when set.
        effort: Overrides the lens's default reasoning effort when set.
        instructions: Extra guidance appended to the built-in lens's system prompt
            when set; the lens keeps its built-in identity and runs with the
            appended instructions.  Read only from the trusted config ref, so a fork
            cannot inject prompt text (see the module docstring's trust note).
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    model: str | None = None
    effort: str | None = None
    instructions: str | None = None


class CustomLensConfig(BaseModel):
    """A user-defined lens that runs alongside the built-ins.

    A custom lens is just another :class:`~heimdall.lens.LensSpec` built from config:
    it runs over the same shared seed via the same ``run_lens`` path (bounded by the
    same token cap + timeout) and its findings flow into synthesis tagged by ``name``.
    Like every other config field its ``system_prompt`` is sourced from the trusted
    ref (base for forks), so a fork PR can never inject a custom-lens prompt.

    Attributes:
        name: Stable lens identifier; must not collide with a built-in lens name and
            must be unique across custom lenses (it tags the lens's findings).
        system_prompt: The lens's review instructions (required).
        model: Claude model for the pass; defaults to "sonnet" like the non-security
            built-ins.
        effort: Reasoning effort for the pass; defaults to "high".
    """

    model_config = {"extra": "forbid"}

    name: str
    system_prompt: str
    model: str = "sonnet"
    effort: str = "high"


class SynthesisConfig(BaseModel):
    """Per-repo override of the synthesis pass's Claude model/effort.

    Only ``model`` and ``effort`` are exposed.  The synthesis pass cannot be disabled
    (it is the single dedup / suppression / verdict authority — dropping it would leave
    no reviewed output) and it takes no repo-supplied prompt: a custom synthesis prompt
    could instruct it to suppress every finding, so its built-in system prompt is fixed.
    Model and effort only tune the reasoning tier, so they carry no injection surface.

    Attributes:
        model: Overrides the synthesis pass's default Claude model when set.
        effort: Overrides the synthesis pass's default reasoning effort when set.
    """

    model_config = {"extra": "forbid"}

    model: str | None = None
    effort: str | None = None


class ResourceLimits(BaseModel):
    """Per-repo overrides of the operator's resource ceilings (env :class:`Settings`).

    Each knob mirrors the same-named field in :class:`heimdall.config.Settings`.  A repo
    may only TIGHTEN a limit: :func:`effective_limits` caps every override at the
    operator's value, so a repo can ask for cheaper/faster reviews but can never raise a
    ceiling to abuse the operator's compute.  An absent knob leaves the operator value
    untouched.  Values must be positive; the clamp against the operator ceiling is applied
    at use, not here, because the ceiling is a runtime service setting.

    Attributes:
        token_cap: Per-lens cumulative-token cap; clamped down to the operator's
            ``lens_token_cap``.
        lens_timeout_seconds: Per-lens wall-clock timeout; clamped down to the operator's
            ``lens_timeout_seconds``.
        review_timeout_seconds: Per-review wall-clock timeout across the whole pipeline;
            clamped down to the operator's ``review_timeout_seconds``.
    """

    model_config = {"extra": "forbid"}

    token_cap: int | None = Field(default=None, gt=0)
    lens_timeout_seconds: float | None = Field(default=None, gt=0)
    review_timeout_seconds: float | None = Field(default=None, gt=0)


class ScopeFilters(BaseModel):
    """Filters that decide whether a PR is in scope for review at all.

    Attributes:
        base_branches: Allowlist of base branch names; when non-empty a PR whose
            base branch is not listed is skipped.  Empty means "any base branch".
        paths: ``fnmatch`` allowlist of changed paths; when non-empty a PR whose changed
            files are all outside these patterns is skipped.  Empty means "any path".
            Note these are ``fnmatch`` patterns, not gitignore globs: ``*`` matches ``/``,
            so ``src/*`` and ``src/**`` are equivalent (see :func:`_paths_out_of_scope`).
        skip_drafts: Skip draft PRs when True.
        skip_bot_authors: Skip PRs authored by a bot account when True.
        opt_out_label: When set and present on the PR, the PR is skipped.
        trigger: When ``auto`` (the default) every in-scope PR event is reviewed —
            today's behavior.  When ``on_signal`` a PR is reviewed only after it has
            been activated by a ``ready_for_review``/``review_requested`` signal; the
            gate lives in the worker (deliberately NOT in :func:`skip_reason`, which
            stays a pure scope filter).
    """

    model_config = {"extra": "forbid"}

    base_branches: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    skip_drafts: bool = True
    skip_bot_authors: bool = True
    opt_out_label: str | None = None
    trigger: Literal["auto", "on_signal"] = "auto"


# ---------------------------------------------------------------------------
# Issue #10 — guardrail caps (diff size, per-repo rate/budget, concurrency).
# Kept in a self-contained block so it can be reconciled cleanly against the
# concurrently-built custom-lenses change (#9) that also extends RepoConfig.
# ---------------------------------------------------------------------------


class GuardrailCaps(BaseModel):
    """Resource guardrails bounding how much review work a repo can trigger.

    Every cap has a SAFE non-unbounded default, so a repo that opts in without a
    ``caps`` block still gets sensible ceilings — an absent cap never means
    "unlimited".

    Attributes:
        max_files: Skip (with a posted note) a PR changing more than this many
            files.  A huge PR is both expensive to review and low-signal.
        max_diff_lines: Skip (with a posted note) a PR whose total changed lines
            (additions + deletions across files) exceed this.
        max_reviews_per_window: Per-repo budget — at most this many reviews may
            START within ``rate_window_seconds``; beyond it a review is skipped.
        rate_window_seconds: The rolling window (seconds) the per-repo budget is
            measured over.
        max_concurrent_per_installation: At most this many reviews may run
            concurrently for one GitHub App installation; a review that would
            exceed it is deferred/skipped rather than started.
    """

    model_config = {"extra": "forbid"}

    max_files: int = Field(default=75, gt=0)
    max_diff_lines: int = Field(default=20_000, gt=0)
    max_reviews_per_window: int = Field(default=20, gt=0)
    rate_window_seconds: float = Field(default=3_600.0, gt=0)
    max_concurrent_per_installation: int = Field(default=4, gt=0)


def diff_cap_skip_note(
    caps: GuardrailCaps,
    *,
    file_count: int,
    diff_lines: int,
) -> str | None:
    """Return a terse skip note when a PR exceeds the size/file cap, else None.

    Distinct from the silent scope skips: when this returns a string the worker
    POSTS it as a COMMENT so the author learns the PR was skipped for size (and
    what the cap is) rather than silently getting no review.

    Args:
        caps: The repo's guardrail caps.
        file_count: Number of files changed in the PR.
        diff_lines: Total changed lines (additions + deletions) in the PR.

    Returns:
        A human-readable note when over either cap, or None to proceed.
    """
    if file_count > caps.max_files:
        return (
            f"Heimdall skipped this PR: too large to review "
            f"({file_count} files changed, cap {caps.max_files}). "
            "Split it into smaller PRs to get a review."
        )
    if diff_lines > caps.max_diff_lines:
        return (
            f"Heimdall skipped this PR: too large to review "
            f"({diff_lines} changed lines, cap {caps.max_diff_lines}). "
            "Split it into smaller PRs to get a review."
        )
    return None


class CommentIncorporation(BaseModel):
    """Per-repo control over folding PR discussion into the review seed.

    Both knobs read from the trust-resolved config ref (base for forks — see the
    module docstring), so a fork PR can never flip the toggle on its own head config.

    Attributes:
        enabled: When True (the default) the PR's conversation comments, inline
            review threads, submitted-review summaries, and Heimdall's own prior
            review are fetched and folded into the seed.  When False the whole
            comment plumbing is skipped — no comment source is fetched or
            materialized — so the seed and the synthesis/lens prompts match the
            pre-feature behavior.
        max_comments: Combined ceiling on inline threads + conversation comments
            that enter the seed; the value :func:`context.prioritize_comments`
            caps/truncates against.  Has a SAFE non-unbounded default, matching the
            guardrail-caps convention that an absent cap never means "unlimited".
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    max_comments: int = Field(default=DEFAULT_MAX_COMMENTS, gt=0)


class RepoConfig(BaseModel):
    """Parsed ``.github/heimdall.yml`` for one repository.

    Attributes:
        lenses: Per-lens overrides keyed by lens name (security/design/cleanliness).
            Lenses absent from the map keep their built-in defaults.
        custom_lenses: User-defined lenses that run alongside the built-ins and whose
            findings reach synthesis tagged by their name.
        synthesis: Per-repo override of the synthesis pass's model/effort (only — the
            pass can't be disabled and takes no repo prompt).  See :func:`tuned_synthesis`.
        limits: Per-repo resource-limit overrides (token cap, per-lens timeout, per-review
            timeout) that may only TIGHTEN the operator ceilings; see :func:`effective_limits`.
        severity_threshold: The lowest severity that blocks the PR
            (REQUEST_CHANGES); findings below it only comment.
        scope: Scope filters deciding whether the PR is reviewed at all.
        caps: Guardrail caps (diff size, per-repo rate/budget, per-installation
            concurrency) with safe defaults when the block is absent.
        comments: Comment-incorporation control — an enable toggle (default on) and
            the max-comments cap fed to the prioritize/truncate path.  When disabled
            no comments enter the seed at all (the plumbing is skipped); safe defaults
            apply when the block is absent.
        docs: Repo-relative paths fetched into every PR seed (no globbing); their
            contents come from the PR head, the list from this trusted config.
            Setting ``docs`` FULLY REPLACES the defaults; ``[]`` means no docs; an
            absent field uses the four defaults (:data:`_DEFAULT_DOCS`).
    """

    model_config = {"extra": "forbid"}

    lenses: dict[str, LensConfig] = Field(default_factory=dict)
    custom_lenses: list[CustomLensConfig] = Field(default_factory=list)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    severity_threshold: Severity = Severity.HIGH
    scope: ScopeFilters = Field(default_factory=ScopeFilters)
    caps: GuardrailCaps = Field(default_factory=GuardrailCaps)
    comments: CommentIncorporation = Field(default_factory=CommentIncorporation)
    docs: list[str] = Field(default_factory=lambda: list(_DEFAULT_DOCS))

    @model_validator(mode="after")
    def _validate_docs_paths(self) -> RepoConfig:
        """Reject docs entries that aren't safe repo-relative paths.

        Defense in depth on top of the materializer's path-escape guard: an
        absolute path or a ``..`` traversal entry is rejected at config load so a
        bad list never reaches the Contents API or the workspace.
        """
        for entry in self.docs:
            path = PurePosixPath(entry)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"docs entry {entry!r} must be a repo-relative path without '..'"
                )
        return self

    @model_validator(mode="after")
    def _validate_custom_lens_names(self) -> RepoConfig:
        """Reject custom-lens names that shadow a built-in or duplicate each other.

        A custom lens tags its findings by name, so a name collision would make the
        synthesized review ambiguous about which lens raised a finding.
        """
        seen: set[str] = set()
        for lens in self.custom_lenses:
            if lens.name in _BUILTIN_LENSES:
                raise ValueError(
                    f"custom lens {lens.name!r} collides with a built-in lens name"
                )
            if lens.name in seen:
                raise ValueError(f"duplicate custom lens name {lens.name!r}")
            seen.add(lens.name)
        return self


def parse_repo_config(text: str) -> RepoConfig:
    """Parse ``.github/heimdall.yml`` text into a :class:`RepoConfig`.

    An empty document (or one that parses to ``null``) yields all-default config —
    the file's mere presence is the opt-in, so a bare file still enables review.

    Args:
        text: The raw YAML content of ``.github/heimdall.yml``.

    Returns:
        The parsed, validated configuration.

    Raises:
        RepoConfigError: The YAML is malformed or fails schema validation.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RepoConfigError(f"heimdall.yml is not valid YAML: {exc}") from exc
    if data is None:
        return RepoConfig()
    if not isinstance(data, dict):
        raise RepoConfigError("heimdall.yml must be a mapping at the top level")
    try:
        return RepoConfig.model_validate(data)
    except ValidationError as exc:
        raise RepoConfigError(f"heimdall.yml failed validation: {exc}") from exc


class RepoConfigError(Exception):
    """Raised when ``.github/heimdall.yml`` is present but cannot be parsed."""


def is_trusted_pr(pr: dict[str, Any]) -> bool:
    """Return True when a PR may have its HEAD ``heimdall.yml`` honored.

    A PR is trusted only when it is NOT from a fork (head repo == base repo) and
    its author association is one a base-repo collaborator/member/owner carries.
    A fork PR is never trusted, regardless of author association, so a malicious
    fork cannot ship a config that weakens the review.

    Args:
        pr: The GitHub PR object (``get_pr`` response).

    Returns:
        True if the head config is safe to honor, False to force the base ref.
    """
    head_repo = (pr.get("head") or {}).get("repo") or {}
    base_repo = (pr.get("base") or {}).get("repo") or {}
    head_name = head_repo.get("full_name")
    base_name = base_repo.get("full_name")
    if head_name is None or base_name is None or head_name != base_name:
        return False
    association = str(pr.get("author_association", "")).upper()
    return association in _TRUSTED_ASSOCIATIONS


def config_ref_for_pr(pr: dict[str, Any]) -> str:
    """Return the git ref to read ``heimdall.yml`` from for this PR.

    Same-repo trusted PRs read from the head ref (so an in-progress config change
    takes effect on the PR that introduces it); fork PRs and untrusted same-repo
    PRs are forced to the base ref.

    Args:
        pr: The GitHub PR object.

    Returns:
        The head ref SHA when trusted, else the base ref SHA.
    """
    if is_trusted_pr(pr):
        return str(pr["head"]["sha"])
    return str(pr["base"]["sha"])


async def load_repo_config(
    github: Any,
    *,
    repo_full_name: str,
    pr: dict[str, Any],
) -> RepoConfig | None:
    """Load the repo config for a PR, or None when the repo has not opted in.

    Reads ``.github/heimdall.yml`` from the trust-resolved ref (base for forks,
    head for same-repo collaborator PRs — see :func:`config_ref_for_pr`).  A
    missing file means the repo has NOT opted in, so this returns None and the
    caller skips the review entirely (posting nothing).

    Args:
        github: An object exposing ``get_file_content`` (a GitHubClient).
        repo_full_name: e.g. "owner/repo".
        pr: The GitHub PR object, used to resolve the trusted config ref.

    Returns:
        The parsed :class:`RepoConfig`, or None when no config file exists.

    Raises:
        RepoConfigError: The file exists but is malformed/invalid.
    """
    ref = config_ref_for_pr(pr)
    try:
        text = await github.get_file_content(
            repo_full_name=repo_full_name,
            path=CONFIG_PATH,
            ref=ref,
            tolerate_missing=True,
        )
    except ValueError as exc:
        # Bad content encoding or an undecodable body (binascii.Error and
        # UnicodeDecodeError are both ValueError) — treat as a malformed config so the
        # caller skips cleanly rather than crashing on an uncaught exception.
        raise RepoConfigError(f"could not read {CONFIG_PATH}: {exc}") from exc
    if text is None:
        logger.info("No %s for %s; opt-in absent, skipping review", CONFIG_PATH, repo_full_name)
        return None
    return parse_repo_config(text)


def _author_is_bot(pr: dict[str, Any]) -> bool:
    """Return True when the PR author is a bot account (user.type == 'Bot')."""
    return str((pr.get("user") or {}).get("type", "")).lower() == "bot"


def _pr_labels(pr: dict[str, Any]) -> set[str]:
    """Return the set of label names attached to the PR."""
    return {
        str(label.get("name"))
        for label in pr.get("labels", [])
        if isinstance(label, dict) and label.get("name") is not None
    }


def _paths_out_of_scope(paths: list[str], changed_paths: list[str]) -> bool:
    """Return True when no changed path matches any allowlisted pattern.

    Patterns are matched with :func:`fnmatch.fnmatch`, NOT gitignore-style globbing:
    ``*`` matches ``/`` too, so ``src/*`` and ``src/**`` behave identically (both match
    ``src/a/b.py``). Use ``fnmatch`` semantics when writing ``paths`` entries. An empty
    ``paths`` allowlist means "any path", so it is never out of scope.
    """
    if not paths:
        return False
    return not any(
        fnmatch.fnmatch(changed, glob) for changed in changed_paths for glob in paths
    )


def skip_reason(
    config: RepoConfig,
    *,
    pr: dict[str, Any],
    changed_paths: list[str],
) -> str | None:
    """Return a human-readable reason to skip the PR, or None to proceed.

    Applies the scope filters in order: base-branch allowlist, path globs, draft
    skip, bot-author skip, opt-out label.  The first failing filter's reason is
    returned so the worker can log why a PR was skipped; None means in scope.

    Args:
        config: The repo configuration.
        pr: The GitHub PR object (base ref, draft flag, author, labels).
        changed_paths: Paths of files changed in the PR.

    Returns:
        A short reason string when the PR is out of scope, else None.
    """
    scope = config.scope
    base_ref = (pr.get("base") or {}).get("ref")
    if scope.base_branches and base_ref not in scope.base_branches:
        return f"base branch {base_ref!r} not in allowlist"
    if _paths_out_of_scope(scope.paths, changed_paths):
        return "no changed path matches the configured path globs"
    if scope.skip_drafts and pr.get("draft", False):
        return "PR is a draft and skip_drafts is set"
    if scope.skip_bot_authors and _author_is_bot(pr):
        return "PR author is a bot and skip_bot_authors is set"
    if scope.opt_out_label is not None and scope.opt_out_label in _pr_labels(pr):
        return f"opt-out label {scope.opt_out_label!r} present"
    return None


def _tuned_builtin(spec: LensSpec, lens_cfg: LensConfig) -> LensSpec:
    """Apply a :class:`LensConfig`'s overrides to a built-in :class:`LensSpec`.

    Per-lens ``instructions`` are appended to (not a replacement for) the built-in
    system prompt, so the lens keeps its built-in identity plus the extra guidance.
    """
    system_prompt = spec.system_prompt
    if lens_cfg.instructions:
        system_prompt = f"{spec.system_prompt}\n\n{lens_cfg.instructions}"
    return LensSpec(
        name=spec.name,
        system_prompt=system_prompt,
        model=lens_cfg.model or spec.model,
        effort=lens_cfg.effort or spec.effort,
    )


def tuned_lenses(config: RepoConfig) -> tuple[LensSpec, ...]:
    """Return the lenses to run: tuned built-ins plus any custom lenses.

    A built-in disabled in the config is dropped (it never runs and never reaches
    synthesis).  An enabled built-in keeps its spec unless the config overrides its
    model/effort or appends per-lens ``instructions`` to its system prompt.  Built-ins
    not mentioned keep their defaults and stay enabled.  Custom lenses (from
    ``config.custom_lenses``) are appended after the built-ins, each turned into a
    :class:`LensSpec`, so they run over the same shared seed via the same ``run_lens``
    path and their findings reach synthesis tagged by their name.

    All prompts come from the already-loaded :class:`RepoConfig`, which was read from
    the trust-resolved ref (base for forks); this function never re-reads the head, so
    fork safety is inherited.

    Args:
        config: The repo configuration.

    Returns:
        The tuned, enabled built-in lenses followed by the custom lenses.
    """
    tuned: list[LensSpec] = []
    for name, spec in _BUILTIN_LENSES.items():
        lens_cfg = config.lenses.get(name)
        if lens_cfg is None:
            tuned.append(spec)
            continue
        if not lens_cfg.enabled:
            continue
        tuned.append(_tuned_builtin(spec, lens_cfg))
    for custom in config.custom_lenses:
        tuned.append(
            LensSpec(
                name=custom.name,
                system_prompt=custom.system_prompt,
                model=custom.model,
                effort=custom.effort,
            )
        )
    return tuple(tuned)


def tuned_synthesis(config: RepoConfig) -> LensSpec:
    """Return the synthesis :class:`LensSpec` with the repo's model/effort override.

    Only ``model``/``effort`` are taken from ``config.synthesis``; the built-in synthesis
    system prompt is preserved verbatim (a repo cannot supply synthesis prompt text — see
    :class:`SynthesisConfig`).  An unset override keeps the built-in default.

    Args:
        config: The repo configuration.

    Returns:
        The synthesis lens spec, model/effort tuned to the config.
    """
    synth = config.synthesis
    return LensSpec(
        name=SYNTHESIS_LENS.name,
        system_prompt=SYNTHESIS_LENS.system_prompt,
        model=synth.model or SYNTHESIS_LENS.model,
        effort=synth.effort or SYNTHESIS_LENS.effort,
    )


@dataclass(frozen=True)
class EffectiveLimits:
    """The resource limits actually applied to a review after the repo/operator clamp.

    Produced by :func:`effective_limits`: each field is the repo override capped at the
    operator's ceiling (repo may tighten, never loosen), or the operator value when the
    repo set no override.

    Attributes:
        token_cap: Per-lens cumulative-token cap in effect.
        lens_timeout_seconds: Per-lens wall-clock timeout in effect.
        review_timeout_seconds: Per-review wall-clock timeout in effect.
    """

    token_cap: int
    lens_timeout_seconds: float
    review_timeout_seconds: float


def effective_limits(
    config: RepoConfig,
    *,
    token_cap: int,
    lens_timeout_seconds: float,
    review_timeout_seconds: float,
) -> EffectiveLimits:
    """Clamp the repo's resource-limit overrides against the operator ceilings.

    A repo may only TIGHTEN a limit: an override above the operator's value is capped
    down to it (``min``), and an absent override leaves the operator value.  This is the
    single enforcement point that keeps a repo from raising a ceiling to abuse the
    operator's compute — the trust ref already prevents a fork from setting these, and
    the clamp bounds even a trusted repo.

    Args:
        config: The repo configuration (its ``limits`` block).
        token_cap: The operator's per-lens token-cap ceiling.
        lens_timeout_seconds: The operator's per-lens timeout ceiling.
        review_timeout_seconds: The operator's per-review timeout ceiling.

    Returns:
        The clamped :class:`EffectiveLimits` to apply to this review.
    """
    limits = config.limits
    return EffectiveLimits(
        token_cap=(
            token_cap if limits.token_cap is None else min(limits.token_cap, token_cap)
        ),
        lens_timeout_seconds=(
            lens_timeout_seconds
            if limits.lens_timeout_seconds is None
            else min(limits.lens_timeout_seconds, lens_timeout_seconds)
        ),
        review_timeout_seconds=(
            review_timeout_seconds
            if limits.review_timeout_seconds is None
            else min(limits.review_timeout_seconds, review_timeout_seconds)
        ),
    )


# Severities ordered low-to-high so a threshold maps to "this severity and worse".
_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def blocking_severities(threshold: Severity) -> frozenset[Severity]:
    """Return the severities that block (REQUEST_CHANGES) at the given threshold.

    Every severity at or above ``threshold`` blocks; anything below it only
    comments.  A threshold of LOW makes every finding block; CRITICAL makes only
    critical findings block.

    Args:
        threshold: The lowest severity that should request changes.

    Returns:
        The frozenset of blocking severities.
    """
    cutoff = _SEVERITY_RANK[threshold]
    return frozenset(sev for sev, rank in _SEVERITY_RANK.items() if rank >= cutoff)
