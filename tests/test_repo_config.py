"""Tests for the per-repo heimdall.yml config: opt-in, trust, scope, tuning.

Covers the model + helpers in heimdall.repo_config.  The GitHub client is mocked
so no network is touched: load_repo_config drives a stub get_file_content.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from heimdall.lens import Severity
from heimdall.repo_config import (
    CONFIG_PATH,
    GuardrailCaps,
    RepoConfig,
    RepoConfigError,
    blocking_severities,
    config_ref_for_pr,
    diff_cap_skip_note,
    is_trusted_pr,
    load_repo_config,
    parse_repo_config,
    skip_reason,
    tuned_lenses,
)

_REPO = "owner/repo"


def _pr(
    *,
    base_ref: str = "main",
    base_sha: str = "basesha",
    head_sha: str = "headsha",
    head_repo: str = "owner/repo",
    base_repo: str = "owner/repo",
    association: str = "OWNER",
    draft: bool = False,
    user_type: str = "User",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal GitHub PR object for the trust/scope helpers."""
    return {
        "draft": draft,
        "author_association": association,
        "user": {"login": "alice", "type": user_type},
        "labels": [{"name": name} for name in (labels or [])],
        "base": {"ref": base_ref, "sha": base_sha, "repo": {"full_name": base_repo}},
        "head": {"ref": "feature", "sha": head_sha, "repo": {"full_name": head_repo}},
    }


# ---------------------------------------------------------------------------
# Parsing + defaults
# ---------------------------------------------------------------------------


def test_bare_file_parses_to_all_defaults() -> None:
    """An empty heimdall.yml still opts in: defaults, every lens enabled."""
    config = parse_repo_config("")
    assert config.severity_threshold is Severity.HIGH
    assert config.lenses == {}
    assert config.scope.skip_drafts is True
    names = {lens.name for lens in tuned_lenses(config)}
    assert names == {"security", "design", "cleanliness"}


def test_invalid_yaml_raises_repo_config_error() -> None:
    """Malformed YAML is rejected with RepoConfigError, not a bare YAMLError."""
    with pytest.raises(RepoConfigError):
        parse_repo_config("lenses: [unterminated")


def test_unknown_top_level_key_rejected() -> None:
    """An unknown key is a typo signal — reject rather than silently ignore."""
    with pytest.raises(RepoConfigError):
        parse_repo_config("not_a_real_key: 1")


def test_non_mapping_top_level_rejected() -> None:
    """A top-level scalar/list is not a valid config document."""
    with pytest.raises(RepoConfigError):
        parse_repo_config("- just\n- a\n- list")


# ---------------------------------------------------------------------------
# Acceptance #1 — no heimdall.yml -> no review (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_config_returns_none() -> None:
    """A repo without heimdall.yml yields None — the caller skips the review."""
    github = AsyncMock()
    github.get_file_content = AsyncMock(return_value=None)

    config = await load_repo_config(github, repo_full_name=_REPO, pr=_pr())

    assert config is None


@pytest.mark.asyncio
async def test_present_config_loads() -> None:
    """A present heimdall.yml is parsed and returned (opt-in honored)."""
    github = AsyncMock()
    github.get_file_content = AsyncMock(return_value="severity_threshold: critical")

    config = await load_repo_config(github, repo_full_name=_REPO, pr=_pr())

    assert isinstance(config, RepoConfig)
    assert config.severity_threshold is Severity.CRITICAL


# ---------------------------------------------------------------------------
# Acceptance #2 — fork PR reads BASE config; head version is ignored
# ---------------------------------------------------------------------------


def test_fork_pr_is_not_trusted() -> None:
    """A PR from a different head repo is a fork — never trusted, even as OWNER."""
    pr = _pr(head_repo="attacker/repo", base_repo="owner/repo", association="OWNER")
    assert is_trusted_pr(pr) is False


def test_same_repo_collaborator_is_trusted() -> None:
    """A same-repo PR from a collaborator/member/owner is trusted."""
    assert is_trusted_pr(_pr(association="COLLABORATOR")) is True
    assert is_trusted_pr(_pr(association="MEMBER")) is True


def test_same_repo_outsider_is_not_trusted() -> None:
    """A same-repo PR from a non-collaborator association is not trusted."""
    assert is_trusted_pr(_pr(association="CONTRIBUTOR")) is False


def test_config_ref_forces_base_for_fork() -> None:
    """A fork PR resolves to the BASE sha, so the head config is never read."""
    pr = _pr(head_repo="attacker/repo", base_sha="BASE", head_sha="HEAD")
    assert config_ref_for_pr(pr) == "BASE"


def test_config_ref_uses_head_for_trusted() -> None:
    """A trusted same-repo PR resolves to the HEAD sha so in-PR config changes apply."""
    pr = _pr(association="OWNER", base_sha="BASE", head_sha="HEAD")
    assert config_ref_for_pr(pr) == "HEAD"


@pytest.mark.asyncio
async def test_fork_pr_loads_config_from_base_ref() -> None:
    """A fork PR loads heimdall.yml from the BASE sha — its head version is ignored."""
    github = AsyncMock()
    github.get_file_content = AsyncMock(return_value="severity_threshold: low")
    pr = _pr(head_repo="attacker/repo", base_sha="BASE_SHA", head_sha="HEAD_SHA")

    await load_repo_config(github, repo_full_name=_REPO, pr=pr)

    github.get_file_content.assert_awaited_once_with(
        repo_full_name=_REPO,
        path=CONFIG_PATH,
        ref="BASE_SHA",
        tolerate_missing=True,
    )


# ---------------------------------------------------------------------------
# Acceptance #3 — disable a lens, change threshold, base/path filters
# ---------------------------------------------------------------------------


def test_disabling_a_lens_drops_it() -> None:
    """A lens marked enabled: false does not appear in the tuned lens set."""
    config = parse_repo_config("lenses:\n  design:\n    enabled: false\n")
    names = {lens.name for lens in tuned_lenses(config)}
    assert "design" not in names
    assert names == {"security", "cleanliness"}


def test_per_lens_model_and_effort_override() -> None:
    """Per-lens model/effort overrides the built-in LensSpec defaults."""
    config = parse_repo_config(
        "lenses:\n  security:\n    model: sonnet\n    effort: high\n"
    )
    security = next(lens for lens in tuned_lenses(config) if lens.name == "security")
    assert security.model == "sonnet"
    assert security.effort == "high"


def test_unmentioned_lens_keeps_built_in_defaults() -> None:
    """A lens absent from the config keeps its built-in model/effort."""
    config = parse_repo_config("lenses:\n  design:\n    enabled: false\n")
    security = next(lens for lens in tuned_lenses(config) if lens.name == "security")
    assert security.model == "opus"
    assert security.effort == "max"


def test_blocking_threshold_default_high() -> None:
    """The default HIGH threshold blocks on high+critical, comments on medium/low."""
    blocking = blocking_severities(Severity.HIGH)
    assert blocking == {Severity.HIGH, Severity.CRITICAL}


def test_blocking_threshold_critical_only_blocks_critical() -> None:
    """A CRITICAL threshold blocks only critical findings; high becomes a comment."""
    blocking = blocking_severities(Severity.CRITICAL)
    assert blocking == {Severity.CRITICAL}
    assert Severity.HIGH not in blocking


def test_blocking_threshold_low_blocks_everything() -> None:
    """A LOW threshold makes every severity block."""
    blocking = blocking_severities(Severity.LOW)
    assert blocking == {Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}


def test_base_branch_filter_skips_non_allowlisted() -> None:
    """A base branch outside the allowlist is skipped."""
    config = parse_repo_config("scope:\n  base_branches: [main]\n")
    reason = skip_reason(config, pr=_pr(base_ref="dev"), changed_paths=["a.py"])
    assert reason is not None
    assert "base branch" in reason


def test_base_branch_filter_allows_listed() -> None:
    """A base branch in the allowlist proceeds (no skip)."""
    config = parse_repo_config("scope:\n  base_branches: [main]\n")
    reason = skip_reason(config, pr=_pr(base_ref="main"), changed_paths=["a.py"])
    assert reason is None


def test_path_filter_skips_when_no_path_matches() -> None:
    """When every changed path is outside the path globs, the PR is skipped."""
    config = parse_repo_config("scope:\n  paths: ['src/**']\n")
    reason = skip_reason(config, pr=_pr(), changed_paths=["docs/readme.md"])
    assert reason is not None
    assert "path" in reason


def test_path_filter_proceeds_when_any_path_matches() -> None:
    """A single changed path matching a glob keeps the PR in scope."""
    config = parse_repo_config("scope:\n  paths: ['src/**']\n")
    reason = skip_reason(
        config, pr=_pr(), changed_paths=["docs/readme.md", "src/app.py"]
    )
    assert reason is None


# ---------------------------------------------------------------------------
# Acceptance #4 — drafts, bot authors, opt-out label skipped
# ---------------------------------------------------------------------------


def test_draft_skipped_when_configured() -> None:
    """A draft PR is skipped when skip_drafts is set (the default)."""
    config = parse_repo_config("")
    reason = skip_reason(config, pr=_pr(draft=True), changed_paths=["a.py"])
    assert reason is not None
    assert "draft" in reason


def test_draft_allowed_when_skip_disabled() -> None:
    """A draft PR proceeds when skip_drafts is explicitly disabled."""
    config = parse_repo_config("scope:\n  skip_drafts: false\n")
    reason = skip_reason(config, pr=_pr(draft=True), changed_paths=["a.py"])
    assert reason is None


def test_bot_author_skipped_when_configured() -> None:
    """A bot-authored PR is skipped when skip_bot_authors is set (the default)."""
    config = parse_repo_config("")
    reason = skip_reason(config, pr=_pr(user_type="Bot"), changed_paths=["a.py"])
    assert reason is not None
    assert "bot" in reason


def test_bot_author_allowed_when_skip_disabled() -> None:
    """A bot-authored PR proceeds when skip_bot_authors is disabled."""
    config = parse_repo_config("scope:\n  skip_bot_authors: false\n")
    reason = skip_reason(config, pr=_pr(user_type="Bot"), changed_paths=["a.py"])
    assert reason is None


def test_opt_out_label_skips() -> None:
    """A PR carrying the configured opt-out label is skipped."""
    config = parse_repo_config("scope:\n  opt_out_label: heimdall-skip\n")
    reason = skip_reason(
        config, pr=_pr(labels=["heimdall-skip"]), changed_paths=["a.py"]
    )
    assert reason is not None
    assert "opt-out" in reason


def test_opt_out_label_absent_proceeds() -> None:
    """A PR without the opt-out label proceeds even when one is configured."""
    config = parse_repo_config("scope:\n  opt_out_label: heimdall-skip\n")
    reason = skip_reason(config, pr=_pr(labels=["bug"]), changed_paths=["a.py"])
    assert reason is None


# ---------------------------------------------------------------------------
# Issue #10 — guardrail caps: defaults, parsing, and the diff-size note
# ---------------------------------------------------------------------------


def test_caps_have_safe_defaults_when_absent() -> None:
    """A config without a caps block gets safe, bounded (non-unlimited) defaults."""
    config = parse_repo_config("")
    caps = config.caps
    assert caps.max_files > 0
    assert caps.max_diff_lines > 0
    assert caps.max_reviews_per_window > 0
    assert caps.rate_window_seconds > 0
    assert caps.max_concurrent_per_installation > 0


def test_caps_are_configurable() -> None:
    """Every cap can be overridden from heimdall.yml."""
    config = parse_repo_config(
        "caps:\n"
        "  max_files: 5\n"
        "  max_diff_lines: 100\n"
        "  max_reviews_per_window: 3\n"
        "  rate_window_seconds: 60\n"
        "  max_concurrent_per_installation: 2\n"
    )
    caps = config.caps
    assert caps.max_files == 5
    assert caps.max_diff_lines == 100
    assert caps.max_reviews_per_window == 3
    assert caps.rate_window_seconds == 60
    assert caps.max_concurrent_per_installation == 2


def test_cap_zero_or_negative_rejected() -> None:
    """A non-positive cap is meaningless (and unsafe) — reject it."""
    with pytest.raises(RepoConfigError):
        parse_repo_config("caps:\n  max_files: 0\n")


def test_unknown_cap_key_rejected() -> None:
    """An unknown caps key is a typo signal — reject rather than ignore."""
    with pytest.raises(RepoConfigError):
        parse_repo_config("caps:\n  max_lines: 100\n")


def test_diff_cap_note_under_cap_returns_none() -> None:
    """Within both caps, no skip note is produced (the PR is reviewed)."""
    caps = GuardrailCaps(max_files=10, max_diff_lines=1000)
    assert diff_cap_skip_note(caps, file_count=5, diff_lines=500) is None


def test_diff_cap_note_over_file_cap() -> None:
    """Over the file cap, a note naming the count and cap is returned."""
    caps = GuardrailCaps(max_files=10, max_diff_lines=10_000)
    note = diff_cap_skip_note(caps, file_count=11, diff_lines=50)
    assert note is not None
    assert "11 files" in note
    assert "10" in note


def test_diff_cap_note_over_line_cap() -> None:
    """Over the diff-line cap, a note naming the line count and cap is returned."""
    caps = GuardrailCaps(max_files=100, max_diff_lines=1000)
    note = diff_cap_skip_note(caps, file_count=3, diff_lines=1001)
    assert note is not None
    assert "1001 changed lines" in note
