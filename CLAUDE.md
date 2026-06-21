# CLAUDE.md

> **Read first:** [`STYLEGUIDE.md`](./STYLEGUIDE.md) — code conventions, follow on every change.

Project-specific rules only. The universal working rules — boundaries, when-stuck,
secrets, done-honesty — live in the global `~/.claude/CLAUDE.md` and apply underneath
this. Only add a rule here when it differs from, or isn't covered by, the global.

## Definition of done

The project's full check — the global done-rule points here for the exact commands. All
must pass before any task is "done":

```
uv run pytest
uv run ruff check .
uv run mypy .
```

## Keep these docs current

Treat `CLAUDE.md` and `STYLEGUIDE.md` as living docs, not write-once boilerplate. As part
of a change that makes a rule here stale, wrong, or redundant, prune or rewrite it in the
same change; add a rule when a real, recurring need shows up. Keep it tight — fewer,
sharper lines beat an accreting pile. (This project directive overrides the global
ask-first-before-editing-docs rule for these two files.)

## Releases

**Branch flow.** `feature → staging → main`. `main` is release-only and protected: changes
land via PR from `staging` (admin override allowed). New work bases its worktree off
`origin/staging`, never `main`.

**Runbook.** (1) On `staging`, bump `version` in `pyproject.toml`. (2) PR `staging → main`,
merge on green CI. (3) On `main`, tag `vX.Y.Z` (matching `pyproject.toml`) and push it — the
tag triggers `release.yml`: multi-arch GHCR push (`:vX.Y.Z` + `:latest`) and a GitHub Release.

**Versioning.** Default: bump the patch (`Z`). Suggest a minor (`Y`) / major (`X`) bump when
the change warrants it; the user sets the final version.
