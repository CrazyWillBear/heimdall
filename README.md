# Heimdall

A GitHub App that auto-reviews pull requests with a Claude-driven review engine.

## Review pipeline

1. **Webhook** (`heimdall/webhook.py`) verifies the signature and enqueues a job.
2. **Worker** (`heimdall/worker.py`) runs `run_review`: it first **gates** the PR
   (opt-in + scope filters + lens tuning — see [Per-repo config](#per-repo-config-githubheimdallyml)),
   then assembles the PR seed context once, fans out the configured lenses over it,
   runs a synthesis pass over their combined findings, and posts exactly one PR
   review (idempotent per head SHA).
3. **Seed context** (`heimdall/context.py`) materializes a workspace on disk
   (`diff.patch`, `pr_metadata.json`, `files/<path>`, `conventions/`) from GitHub API
   data only — no PR code is executed.
4. **Lens runner** (`heimdall/lens.py`) drives `claude -p` over that workspace.

## Lenses

A *lens* is one read-only `claude -p` pass over the seed workspace. The reusable seam
is `run_lens(lens=..., workspace_dir=..., ...) -> LensResult`. Three lenses run over the
same shared seed, each bounded independently:

- **`SECURITY_LENS`** — security posture (model **opus**, effort **max**).
- **`DESIGN_LENS`** — design-fit / architecture (model **sonnet**, effort **high**).
- **`CLEANLINESS_LENS`** — readability, dead/duplicated code, doc hygiene
  (model **sonnet**, effort **high**).

Model and effort live on each `LensSpec`, so `build_claude_argv` threads them through a
single code path. A failure in one lens is isolated (logged, that lens dropped); the rest
still reach synthesis.

The invocation (`build_claude_argv`) is headless with JSON output and restricts tools to
read-only **Read/Grep/Glob** plus the single allowlisted **`heimdall-context`** Bash
wrapper (`heimdall/context_cli.py`). **Write** and **Edit** are explicitly disallowed; raw
Bash carries no deny rule because an unscoped `Bash` deny would take precedence over and
neuter the wrapper's allow rule — under default-deny, anything off the allowlist (including
raw Bash) is already blocked. The subprocess is spawned via `create_subprocess_exec` (no
shell).

Each lens run is bounded by a **per-agent cumulative-token cap** (default 400k) and a
**per-lens wall-clock timeout** (default 1800s). Exceeding either kills the subprocess and
raises `LensTokenCapError` / `LensTimeoutError`; the worker logs the abort and drops that
lens.

## Synthesis

A 4th `claude -p` pass (`run_synthesis`, using `SYNTHESIS_LENS` on **opus/max**) receives
the combined findings of all three lenses and: **dedups** overlapping findings across
lenses, **ranks** by severity, attributes each survivor to its originating lens, writes the
**verdict**, and formats the review. The synthesis call is bounded by the same token cap
and timeout. When every lens fails, or synthesis itself aborts, that run produces no
review (the per-review retry/failure handling below then takes over).

### Failure, retry, and the per-review timeout

The whole review pipeline (`run_review`) is additionally bounded by a **per-review
wall-clock timeout** (default 2400s) that wraps the entire lens-fanout + synthesis pipeline
— distinct from, and looser than, the per-lens timeout. On any pipeline failure or timeout
(including every lens failing or synthesis aborting) the worker **retries exactly once**;
if the retry also fails it posts a terse **"Heimdall review failed" COMMENT** note (never
REQUEST_CHANGES, since a pipeline failure is not a verdict) and records the SHA so the
failed commit is not re-reviewed in a loop.

### Logging discipline

Default logs are **metadata-only** — repo / PR / SHA / timing / verdict. **Tokens and
secrets are never logged** (installation token, private key, `ANTHROPIC_API_KEY`). Findings
and code-snippet text are logged **only** when the `DEBUG_LOGGING` flag is set.

### Findings and verdict

Each lens reports `Finding`s carrying a `Severity` (critical/high/medium/low). Synthesis
returns `TaggedFinding`s (a finding plus its lens). `verdict_for_tagged(...)` maps any
**surviving** finding whose severity meets the repo's **blocking threshold** to
**REQUEST_CHANGES**, otherwise **COMMENT** (the default threshold is `high`, so high/critical
block — see [Per-repo config](#per-repo-config-githubheimdallyml)).
`format_synthesis_body(...)` renders the posted body: findings grouped by severity
(worst-first), each tagged with the lens that raised it.

### Across-push review lifecycle

When a PR receives a new push, `run_review` retires the prior Heimdall review before
posting the fresh one so only the latest review stays active. The prior review's id,
GraphQL node id, and verdict are persisted in SQLite (`posted_reviews` table), so the
lifecycle survives a worker restart. Per the stored verdict, a prior **REQUEST_CHANGES**
review is **dismissed** (REST `…/reviews/{id}/dismissals`) and a prior **COMMENT** review
is **minimized** as outdated (GraphQL `minimizeComment`) — dismissal is invalid for
COMMENT events. The freshly posted review then overwrites the stored record.

## Configuration

Two layers: **service config** (env-based, machine-wide) and **per-repo config**
(`.github/heimdall.yml`, checked into each reviewed repo).

### Service config

Settings (`heimdall/config.py`) read from the environment / `.env`. Lens knobs:
`CLAUDE_BINARY`, `LENS_TOKEN_CAP`, `LENS_TIMEOUT_SECONDS`. Review knobs:
`REVIEW_TIMEOUT_SECONDS`, `DEBUG_LOGGING`.

### Per-repo config (`.github/heimdall.yml`)

Per-repo behavior lives in `heimdall/repo_config.py` (`RepoConfig`) and is **opt-in**:
a repo with **no `.github/heimdall.yml` is never reviewed**. The file's mere presence
opts the repo in; a bare/empty file uses all defaults.

**Trust / fork safety.** Config is read from the **base** branch ref by default. Only a
same-repo PR from a trusted author association (`OWNER`/`MEMBER`/`COLLABORATOR`) may have
its **head** config honored; a **fork PR is always forced to the base ref**, so a malicious
fork cannot ship a config that disables the security lens or widens scope
(`config_ref_for_pr` / `is_trusted_pr`).

```yaml
# .github/heimdall.yml — every field optional; shown with its default.
severity_threshold: high        # lowest severity that blocks (REQUEST_CHANGES)
lenses:                         # per-lens model/effort/enable overrides
  security:
    model: opus
    effort: max
  design:
    enabled: false              # a disabled lens never runs and never reaches synthesis
scope:                          # filters that skip the PR entirely
  base_branches: [main]         # allowlist; empty = any base branch
  paths: ['src/**']             # glob allowlist of changed paths; empty = any path
  skip_drafts: true             # skip draft PRs
  skip_bot_authors: true        # skip PRs authored by a bot account
  opt_out_label: heimdall-skip  # skip a PR carrying this label
caps:                           # guardrail caps; every field has a SAFE default
  max_files: 75                 # skip (with a posted note) a PR over this many files
  max_diff_lines: 20000         # skip (with a posted note) a PR over this many changed lines
  max_reviews_per_window: 20    # per-repo budget: max reviews started per window
  rate_window_seconds: 3600     # rolling window the per-repo budget is measured over
  max_concurrent_per_installation: 4  # max reviews running at once per installation
```

The worker gate (`_gate_review` in `heimdall/worker.py`) loads this config, applies the
scope filters (`skip_reason`), then threads lens tuning (`tuned_lenses`) and the blocking
threshold (`blocking_severities`) into the pipeline.

**Guardrail caps** (`GuardrailCaps` in `heimdall/repo_config.py`) bound review cost; every
cap has a safe, non-unbounded default so an absent `caps` block still has ceilings. They are
enforced in three places, all in `run_review`:

- **Diff size / file count** (`max_files`, `max_diff_lines`) — checked in `_gate_review` from
  the PR-files payload. Over the cap the PR is **skipped with a posted COMMENT note**
  (`diff_cap_skip_note`) — unlike the silent scope skips — so the author learns why.
- **Per-repo rate / budget** (`max_reviews_per_window`, `rate_window_seconds`) — a DB-backed
  rolling-window count of review starts (`review_events` table). Over budget the review is
  skipped silently; the SHA is not recorded so a later push still gets a review.
- **Per-installation concurrency** (`max_concurrent_per_installation`) — a DB-backed
  in-flight counter per installation (`inflight_reviews` table). The cap **value** comes from
  the triggering repo's config, but the counter is genuinely per-installation (shared across
  that installation's repos). A slot is atomically claimed at review start
  (`try_acquire_inflight`, a single guarded upsert so concurrent acquirers never overshoot)
  and released in a `finally` on **every** exit path (`release_inflight`); at the cap the run
  defers (skips this delivery, no SHA recorded). All three counters are SQLite-backed, so the
  caps survive worker restarts.

## Development

```
uv run pytest
uv run ruff check .
uv run mypy .
```
