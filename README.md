# Heimdall

A GitHub App that auto-reviews pull requests with a Claude-driven review engine.

## Review pipeline

1. **Webhook** (`heimdall/webhook.py`) verifies the signature and enqueues a job.
2. **Worker** (`heimdall/worker.py`) runs `run_review`: it assembles the PR seed
   context once, fans out three lenses over it, runs a synthesis pass over their
   combined findings, and posts exactly one PR review (idempotent per head SHA).
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
high/critical **surviving** finding to **REQUEST_CHANGES**, otherwise **COMMENT**.
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

Settings (`heimdall/config.py`) read from the environment / `.env`. Lens knobs:
`CLAUDE_BINARY`, `LENS_TOKEN_CAP`, `LENS_TIMEOUT_SECONDS`. Review knobs:
`REVIEW_TIMEOUT_SECONDS`, `DEBUG_LOGGING`.

## Development

```
uv run pytest
uv run ruff check .
uv run mypy .
```
