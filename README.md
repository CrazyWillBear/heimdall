# Heimdall

A GitHub App that auto-reviews pull requests with a Claude-driven review engine.

## Review pipeline

1. **Webhook** (`heimdall/webhook.py`) verifies the signature and enqueues a job.
2. **Worker** (`heimdall/worker.py`) runs `run_review`: it assembles the PR seed
   context, runs the review lens(es) over it, maps findings to a verdict, and posts
   exactly one PR review (idempotent per head SHA).
3. **Seed context** (`heimdall/context.py`) materializes a workspace on disk
   (`diff.patch`, `pr_metadata.json`, `files/<path>`, `conventions/`) from GitHub API
   data only — no PR code is executed.
4. **Lens runner** (`heimdall/lens.py`) drives `claude -p` over that workspace.

## Lenses

A *lens* is one read-only `claude -p` pass over the seed workspace. The reusable seam
is `run_lens(lens=..., workspace_dir=..., ...) -> LensResult`. Today there is one lens,
`SECURITY_LENS`; more (Design-fit, Cleanliness) plus a synthesis pass plug into the same
runner.

The invocation (`build_claude_argv`) pins **opus at max effort**, headless with JSON
output, and restricts tools to read-only **Read/Grep/Glob** plus the single allowlisted
**`heimdall-context`** Bash wrapper (`heimdall/context_cli.py`). Raw Bash, Write, and Edit
are explicitly disallowed. The subprocess is spawned via `create_subprocess_exec` (no shell).

Each run is bounded by a **per-agent cumulative-token cap** (default 400k) and a
**wall-clock timeout** (default 1800s). Exceeding either kills the subprocess and raises
`LensTokenCapError` / `LensTimeoutError`; the worker logs the abort and posts nothing.

### Findings and verdict

Each lens reports `Finding`s carrying a `Severity` (critical/high/medium/low).
`verdict_for(findings)` maps any high/critical finding to **REQUEST_CHANGES**, otherwise
**COMMENT**. `format_review_body(findings)` renders the posted review body.

## Configuration

Settings (`heimdall/config.py`) read from the environment / `.env`. Lens knobs:
`CLAUDE_BINARY`, `LENS_TOKEN_CAP`, `LENS_TIMEOUT_SECONDS`.

## Development

```
uv run pytest
uv run ruff check .
uv run mypy .
```
