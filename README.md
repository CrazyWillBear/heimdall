# Heimdall

Heimdall is a self-hosted GitHub App that automatically reviews pull requests with a
Claude-driven, multi-lens review engine. When a PR is opened or updated it fans out three
independent review lenses — **Security**, **Design-fit**, and **Cleanliness** — over a
read-only seed of the PR, runs a synthesis pass that dedups and ranks their findings, and
posts exactly one PR review (inline comments plus a body) with a verdict.

Heimdall is **opt-in per repo**: a repository is only reviewed when it checks in a
`.github/heimdall.yml` file. PR code is **never executed** — every lens reads from a
materialized seed assembled purely from GitHub API data.

## How it works

```
GitHub  ──pull_request webhook──▶  Service (FastAPI)  ──enqueue──▶  Redis / Arq queue
                                        │ verify signature                   │
                                        │ ack 202 immediately                ▼
                                        └─ cancel stale job on new push   Worker (run_review)
                                                                             │
   gate (opt-in + scope + caps) ─▶ assemble seed ─▶ 3 lenses ─▶ synthesis ─▶ verdict
                                                                             │
                              dismiss/minimize prior review ─▶ post one review (inline + body)
```

### 1. The service (`heimdall/app.py`, `heimdall/webhook.py`)

A FastAPI app receives GitHub `pull_request` webhooks. For every request it:

1. **Verifies the HMAC-SHA256 signature** (`X-Hub-Signature-256`) against the configured
   webhook secret — an unsigned or mismatched request gets a 401.
2. Acts only on the relevant actions — `opened`, `reopened`, `synchronize`,
   `ready_for_review` — and ignores everything else (and skips draft PRs) with a 204.
3. **Enqueues** a `ReviewJob` onto the Redis/Arq queue and **acks 202 immediately**, so the
   review runs asynchronously in the worker. On a new push to a PR, any earlier *queued*
   (not-yet-running) job for the same PR is cancelled first (`cancel_stale_jobs`), so only
   the latest commit is reviewed.

The worker authenticates as the **GitHub App installation** (per-job, from the App id +
private key) to read the PR and post the review.

### 2. The worker pipeline (`heimdall/worker.py`)

`run_review` is the Arq task. For each job it runs, in order:

1. **Idempotency guard** — if this exact head SHA was already reviewed, skip.
2. **Gate** (`_gate_review`):
   - **Opt-in** — load `.github/heimdall.yml` from the trust-resolved ref. No file → the
     repo has not opted in → nothing is posted. A malformed file is skipped (never crashes
     the worker).
   - **Scope filters** (`skip_reason`) — base-branch allowlist, path globs, draft skip,
     bot-author skip, opt-out label. A filtered PR is skipped silently.
   - **Size caps** (`diff_cap_skip_note`) — a PR over `max_files` / `max_diff_lines` is
     skipped **with a posted COMMENT note** so the author learns why.
3. **Rate / concurrency caps** — a per-repo rolling-window budget (`max_reviews_per_window`
   over `rate_window_seconds`) and a per-installation concurrency cap
   (`max_concurrent_per_installation`), both DB-backed so they survive restarts.
4. **Assemble seed** (once) — materialize the PR seed into a temp workspace.
5. **Three lenses + synthesis** — fan out the config-tuned lenses over the shared seed, then
   a 4th synthesis pass dedups/ranks/tags their findings and writes the verdict.
6. **Retry-once / per-review timeout** — the whole pipeline is bounded by a per-review
   wall-clock timeout and retried exactly once on any failure/timeout. If the retry also
   fails, a terse **"review failed" COMMENT** is posted (never REQUEST_CHANGES) and the SHA
   recorded so the failed commit is not re-reviewed in a loop.
7. **Across-push lifecycle** — retire the prior Heimdall review (a REQUEST_CHANGES review is
   **dismissed**, a COMMENT review is **minimized** as outdated) and delete its stale inline
   comments, then **post exactly one review**: findings on a changed diff line ride as
   **inline comments** in the same submission, off-diff (or unparseable-location) findings
   render in the **body**.
8. **Metadata-only logging** — default logs carry only repo / PR / SHA / timing / verdict.
   Tokens and secrets are never logged; findings and code text are logged only when
   `DEBUG_LOGGING` is set.

### 3. The seed context (`heimdall/context.py`, `heimdall/context_cli.py`)

`assemble_pr_context` materializes a read-only workspace on disk from GitHub API data only —
**no PR code is executed**:

- `diff.patch` — the unified diff
- `pr_metadata.json` — title, body, author, base/head refs + SHAs, linked issues
- `files/<path>` — full content of each changed file at the head SHA (binary/oversize files
  skipped; path traversal rejected)
- `docs/<name>` — repo docs (`STYLEGUIDE.md`, `CLAUDE.md`, `README.md`)
  when present

Each lens reads this workspace through the **`heimdall-context`** CLI wrapper — the single
allowlisted Bash command — with subcommands `diff`, `pr`, `file <path>`, and `docs`.

### 4. The lenses and synthesis (`heimdall/lens.py`)

A *lens* is one read-only `claude -p` pass over the seed. Three built-ins run over the same
shared seed, each bounded independently:

| Lens          | Focus                                                | Model  | Effort |
| ------------- | ---------------------------------------------------- | ------ | ------ |
| `security`    | Security posture                                     | opus   | max    |
| `design`      | Design-fit / architecture                            | sonnet | high   |
| `cleanliness` | Readability, dead/duplicated code, doc hygiene       | sonnet | high   |

The `claude -p` invocation is headless with JSON output and restricts tools to the read-only
**Read / Grep / Glob** plus the single allowlisted **`heimdall-context`** Bash wrapper.
**Write** and **Edit** are explicitly disallowed; raw Bash carries no deny rule because an
unscoped Bash deny would override and neuter the wrapper's allow rule — under default-deny,
anything off the allowlist (including raw Bash) is already blocked. The subprocess is spawned
via `create_subprocess_exec` (no shell). PR code is therefore **never executed**.

**Filesystem-read confinement** is enforced at the OS level by a **bubblewrap (`bwrap`) sandbox**
wrapped around every lens (and the synthesis) `claude` subprocess. The seed is bound **read-only**
at the fixed in-sandbox path `/workspace` and nothing sensitive is reachable: the worker project
dir (its `.env` / `heimdall.db`) is **never** bound in, `/tmp` is a private tmpfs, and `~/.claude`,
the OS, CA, DNS, and `claude`/`node`/venv runtime paths are read-only. PID/IPC are unshared; the
network is kept (`--share-net`). So even an absolute-path `Read`/`Grep`/`Glob` from a prompt-injected
PR lands on a filesystem where no worker secret exists. The wrap is **fail-closed**: if `bwrap`
can't be resolved or the sandbox can't be built, that lens errors and is dropped — it never runs
unsandboxed. Configure nonstandard `claude`/`node`/CA locations via `SANDBOX_EXTRA_READ_ONLY_BINDS`
and the `bwrap` path via `BWRAP_BINARY`. Defence in depth still holds beneath the sandbox: a
**strict env allowlist** (only `PATH`/`HOME`/`ANTHROPIC_API_KEY` plus `CLAUDE_ENV_PASSTHROUGH`)
keeps secrets out of the child's environment, and PR code is never *executed*.

> **Requires `bwrap` on the worker host.** Install bubblewrap (it works in either setuid or
> unprivileged-userns mode). Without it, every lens fails closed and no review is produced.

Each run is bounded by a **per-agent cumulative-token cap** (default 400k) and a **per-lens
wall-clock timeout** (default 1800s); exceeding either kills the subprocess and drops that
lens. A failure in one lens is isolated — the rest still reach synthesis.

A **4th synthesis pass** (`run_synthesis`, opus/max) receives the combined findings of every
lens and: **dedups** overlapping findings across lenses, **ranks** by severity, **attributes**
each survivor to its originating lens, writes the **verdict**, and formats the
severity-grouped, lens-tagged body. When every lens fails or synthesis itself aborts, that run
produces no review (the retry/failure handling above takes over).

**Verdict.** Each finding carries a `Severity` (critical / high / medium / low). Any surviving
finding whose severity meets the repo's **blocking threshold** maps the review to
**REQUEST_CHANGES**; otherwise the review is a **COMMENT**. The default threshold is `high`,
so high/critical block.

### Persistence (`heimdall/db.py`)

State lives in SQLite so the service survives restarts: in-flight jobs, the last-reviewed SHA
per PR (idempotency), posted reviews (id + GraphQL node id + verdict, for the across-push
dismiss/minimize lifecycle), per-repo review timestamps (`review_events`, the rate/budget
cap), and a per-installation in-flight counter (`inflight_reviews`, the concurrency cap). All
of it is DB-backed, so the caps and lifecycle hold across worker restarts.

## Self-host setup

### Prerequisites

- **Python ≥ 3.12** and [`uv`](https://docs.astral.sh/uv/).
- A running **Redis** instance (the Arq queue).
- The **`claude` CLI** on the worker host (the lenses shell out to it), plus an
  **`ANTHROPIC_API_KEY`** in the worker's environment for the CLI to authenticate.
- A **GitHub App** (see below).

Install dependencies:

```
uv sync
```

### Create the GitHub App

Create a GitHub App and configure it to:

- subscribe to **Pull request** webhook events,
- point its webhook URL at your service's `/webhook` endpoint,
- set a **webhook secret** (matches `WEBHOOK_SECRET`),
- grant read access to PR contents/metadata and **read/write to Pull requests** (to post
  reviews, dismiss, and minimize),
- generate a **private key** (PEM) and note the **App ID** and **installation**.

Install the App on the repositories you want reviewed. A repo is only reviewed once it also
checks in a `.github/heimdall.yml` (see the config reference below).

### Configure the environment

Set the service env settings (`heimdall/config.py`) — secrets via a `.env` file or injected
by the deployment. See the [Service env reference](#service-env-reference) for the full list.

### Run the service and the worker

The service and the worker are two processes that share Redis and the SQLite database.

Run the **web service** (the FastAPI app is built by the `create_app` factory):

```
uv run uvicorn --factory heimdall.app:create_app --host 0.0.0.0 --port 8000
```

Run the **worker** (the Arq worker that processes the queue) with either the console script
or Arq directly:

```
uv run heimdall-worker
# equivalently:
uv run arq heimdall.worker.WorkerSettings
```

The `heimdall-context` console script is invoked internally by the lenses; you do not run it
by hand.

## Operation

Once installed and configured:

1. A contributor opens or pushes to a PR in an opted-in repo.
2. GitHub delivers a `pull_request` webhook; the service verifies it and enqueues a job,
   cancelling any stale queued job for the same PR.
3. The worker gates the PR (opt-in + scope + caps), assembles the seed, runs the three lenses
   + synthesis, and posts one review with a verdict — REQUEST_CHANGES when a finding meets the
   repo's blocking threshold, otherwise COMMENT.
4. On a new push, the prior review is dismissed (REQUEST_CHANGES) or minimized (COMMENT) and a
   fresh review replaces it.

**Tuning a repo.** Edit `.github/heimdall.yml` to enable/disable or re-tune lenses, add custom
lenses, change the blocking threshold, narrow scope, or adjust the guardrail caps.

**Fork safety.** Config is read from the **base** branch ref by default. Only a same-repo PR
from a trusted author association (`OWNER` / `MEMBER` / `COLLABORATOR`) may have its **head**
config honored; a **fork PR is always forced to the base ref**, so a malicious fork can never
ship a config that disables the security lens, injects a lens prompt, or widens scope.

## `.github/heimdall.yml` config reference

Per-repo behavior lives in `RepoConfig` (`heimdall/repo_config.py`) and is **opt-in**: a repo
with **no `.github/heimdall.yml` is never reviewed**. The file's mere presence opts the repo
in; a bare/empty file uses all defaults. Unknown keys are rejected (`extra: forbid`).

Every field below is optional and shown with its **real default**.

```yaml
# .github/heimdall.yml — every field optional; shown with its default.

severity_threshold: high          # lowest severity that blocks (REQUEST_CHANGES);
                                  # below it findings only comment. Default: high.

lenses:                           # per-lens overrides, keyed by built-in lens name.
  security:                       # built-in: opus / max
    enabled: true                 # a disabled lens never runs and never reaches synthesis
    model: opus                   # override the lens's default model (else built-in default)
    effort: max                   # override the lens's default effort (else built-in default)
    instructions: |               # APPENDED to (not a replacement for) the built-in prompt
      Pay extra attention to auth and input validation.
  design:                         # built-in: sonnet / high
    enabled: true
  cleanliness:                    # built-in: sonnet / high
    enabled: true

custom_lenses:                    # user-defined lenses that run alongside the built-ins
  - name: accessibility           # required; unique, must not collide with a built-in name
    system_prompt: |              # required; the lens's review instructions
      Review the diff for web accessibility (WCAG) regressions.
    model: sonnet                 # optional. Default: sonnet
    effort: high                  # optional. Default: high

scope:                            # filters that skip the PR entirely
  base_branches: []               # allowlist of base branches; empty = any. Default: []
  paths: []                       # fnmatch allowlist of changed paths; empty = any. Default: []
  skip_drafts: true               # skip draft PRs. Default: true
  skip_bot_authors: true          # skip PRs authored by a bot account. Default: true
  opt_out_label: null             # skip a PR carrying this label. Default: null (unset)

caps:                             # guardrail caps; every field has a SAFE, non-unbounded default
  max_files: 75                   # skip (with a posted note) a PR over this many files. Default: 75
  max_diff_lines: 20000           # skip (with a posted note) a PR over this many changed
                                  # lines (additions + deletions). Default: 20000
  max_reviews_per_window: 20      # per-repo budget: max reviews started per window. Default: 20
  rate_window_seconds: 3600       # rolling window the per-repo budget is measured over (s).
                                  # Default: 3600
  max_concurrent_per_installation: 4  # max reviews running at once per installation. Default: 4
```

### Field-by-field

**Top level (`RepoConfig`)**

| Field                | Type                       | Default | Meaning                                                              |
| -------------------- | -------------------------- | ------- | ------------------------------------------------------------------- |
| `lenses`             | map of name → lens config  | `{}`    | Per-built-in-lens overrides; absent lenses keep their defaults.     |
| `custom_lenses`      | list of custom lenses      | `[]`    | User-defined lenses that run alongside the built-ins.               |
| `severity_threshold` | `critical`/`high`/`medium`/`low` | `high` | Lowest severity that blocks (REQUEST_CHANGES); below it comments. |
| `scope`              | scope filters              | all defaults | Whether the PR is reviewed at all.                             |
| `caps`               | guardrail caps             | all defaults | Resource ceilings on review work.                             |

**Per-lens override (`lenses.<security|design|cleanliness>`, `LensConfig`)** — overrides a
built-in lens. `instructions` is **appended** to the built-in system prompt (the lens keeps its
built-in identity). Read only from the trusted config ref, so a fork cannot inject prompt text.

| Field          | Type      | Default | Meaning                                                  |
| -------------- | --------- | ------- | -------------------------------------------------------- |
| `enabled`      | bool      | `true`  | When false the lens never runs and never reaches synthesis. |
| `model`        | string    | unset   | Overrides the lens's default Claude model.               |
| `effort`       | string    | unset   | Overrides the lens's default reasoning effort.           |
| `instructions` | string    | unset   | Extra guidance **appended** to the built-in system prompt. |

**Custom lens entry (`custom_lenses[]`, `CustomLensConfig`)** — a user-defined lens run over the
same shared seed via the same bounded `run_lens` path; its findings reach synthesis tagged by
`name`. Its `system_prompt` is read from the trusted ref (base for forks), so a fork cannot
inject a custom-lens prompt.

| Field           | Type   | Default  | Meaning                                                       |
| --------------- | ------ | -------- | ------------------------------------------------------------- |
| `name`          | string | required | Unique lens id; **must not** collide with a built-in name.    |
| `system_prompt` | string | required | The lens's review instructions.                               |
| `model`         | string | `sonnet` | Claude model for the pass.                                    |
| `effort`        | string | `high`   | Reasoning effort for the pass.                                |

A custom-lens `name` that shadows a built-in (`security` / `design` / `cleanliness`) or
duplicates another custom lens is **rejected at load time**.

**Scope filters (`scope`, `ScopeFilters`)** — applied in order; the first failing filter skips
the PR (silently).

| Field              | Type          | Default | Meaning                                                       |
| ------------------ | ------------- | ------- | ------------------------------------------------------------- |
| `base_branches`    | list[string]  | `[]`    | Allowlist of base branches; empty = any base branch.          |
| `paths`            | list[string]  | `[]`    | `fnmatch` allowlist of changed paths; empty = any. `*` matches `/`, so `src/*` ≡ `src/**`. |
| `skip_drafts`      | bool          | `true`  | Skip draft PRs.                                                |
| `skip_bot_authors` | bool          | `true`  | Skip PRs authored by a bot account.                           |
| `opt_out_label`    | string        | unset   | When set and present on the PR, skip the PR.                  |

**Guardrail caps (`caps`, `GuardrailCaps`)** — every cap has a safe default, so an absent `caps`
block still has ceilings (a missing cap never means "unlimited"). All counters are SQLite-backed
and survive restarts.

| Field                             | Type  | Default | Meaning                                                                 |
| --------------------------------- | ----- | ------- | ----------------------------------------------------------------------- |
| `max_files`                       | int   | `75`    | Skip (with a posted COMMENT note) a PR changing more files than this.   |
| `max_diff_lines`                  | int   | `20000` | Skip (with a posted note) a PR whose changed lines exceed this.         |
| `max_reviews_per_window`          | int   | `20`    | Per-repo budget: max reviews **started** per rolling window.            |
| `rate_window_seconds`             | float | `3600`  | The rolling window the per-repo budget is measured over (seconds).      |
| `max_concurrent_per_installation` | int   | `4`     | Max reviews running concurrently per GitHub App installation.           |

Over the size caps the PR is skipped **with a posted note** (`diff_cap_skip_note`) — unlike the
silent scope skips — so the author learns why. Over the rate budget the review is skipped
silently and the SHA is **not** recorded, so a later push still gets reviewed. At the concurrency
cap the run defers (the slot is atomically claimed at start and released on every exit path).

## Service env reference

Service-level configuration (`Settings` in `heimdall/config.py`) is read from the environment
or a `.env` file. Secrets must come from env/`.env` — never commit them.

| Env var                  | Required | Default                                | Meaning                                                                 |
| ------------------------ | -------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `WEBHOOK_SECRET`         | yes      | —                                      | GitHub webhook HMAC secret used to verify `X-Hub-Signature-256`.        |
| `GITHUB_APP_ID`          | yes      | —                                      | GitHub App numeric ID.                                                  |
| `GITHUB_APP_PRIVATE_KEY` | yes      | —                                      | PEM-encoded RSA private key for the App (installation auth).            |
| `REDIS_URL`              | no       | `redis://localhost:6379`               | Redis connection URL for the Arq queue.                                 |
| `DATABASE_URL`           | no       | `sqlite+aiosqlite:///./heimdall.db`    | SQLite database URL for persistence.                                    |
| `CLAUDE_BINARY`          | no       | `claude`                               | Path or name of the `claude` CLI the lenses invoke.                     |
| `CLAUDE_ENV_PASSTHROUGH` | no       | `[]`                                   | Extra env-var names forwarded to the `claude` child beyond the `PATH`/`HOME`/`ANTHROPIC_API_KEY` allowlist (e.g. `HTTPS_PROXY`, `NODE_EXTRA_CA_CERTS`). |
| `BWRAP_BINARY`           | no       | `bwrap`                                | Path or name of the bubblewrap (`bwrap`) executable used to sandbox each lens `claude` subprocess; resolved on `PATH` unless an absolute path is given. |
| `SANDBOX_EXTRA_READ_ONLY_BINDS` | no | `[]`                                  | Extra host paths bound **read-only** into the lens sandbox, for nonstandard `claude`/`node`/CA installs. The seed, OS, CA, DNS, `~/.claude`, and venv are bound automatically; the worker project dir is **never** bound. |
| `LENS_TOKEN_CAP`         | no       | `400000`                               | Per-agent cumulative-token cap for a single lens run.                   |
| `LENS_TIMEOUT_SECONDS`   | no       | `1800`                                 | Per-lens wall-clock timeout (s) before a lens subprocess is killed.     |
| `REVIEW_TIMEOUT_SECONDS` | no       | `2400`                                 | Per-review wall-clock timeout (s) across the whole pipeline.            |
| `DEBUG_LOGGING`          | no       | `false`                                | When true, log findings and code text; default logs are metadata-only. |

The `claude` CLI on the worker host also needs `ANTHROPIC_API_KEY` in its environment to
authenticate the lens calls.

## Development

The full done-check — all three must pass:

```
uv run pytest
uv run ruff check .
uv run mypy .
```

## License

[Apache License 2.0](./LICENSE).
