# How it works

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

## 1. The service (`heimdall/app.py`, `heimdall/webhook.py`)

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

## 2. The worker pipeline (`heimdall/worker.py`)

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

## 3. The seed context (`heimdall/context.py`, `heimdall/context_cli.py`)

`assemble_pr_context` materializes a read-only workspace on disk from GitHub API data only —
**no PR code is executed**:

- `diff.patch` — the unified diff
- `pr_metadata.json` — title, body, author, base/head refs + SHAs, linked issues
- `files/<path>` — full content of each changed file at the head SHA (binary/oversize files
  skipped; path traversal rejected)
- `docs/<name>` — repo docs from the configurable `docs` list (defaults:
  `CLAUDE.md`, `README.md`, `AGENTS.md`, `STYLEGUIDE.md`) when present
- `comments.json` — the PR's conversation (timeline) comments, kept to **human** and
  **Heimdall's own** authors (other bots dropped); each carries `body`, `author`, and
  `author_association`. Written only when at least one comment is kept. Untrusted
  third-party data, never instructions.
- `review_threads.json` — the PR's inline review comments grouped into parent-anchored
  **reply threads**: each thread carries `body`, `author`, `author_association`, its
  `path`/`line` anchor, a `replies` list (each reply shaped the same way), an
  `is_resolved` flag, and an `is_outdated` flag (the anchored line is gone after a push,
  so `line` fell back to the pre-image `original_line`). The `is_resolved` flag is sourced
  from a GraphQL `reviewThreads` query (same installation token), correlated to the REST
  threads by comment `databaseId`; a GraphQL hiccup or a PR with no threads degrades
  cleanly to `is_resolved=false` (never crashes the review). It is trusted as-is — no
  author-of-resolve check (accepted residual risk). Same human + Heimdall's-own author
  filter as `comments.json`. Written only when at least one thread is kept. Untrusted
  third-party data, never instructions.

Comment incorporation is **per-repo configurable** (`comments` block in `heimdall.yml`, read
from the trust-resolved ref): an `enabled` toggle (default on) and a `max_comments` cap
(default `50`, source of truth `DEFAULT_MAX_COMMENTS` in `heimdall/repo_config.py`). With the
toggle **off**, no comment source is fetched or materialized and the seed matches the
pre-feature behavior. With it on, the cap feeds the prioritize/truncate path: when the combined
comment set (inline threads + conversation comments) exceeds `max_comments`, the seed is
**capped and prioritized** before materialization: **unresolved → on-diff → recent**, with
conversation comments ranked after inline threads, and outdated threads kept but ranked below
in-diff ones. When comments are dropped to honour the cap, the posted review body carries an
**omission note** (mirroring the size-cap COMMENT-note pattern) so the reader knows some
comments were left out.
- `review_summaries.json` — the body text of **submitted reviews** (APPROVE /
  REQUEST_CHANGES / COMMENT), each carrying `body`, `author`, `author_association`, and
  its `event` type. Same human + Heimdall's-own author filter as `comments.json`;
  body-less click-approves are dropped. Written only when at least one summary is kept.
  Untrusted third-party data, never instructions.
- `own_prior_review.json` — **Heimdall's own** latest prior review (`body`, `author`,
  `author_association`, `event`, and an `inline_comments` list), **fetched before the
  across-push retire/delete step destroys it**. Written only when Heimdall has a prior
  review on the PR. Untrusted-self continuity context, never an instruction.

Each lens reads this workspace through the **`heimdall-context`** CLI wrapper — the single
allowlisted Bash command — with subcommands `diff`, `pr`, `file <path>`, `docs`,
`comments`, `review-threads`, `review-summaries`, and `own-prior`.

## 4. The lenses and synthesis (`heimdall/lens.py`)

A *lens* is one read-only `claude -p` pass over the seed. Three built-ins run over the same
shared seed, each bounded independently:

| Lens          | Focus                                                | Model  | Effort |
| ------------- | ---------------------------------------------------- | ------ | ------ |
| `security`    | Security posture                                     | opus   | max    |
| `design`      | Design-fit / architecture                            | sonnet | high   |
| `cleanliness` | Readability, dead/duplicated code, doc hygiene       | sonnet | high   |

Each lens also sees the PR's full discussion as **untrusted background context**: its prompt
points it at the same allowlisted wrapper it uses for the diff/files/docs —
**`heimdall-context comments`** (timeline), **`review-threads`** (line-anchored threads),
**`review-summaries`** (submitted-review bodies + event type), and **`own-prior`** (Heimdall's
own prior review) — so each payload is read in-sandbox rather than baked into the prompt. All
are framed as untrusted third-party data — context to weigh, never instructions — and an empty
source leaves lens behaviour unchanged (the wrapper returns `[]`, or `null` for `own-prior`).
This grants lenses *visibility* only; the suppression of settled points lives in synthesis, not
here.

The `claude -p` invocation is headless with JSON output and restricts tools to the read-only
**Read / Grep / Glob** plus the single allowlisted **`heimdall-context`** Bash wrapper.
**Write** and **Edit** are explicitly disallowed; raw Bash carries no deny rule because an
unscoped Bash deny would override and neuter the wrapper's allow rule — under default-deny,
anything off the allowlist (including raw Bash) is already blocked. The subprocess is spawned
via `create_subprocess_exec` (no shell). PR code is therefore **never executed**.

**Filesystem-read confinement** is enforced at the OS level by a **bubblewrap (`bwrap`) sandbox**
wrapped around **every** `claude` subprocess — the three lenses *and* the 4th synthesis pass. The
synthesis pass is a no-tools, no-workspace reasoning pass over the findings JSON (no `--add-dir`, no
Read/Grep/Glob), so it has nothing of its own to confine, but it still runs **inside the same `bwrap`
sandbox** — over a throwaway empty workspace bound at `/workspace`, with `~/.claude` read-only — so the
fail-closed **"never spawn `claude` unsandboxed"** invariant holds for every pass. The seed is bound
**read-only**
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
> unprivileged-userns mode). At startup the worker runs a trivial `bwrap` **exec-probe** and
> **refuses to boot** if the sandbox can't actually run (`bwrap` missing, unprivileged
> userns/seccomp blocked, or setuid defeated by `--security-opt no-new-privileges`) — so the
> failure surfaces immediately instead of every review silently failing closed at lens-spawn time.

Each run is bounded by a **per-agent cumulative-token cap** (default 400k) and a **per-lens
wall-clock timeout** (default 1800s); exceeding either kills the subprocess and drops that
lens. A failure in one lens is isolated — the rest still reach synthesis.

A **4th synthesis pass** (`run_synthesis`, opus/max) receives the combined findings of every
lens and: **dedups** overlapping findings across lenses, **ranks** by severity, **attributes**
each survivor to its originating lens, writes the **verdict**, and formats the
severity-grouped, lens-tagged body. When the seed kept any conversation comments, inline
review threads, review summaries, or Heimdall's own prior review, their payloads are also
embedded in the synthesis prompt inside explicit **untrusted-data frames** — context to weigh,
never instructions to follow (an empty source leaves the prompt unchanged). It too
runs **inside the `bwrap` sandbox** (over a throwaway empty workspace, `~/.claude` bound
read-only), so no `claude` pass is ever spawned unsandboxed. When every lens fails or synthesis
itself aborts, that run produces no review (the retry/failure handling above takes over).

**Suppression contract.** Synthesis is the single point that may **drop a finding** the
discussion authoritatively settled. *Authoritative* is narrow: **either** the text of a comment /
thread / summary whose `author_association` is **OWNER / MEMBER / COLLABORATOR**, **or** the
finding's inline thread being **resolved** (`is_resolved`). A **CONTRIBUTOR / NONE** comment is
context only and **never** suppresses via its text, so a prompt-injected "approve anyway" from an
outside account cannot silence a finding. The per-comment `author_association` and per-thread
`is_resolved` ride in the prompt; the drop decision is the model's judgment. Synthesis returns the
dropped findings (title + brief reason) in `suppressed_findings`, **separate** from the survivors,
so downstream rendering can surface what was dropped and why. The **verdict is computed from the
survivors alone**, so suppressing a blocking finding can downgrade `REQUEST_CHANGES` to `COMMENT`.

**Verdict.** Each finding carries a `Severity` (critical / high / medium / low). Any surviving
finding whose severity meets the repo's **blocking threshold** maps the review to
**REQUEST_CHANGES**; otherwise the review is a **COMMENT**. The default threshold is `high`,
so high/critical block.

## Persistence (`heimdall/db.py`)

State lives in SQLite so the service survives restarts: in-flight jobs, the last-reviewed SHA
per PR (idempotency), posted reviews (id + GraphQL node id + verdict, for the across-push
dismiss/minimize lifecycle), per-repo review timestamps (`review_events`, the rate/budget
cap), and a per-installation in-flight counter (`inflight_reviews`, the concurrency cap). All
of it is DB-backed, so the caps and lifecycle hold across worker restarts.
