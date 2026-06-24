# Configuration

## `.github/heimdall.yml` config reference

Per-repo behavior lives in `RepoConfig` (`heimdall/repo_config.py`) and is **opt-in**: a repo
with **no `.github/heimdall.yml` is never reviewed**. The file's mere presence opts the repo
in; a bare/empty file uses all defaults. Unknown keys are rejected (`extra: forbid`).

Every field below is optional and shown with its **real default**.

```yaml
# .github/heimdall.yml — every field optional; shown with its default.

severity_threshold: high          # lowest severity that blocks (REQUEST_CHANGES);
                                  # below it findings only comment. Default: high.

docs:                             # repo-relative doc paths fed into every PR seed.
  - CLAUDE.md                     # setting `docs` FULLY REPLACES this default list;
  - README.md                     # `docs: []` means no docs; an absent field uses
  - AGENTS.md                     # these four. No globbing; absolute/`..` paths are
  - STYLEGUIDE.md                 # rejected at load. Default: the four shown here.

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

comments:                         # fold PR discussion into the review seed
  enabled: true                   # when false NO comments enter the seed (the whole comment
                                  # plumbing is skipped). Default: true
  max_comments: 50                # combined cap on inline threads + conversation comments
                                  # kept in the seed; over it the set is prioritized and
                                  # truncated (with an omission note). Default: 50
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
| `comments`           | comment-incorporation      | all defaults | Whether PR discussion is folded into the seed, and the cap on how much. |
| `docs`               | list of repo-relative paths | `[CLAUDE.md, README.md, AGENTS.md, STYLEGUIDE.md]` | Docs fed into every PR seed; setting it **fully replaces** the defaults, `[]` means none. Contents come from the PR head; the list from the trusted config. No globbing; absolute/`..` entries rejected at load. |

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
cap the run re-queues itself (arq `Retry`, ~60s backoff) and re-runs once a slot frees rather than
being dropped; the slot is atomically claimed when one is available and released on every exit
path. The deferrals are bounded by the worker's `max_tries`, after which the commit is dropped.

**Comment incorporation (`comments`, `CommentIncorporation`)** — controls whether the PR's
discussion (conversation comments, inline review threads, submitted-review summaries, and
Heimdall's own prior review) is folded into the review seed. Read from the trusted config ref
(base for forks), so a fork PR can never flip the toggle on its own head config.

| Field          | Type | Default | Meaning                                                                        |
| -------------- | ---- | ------- | ------------------------------------------------------------------------------ |
| `enabled`      | bool | `true`  | When false, **no comments enter the seed** — the whole comment plumbing is skipped (nothing fetched/materialized) and the review matches pre-feature behavior. |
| `max_comments` | int  | `50`    | Combined cap on inline threads + conversation comments kept in the seed; over it the set is prioritized and truncated (with an omission note in the posted body). |

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

The optional OAuth refresher sidecar (see [Docker deployment](self-hosting.md#docker-deployment) → OAuth) reads
two further env vars, consumed only by `heimdall-refresh` and deliberately **not** part of
`Settings` (the sidecar carries none of the App secrets): `CLAUDE_REFRESH_MODEL` (default `haiku`)
and `CLAUDE_REFRESH_INTERVAL_SECONDS` (default `1800`).
