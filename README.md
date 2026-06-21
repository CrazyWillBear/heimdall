<p align="center">
  <img src="docs/assets/heimdall-emblem.png" alt="Heimdall" width="420">
</p>

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

The service verifies and enqueues webhooks; the worker gates each PR, assembles a read-only
seed, fans out the lenses, and posts one synthesized review. Full walkthrough —
service, worker pipeline, seed, lenses + synthesis, persistence — in
[docs/architecture.md](docs/architecture.md).

## Documentation

| Page | What's in it |
| ---- | ------------ |
| [Architecture](docs/architecture.md) | How it works end to end. |
| [Self-hosting](docs/self-hosting.md) | Full setup — GitHub App, Docker deployment, OAuth sidecar. |
| [Operation](docs/operation.md) | Review lifecycle, tuning a repo, fork safety. |
| [Configuration](docs/configuration.md) | Every `.github/heimdall.yml` field and service env var. |

## Self-host

Quickstart for a bare-metal run (full guide, including the GitHub App and Docker, in
[docs/self-hosting.md](docs/self-hosting.md)):

- **Prerequisites:** Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/), a running
  **Redis**, the **`claude` CLI** on the worker host with an **`ANTHROPIC_API_KEY`**, and a
  **GitHub App** subscribed to `pull_request` webhooks pointed at `/webhook`.
- **Install:** `uv sync`.
- **Configure:** set the service env (`heimdall/config.py`) via `.env` — see the
  [Service env reference](docs/configuration.md#service-env-reference).
- **Run** the two processes (they share Redis and SQLite):

  ```
  uv run uvicorn --factory heimdall.app:create_app --host 0.0.0.0 --port 8000
  uv run heimdall-worker
  ```

A repo is only reviewed once it checks in a `.github/heimdall.yml` — see the
[configuration reference](docs/configuration.md).

## Development

The full done-check — all three must pass:

```
uv run pytest
uv run ruff check .
uv run mypy .
```

## License

[Apache License 2.0](./LICENSE).
