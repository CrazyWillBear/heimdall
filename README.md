<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/heimdall-emblem-dark.png">
    <img src="docs/assets/heimdall-emblem.png" alt="Heimdall" width="420">
  </picture>
  <br><br>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/heimdall-wordmark-dark.svg">
    <img src="docs/assets/heimdall-wordmark.svg" alt="HEIMDALL" width="400">
  </picture>
  <br><br>

  [![CI](https://img.shields.io/github/actions/workflow/status/CrazyWillBear/heimdall/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/CrazyWillBear/heimdall/actions/workflows/ci.yml)
  [![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=for-the-badge)](./LICENSE)
  [![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Release](https://img.shields.io/github/v/release/CrazyWillBear/heimdall?style=for-the-badge&color=success)](https://github.com/CrazyWillBear/heimdall/releases/latest)
  [![Codecov](https://img.shields.io/codecov/c/github/CrazyWillBear/heimdall?style=for-the-badge&logo=codecov&logoColor=white)](https://app.codecov.io/gh/CrazyWillBear/heimdall)
  [![Ruff](https://img.shields.io/badge/ruff-checked-261230?style=for-the-badge&logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
  [![Mypy](https://img.shields.io/badge/mypy-strict-2A6DB2?style=for-the-badge&logo=python&logoColor=white)](https://mypy-lang.org/)
  [![GHCR](https://img.shields.io/badge/ghcr.io-heimdall-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/CrazyWillBear/heimdall/pkgs/container/heimdall)
</div>

## 🛡️ Summary

Heimdall is a self-hosted GitHub App that automatically reviews pull requests with a
Claude-driven, multi-lens review engine. When a PR is opened or updated it fans out three
independent review lenses — 🔒 **Security**, 🧭 **Design-fit**, and 🧹 **Cleanliness** — over a
read-only seed of the PR, runs a synthesis pass that dedups and ranks their findings, and
posts exactly one PR review (inline comments plus a body) with a verdict.

Heimdall is **opt-in per repo**: a repository is only reviewed when it checks in a
`.github/heimdall.yml` file. PR code is **never executed** — every lens reads from a
materialized seed assembled purely from GitHub API data.

## 🧠 How it works

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

## 📚 Documentation

| Page | What's in it |
| ---- | ------------ |
| [Architecture](docs/architecture.md) | How it works end to end. |
| [Self-hosting](docs/self-hosting.md) | Full setup — GitHub App, Docker deployment, OAuth sidecar. |
| [Operation](docs/operation.md) | Review lifecycle, tuning a repo, fork safety. |
| [Configuration](docs/configuration.md) | Every `.github/heimdall.yml` field and service env var. |

## 🐳 Self-host

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

## 🧪 Development

The full done-check — all three must pass:

```
uv run pytest
uv run ruff check .
uv run mypy .
```

## ⚖️ License

[Apache License 2.0](./LICENSE).
