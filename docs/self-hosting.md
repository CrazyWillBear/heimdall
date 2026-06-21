# Self-hosting setup

## Prerequisites

- **Python ≥ 3.12** and [`uv`](https://docs.astral.sh/uv/).
- A running **Redis** instance (the Arq queue).
- The **`claude` CLI** on the worker host (the lenses shell out to it), plus an
  **`ANTHROPIC_API_KEY`** in the worker's environment for the CLI to authenticate.
- A **GitHub App** (see below).

Install dependencies:

```
uv sync
```

## Create the GitHub App

Create a GitHub App and configure it to:

- subscribe to **Pull request** webhook events,
- point its webhook URL at your service's `/webhook` endpoint,
- set a **webhook secret** (matches `WEBHOOK_SECRET`),
- grant read access to PR contents/metadata and **read/write to Pull requests** (to post
  reviews, dismiss, and minimize),
- generate a **private key** (PEM) and note the **App ID** and **installation**.

Install the App on the repositories you want reviewed. A repo is only reviewed once it also
checks in a `.github/heimdall.yml` (see the [config reference](configuration.md)).

## Configure the environment

Set the service env settings (`heimdall/config.py`) — secrets via a `.env` file or injected
by the deployment. See the [Service env reference](configuration.md#service-env-reference) for the full list.

## Run the service and the worker

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

## Docker deployment

`docker-compose.yml` brings up the whole stack — **web** (the FastAPI service), **worker**
(the Arq worker), **redis** (the queue), and **caddy** (TLS termination + reverse proxy). The
shared image (`deploy/Dockerfile`) installs heimdall **non-editable** under a venv plus `bwrap`,
Node, and the `claude` CLI, so the lens sandbox can bind the venv read-only while the worker's
project/state dir is never exposed.

**1. Create the GitHub App (manifest flow).** Open `deploy/app-manifest.html` in a browser,
enter your domain, and click **Create GitHub App** — GitHub creates an App with exactly the
right permissions (Pull requests: read & write; Contents & Metadata: read) subscribed to
`pull_request`, with its webhook pointed at `https://<domain>/webhook`. GitHub then redirects to
`https://<domain>/?code=<CODE>` (a 404 page is fine — copy the `code` from the address bar) and
you exchange it for the credentials:

```
gh api -X POST /app-manifests/<CODE>/conversions
```

From the JSON response: put `id` in `GITHUB_APP_ID` and `webhook_secret` in `WEBHOOK_SECRET`
(in `.env`), and write `pem` to `secrets/github_app_private_key.pem`. Install the App on the
repos you want reviewed.

**2. Configure secrets.** Copy `.env.example` to `.env` and fill in `WEBHOOK_SECRET`,
`GITHUB_APP_ID`, `ANTHROPIC_API_KEY`, and `DOMAIN`. The App private key is a multiline PEM that
cannot live in an env file, so it is mounted as a Compose secret — put it at
`secrets/github_app_private_key.pem` (both `.env` and `secrets/` are gitignored).

**3. Point DNS at the host.** `DOMAIN` must resolve to this machine with ports 80 and 443 open;
Caddy obtains a certificate automatically on first start.

**4. Bring it up.**

```
docker compose up -d --build
```

**Sandbox requirements (worker only).** The worker runs each lens under `bwrap` using the
**unprivileged user-namespace** path — no setuid, no added capabilities. Docker's defaults block
it three ways, so the worker service runs with `seccomp=unconfined` (the default profile blocks
user-namespace creation), `systempaths=unconfined` (the default masks `/proc`), and
`apparmor=unconfined` (on AppArmor hosts — Debian/Ubuntu — the `docker-default` profile denies
the mount ops bwrap needs; `seccomp=unconfined` does not lift AppArmor). All three are already
wired in `docker-compose.yml`, applied to the worker alone. The worker runs its `bwrap`
exec-probe at startup and **refuses to boot** if the sandbox can't run, so a misconfiguration
surfaces immediately. Verify the sandbox inside the built image with:

```
docker compose run --rm worker bwrap --ro-bind / / --unshare-all --share-net -- true
```

An exit code of 0 means the sandbox works. (Do **not** add `no-new-privileges` expecting setuid
semantics — this deployment uses the userns path, not setuid.)

**OAuth (subscription) auth — optional.** By default the worker authenticates the `claude` CLI
with `ANTHROPIC_API_KEY` (zero extra infrastructure). To bill against a Claude **subscription**
instead, enable the `claude-refresher` sidecar with the `oauth` Compose profile. OAuth access
tokens expire after hours and the CLI refreshes them by rewriting `~/.claude/.credentials.json` —
a write the worker's lens sandbox blocks by design, so on a headless host nothing would keep the
token alive. The sidecar fixes that: it shares the worker's `~/.claude` through the `claude_creds`
volume **read-write** and periodically runs a throwaway `claude -p` ping that refreshes (and
rotates) the token as a side effect. The worker mounts the same volume **read-only** and its lens
sandbox re-binds it read-only on top, so credentials are never writable from any path that runs PR
code — **the security control is unchanged**. The refresher makes no GitHub calls and is handed
none of the App secrets.

1. **Seed the credentials once.** On a fresh host the `claude_creds` volume is empty, so
   authenticate `claude` once **into the volume** — run it interactively and complete the login,
   or use `claude setup-token`, or copy an existing `~/.claude/.credentials.json` in:

   ```
   docker compose run --rm -it claude-refresher claude
   ```

   Then leave `ANTHROPIC_API_KEY` **empty** in `.env` so the CLI uses the subscription rather than
   the key.

2. **Bring the stack up with the sidecar:**

   ```
   docker compose --profile oauth up -d --build
   ```

   Tune the sidecar with `CLAUDE_REFRESH_MODEL` (default `haiku`, the cheapest — the reply is
   discarded) and `CLAUDE_REFRESH_INTERVAL_SECONDS` (default `1800`; well under the token's
   hours-long lifetime). Without the `oauth` profile the sidecar never starts and the deployment
   stays on API-key auth.

**Replaying a webhook (no public tunnel).** To exercise a real review without GitHub delivering
a webhook (e.g. a private host), `scripts/replay_webhook.py` builds and signs a `pull_request`
payload and POSTs it to the service (published on `127.0.0.1:8000` by Compose). The App
credentials must be for a real installed App, since the worker still fetches the PR and posts the
review via an installation token:

```
uv run python scripts/replay_webhook.py \
    --repo owner/repo --pr 42 --sha <head-sha> --installation-id <id>
```

(`--secret` defaults to `$WEBHOOK_SECRET`, `--url` to `http://localhost:8000/webhook`.)
