# Shared image for the Heimdall web service and the Arq worker.
#
# The worker runs each review lens as a `claude -p` subprocess confined in a bubblewrap
# (bwrap) sandbox.  The image therefore ships bwrap, the `claude` CLI, and Node, and
# installs heimdall **non-editable** into a venv so its code lives under a sandbox-bound
# path (the venv = sys.prefix) while the project dir — which would hold .env / the DB —
# is never bound.  See heimdall/lens.py:build_bwrap_prefix for the exact bind set.
#
# Sandbox mode: bwrap uses the *unprivileged user-namespace* path (no setuid).  Docker's
# default seccomp profile blocks user-namespace creation, so the worker container must run
# with `security_opt: ["seccomp=unconfined"]` (wired in docker-compose.yml).  The worker's
# startup exec-probe (heimdall/lens.sandbox_exec_probe) verifies this at boot.
FROM python:3.14-slim-bookworm

# OS packages: bubblewrap for the sandbox; curl/gnupg/ca-certificates for the NodeSource
# repo and TLS; then Node 22 (matches the host) for the claude CLI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        curl \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The Claude Code CLI the lenses shell out to (provides the `claude` command under /usr,
# which the sandbox binds read-only).
RUN npm install -g @anthropic-ai/claude-code

# uv for a reproducible, non-editable install from the committed lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Install heimdall + runtime deps non-editable into /opt/venv (= sys.prefix, bound ro in
# the sandbox).  Copy only what the wheel build needs so a source-only change still hits
# the dependency layer cache.
COPY pyproject.toml uv.lock README.md ./
COPY heimdall ./heimdall
RUN uv sync --frozen --no-dev --no-editable

COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Run unprivileged.  ~/.claude is hard-bound (--ro-bind) by the sandbox, so it must exist
# for the chosen HOME or every lens fails closed at spawn; /data holds the SQLite DB and a
# named volume inherits this dir's ownership on first mount.
RUN useradd --create-home --uid 10001 heimdall \
    && mkdir -p /home/heimdall/.claude /data \
    && chown -R heimdall:heimdall /home/heimdall /data
USER heimdall

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["heimdall-worker"]
