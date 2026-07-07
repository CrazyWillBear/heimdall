"""SQLite persistence for job tracking, posted reviews, and guardrail state.

Tables: in-flight jobs, last-reviewed head SHAs, posted reviews, plus the
guardrail state — per-repo review timestamps (``review_events``, for the rate/
budget cap) and a per-installation in-flight counter (``inflight_reviews``, for
the concurrency cap).  All of it is SQLite-backed so the caps survive restarts.

Uses aiosqlite for async access. The Database class owns the connection lifecycle;
helper functions receive a Database instance so they can be tested with an
in-memory fixture without touching the filesystem.
"""

from __future__ import annotations

import aiosqlite


class Database:
    """Thin wrapper around an aiosqlite connection with schema migration."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and create tables if they don't exist."""
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            await self._conn.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        """Return the open connection, raising if not initialised."""
        if self._conn is None:
            raise RuntimeError("Database not initialised; call initialize() first")
        return self._conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_jobs (
    repo_full_name TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    job_id         TEXT NOT NULL,
    head_sha       TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (repo_full_name, pr_number)
);

CREATE TABLE IF NOT EXISTS reviewed_shas (
    repo_full_name TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    sha            TEXT NOT NULL,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (repo_full_name, pr_number)
);

CREATE TABLE IF NOT EXISTS posted_reviews (
    repo_full_name TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    review_id      INTEGER NOT NULL,
    node_id        TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (repo_full_name, pr_number)
);

-- Guardrail: per-repo rate/budget.  One row per review actually started, keyed
-- by repo with a unix-epoch timestamp, so a rolling-window count can decide
-- whether a fresh review is within the per-repo budget.  Survives restart.
CREATE TABLE IF NOT EXISTS review_events (
    repo_full_name TEXT NOT NULL,
    occurred_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_events_repo_time
    ON review_events (repo_full_name, occurred_at);

-- Guardrail: per-installation concurrency.  A single in-flight counter per
-- installation, incremented when a review starts and decremented when it ends,
-- so concurrent reviews per installation never exceed the configured cap.  The
-- counter is DB-backed (not in-process) so the cap holds across worker restarts
-- and multiple worker processes sharing the same database.
CREATE TABLE IF NOT EXISTS inflight_reviews (
    installation_id INTEGER NOT NULL PRIMARY KEY,
    in_flight       INTEGER NOT NULL DEFAULT 0
);

-- on_signal trigger: PRs a repo has explicitly signaled for review.  Activation is
-- sticky (a removed review request never deactivates it) and drives the
-- scope.trigger: on_signal gate so every later push re-reviews an activated PR.  A
-- plain additive CREATE TABLE IF NOT EXISTS — no migration needed.
CREATE TABLE IF NOT EXISTS activated_prs (
    repo_full_name TEXT NOT NULL,
    pr_number      INTEGER NOT NULL,
    activated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (repo_full_name, pr_number)
);
"""


async def upsert_job(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
    job_id: str,
    head_sha: str,
) -> None:
    """Insert or replace the in-flight job record for a PR."""
    await db.conn.execute(
        """
        INSERT INTO review_jobs (repo_full_name, pr_number, job_id, head_sha)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(repo_full_name, pr_number) DO UPDATE SET
            job_id     = excluded.job_id,
            head_sha   = excluded.head_sha,
            created_at = datetime('now')
        """,
        (repo_full_name, pr_number, job_id, head_sha),
    )
    await db.conn.commit()


async def get_job(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> aiosqlite.Row | None:
    """Return the current in-flight job row for a PR, or None."""
    async with db.conn.execute(
        "SELECT * FROM review_jobs WHERE repo_full_name = ? AND pr_number = ?",
        (repo_full_name, pr_number),
    ) as cursor:
        return await cursor.fetchone()


async def set_last_reviewed_sha(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
    sha: str,
) -> None:
    """Record the last SHA for which a review was successfully posted."""
    await db.conn.execute(
        """
        INSERT INTO reviewed_shas (repo_full_name, pr_number, sha)
        VALUES (?, ?, ?)
        ON CONFLICT(repo_full_name, pr_number) DO UPDATE SET
            sha        = excluded.sha,
            updated_at = datetime('now')
        """,
        (repo_full_name, pr_number, sha),
    )
    await db.conn.commit()


async def get_last_reviewed_sha(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> str | None:
    """Return the last reviewed SHA for a PR, or None if never reviewed."""
    async with db.conn.execute(
        "SELECT sha FROM reviewed_shas WHERE repo_full_name = ? AND pr_number = ?",
        (repo_full_name, pr_number),
    ) as cursor:
        row = await cursor.fetchone()
        return str(row["sha"]) if row is not None else None


async def set_pr_activated(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> None:
    """Mark a PR as activated for review under the on_signal trigger.

    Idempotent and sticky: a repeat call is a no-op (the ``activated_at`` timestamp is
    NOT refreshed) and activation is never cleared here, so a later removed review
    request cannot deactivate the PR.  Every subsequent push then re-reviews it.
    """
    await db.conn.execute(
        """
        INSERT INTO activated_prs (repo_full_name, pr_number)
        VALUES (?, ?)
        ON CONFLICT(repo_full_name, pr_number) DO NOTHING
        """,
        (repo_full_name, pr_number),
    )
    await db.conn.commit()


async def is_pr_activated(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> bool:
    """Return True when the PR has been activated for review, else False."""
    async with db.conn.execute(
        "SELECT 1 FROM activated_prs WHERE repo_full_name = ? AND pr_number = ?",
        (repo_full_name, pr_number),
    ) as cursor:
        row = await cursor.fetchone()
        return row is not None


async def set_posted_review(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
    review_id: int,
    node_id: str,
    verdict: str,
) -> None:
    """Record the Heimdall review just posted for a PR.

    Stores the REST ``review_id`` (used for dismissal), the GraphQL ``node_id``
    (used for minimization), and the ``verdict`` event (REQUEST_CHANGES or
    COMMENT) so a later push can act on the prior review before posting a fresh
    one.  Overwrites any previously stored review for the same PR, keeping only
    the latest active review tracked.
    """
    await db.conn.execute(
        """
        INSERT INTO posted_reviews
            (repo_full_name, pr_number, review_id, node_id, verdict)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(repo_full_name, pr_number) DO UPDATE SET
            review_id  = excluded.review_id,
            node_id    = excluded.node_id,
            verdict    = excluded.verdict,
            updated_at = datetime('now')
        """,
        (repo_full_name, pr_number, review_id, node_id, verdict),
    )
    await db.conn.commit()


async def get_posted_review(
    db: Database,
    *,
    repo_full_name: str,
    pr_number: int,
) -> aiosqlite.Row | None:
    """Return the prior posted-review row for a PR, or None if never reviewed.

    The row exposes ``review_id``, ``node_id``, and ``verdict`` columns for the
    most recent Heimdall review on the PR.
    """
    async with db.conn.execute(
        "SELECT * FROM posted_reviews WHERE repo_full_name = ? AND pr_number = ?",
        (repo_full_name, pr_number),
    ) as cursor:
        return await cursor.fetchone()


async def record_review_event(
    db: Database,
    *,
    repo_full_name: str,
    occurred_at: float,
) -> None:
    """Record that a review for ``repo_full_name`` started at ``occurred_at``.

    Appends one row to ``review_events`` (a unix-epoch timestamp). Together with
    :func:`count_recent_reviews` this drives the per-repo rolling-window rate cap.
    The caller supplies the timestamp so tests can drive a deterministic clock.
    """
    await db.conn.execute(
        "INSERT INTO review_events (repo_full_name, occurred_at) VALUES (?, ?)",
        (repo_full_name, occurred_at),
    )
    await db.conn.commit()


async def try_record_review_event(
    db: Database,
    *,
    repo_full_name: str,
    occurred_at: float,
    cutoff: float,
    max_reviews: int,
) -> bool:
    """Atomically reserve a per-repo rate slot, recording the event only if under cap.

    Inserts one ``review_events`` row only when the count within the rolling window
    (``occurred_at >= cutoff``) is below ``max_reviews``.  The count-and-insert is a
    SINGLE SQL statement (an ``INSERT ... SELECT ... WHERE (subquery count) < cap``), so
    there is no read-then-write window another coroutine can interleave through — unlike
    a separate :func:`count_recent_reviews` + :func:`record_review_event`, the window
    budget can never be overshot.  Mirrors :func:`try_acquire_inflight`.

    Args:
        db: The database.
        repo_full_name: The repo whose rolling-window budget is gated.
        occurred_at: Unix-epoch timestamp recorded for the reserved event.
        cutoff: Window start (typically ``now - rate_window_seconds``); only events
            at/after it count toward the budget.
        max_reviews: The maximum reviews allowed within the window.

    Returns:
        True when a slot was reserved (caller may proceed), False when the repo is
        already at the window budget (caller must skip).
    """
    cursor = await db.conn.execute(
        """
        INSERT INTO review_events (repo_full_name, occurred_at)
        SELECT ?, ?
        WHERE (
            SELECT COUNT(*) FROM review_events
            WHERE repo_full_name = ? AND occurred_at >= ?
        ) < ?
        """,
        (repo_full_name, occurred_at, repo_full_name, cutoff, max_reviews),
    )
    await db.conn.commit()
    # rowcount is 1 when the WHERE guard held and the row was inserted; 0 when the
    # window was already at the budget.
    return cursor.rowcount == 1


async def count_recent_reviews(
    db: Database,
    *,
    repo_full_name: str,
    since: float,
) -> int:
    """Count reviews recorded for ``repo_full_name`` at or after ``since``.

    ``since`` is a unix-epoch cutoff (typically ``now - window``), so the count
    is the number of reviews within the rolling window — the value compared
    against the per-repo budget before a fresh review is allowed.
    """
    async with db.conn.execute(
        """
        SELECT COUNT(*) AS n FROM review_events
        WHERE repo_full_name = ? AND occurred_at >= ?
        """,
        (repo_full_name, since),
    ) as cursor:
        row = await cursor.fetchone()
        return int(row["n"]) if row is not None else 0


async def prune_review_events(
    db: Database,
    *,
    repo_full_name: str,
    before: float,
) -> None:
    """Delete ``review_events`` rows for a repo older than ``before``.

    Keeps the table bounded: once events fall outside the rolling window they can
    never affect a future count, so they are pruned to stop unbounded growth.
    """
    await db.conn.execute(
        "DELETE FROM review_events WHERE repo_full_name = ? AND occurred_at < ?",
        (repo_full_name, before),
    )
    await db.conn.commit()


async def try_acquire_inflight(
    db: Database,
    *,
    installation_id: int,
    cap: int,
) -> bool:
    """Atomically claim an in-flight review slot for an installation.

    Increments the installation's in-flight counter only when it is below ``cap``,
    and reports whether the slot was claimed.  The check-and-increment is a SINGLE
    SQL statement (an upsert whose UPDATE has a ``WHERE in_flight < cap`` guard), so
    there is no read-then-write window another coroutine could interleave through —
    the cap can never be overshot.  ``rowcount`` reports whether a row was written,
    which is the acquire result.  A successful acquire MUST be paired with exactly
    one :func:`release_inflight` (in a ``finally``) so the counter cannot leak on
    any exit path.

    Args:
        db: The database.
        installation_id: The GitHub App installation to gate.
        cap: The maximum concurrent reviews allowed for this installation.

    Returns:
        True when a slot was claimed (caller may proceed), False when the
        installation is already at the cap (caller must defer/skip).
    """
    cursor = await db.conn.execute(
        """
        INSERT INTO inflight_reviews (installation_id, in_flight)
        VALUES (?, 1)
        ON CONFLICT(installation_id) DO UPDATE SET
            in_flight = in_flight + 1
        WHERE in_flight < ?
        """,
        (installation_id, cap),
    )
    await db.conn.commit()
    # rowcount is 1 when a row was inserted (first acquire) or updated (the
    # WHERE guard held); 0 when the guard failed because the cap was reached.
    return cursor.rowcount == 1


async def release_inflight(
    db: Database,
    *,
    installation_id: int,
) -> None:
    """Release one in-flight review slot for an installation.

    Decrements the counter, clamped at zero so a stray double-release can never
    drive it negative (which would later over-admit reviews).  Idempotent for a
    missing row (treated as already zero).
    """
    await db.conn.execute(
        """
        UPDATE inflight_reviews
        SET in_flight = MAX(in_flight - 1, 0)
        WHERE installation_id = ?
        """,
        (installation_id,),
    )
    await db.conn.commit()


async def get_inflight_count(
    db: Database,
    *,
    installation_id: int,
) -> int:
    """Return the current in-flight review count for an installation (0 if none)."""
    async with db.conn.execute(
        "SELECT in_flight FROM inflight_reviews WHERE installation_id = ?",
        (installation_id,),
    ) as cursor:
        row = await cursor.fetchone()
        return int(row["in_flight"]) if row is not None else 0
