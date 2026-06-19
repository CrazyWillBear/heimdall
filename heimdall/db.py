"""SQLite persistence for in-flight jobs and last-reviewed head SHAs.

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
