"""Tests for SQLite persistence: job tracking and last-reviewed SHA."""

from __future__ import annotations

import pytest
import pytest_asyncio

from heimdall.db import (
    Database,
    get_job,
    get_last_reviewed_sha,
    set_last_reviewed_sha,
    upsert_job,
)


@pytest_asyncio.fixture()
async def db() -> Database:
    """In-memory SQLite database for tests."""
    database = Database(":memory:")
    await database.initialize()
    return database


@pytest.mark.asyncio
async def test_upsert_and_get_job(db: Database) -> None:
    """Jobs can be stored and retrieved by PR key."""
    await upsert_job(db, repo_full_name="owner/repo", pr_number=1, job_id="jid-1", head_sha="abc")
    job = await get_job(db, repo_full_name="owner/repo", pr_number=1)
    assert job is not None
    assert job["job_id"] == "jid-1"
    assert job["head_sha"] == "abc"


@pytest.mark.asyncio
async def test_upsert_replaces_existing_job(db: Database) -> None:
    """A second upsert replaces the first (cancel-stale pattern)."""
    await upsert_job(db, repo_full_name="owner/repo", pr_number=1, job_id="jid-1", head_sha="aaa")
    await upsert_job(db, repo_full_name="owner/repo", pr_number=1, job_id="jid-2", head_sha="bbb")
    job = await get_job(db, repo_full_name="owner/repo", pr_number=1)
    assert job is not None
    assert job["job_id"] == "jid-2"
    assert job["head_sha"] == "bbb"


@pytest.mark.asyncio
async def test_last_reviewed_sha(db: Database) -> None:
    """last-reviewed SHA is stored per PR and returns None when absent."""
    sha = await get_last_reviewed_sha(db, repo_full_name="owner/repo", pr_number=2)
    assert sha is None
    await set_last_reviewed_sha(db, repo_full_name="owner/repo", pr_number=2, sha="cafebabe")
    sha = await get_last_reviewed_sha(db, repo_full_name="owner/repo", pr_number=2)
    assert sha == "cafebabe"
