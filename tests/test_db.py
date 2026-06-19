"""Tests for SQLite persistence: job tracking and last-reviewed SHA."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from heimdall.db import (
    Database,
    get_job,
    get_last_reviewed_sha,
    get_posted_review,
    set_last_reviewed_sha,
    set_posted_review,
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


@pytest.mark.asyncio
async def test_posted_review_absent_returns_none(db: Database) -> None:
    """get_posted_review returns None for a PR that has never been reviewed."""
    review = await get_posted_review(db, repo_full_name="owner/repo", pr_number=9)
    assert review is None


@pytest.mark.asyncio
async def test_set_and_get_posted_review(db: Database) -> None:
    """The prior review id, node id, and verdict are stored and read back per PR."""
    await set_posted_review(
        db,
        repo_full_name="owner/repo",
        pr_number=3,
        review_id=100,
        node_id="MDA=",
        verdict="REQUEST_CHANGES",
    )
    review = await get_posted_review(db, repo_full_name="owner/repo", pr_number=3)
    assert review is not None
    assert review["review_id"] == 100
    assert review["node_id"] == "MDA="
    assert review["verdict"] == "REQUEST_CHANGES"


@pytest.mark.asyncio
async def test_set_posted_review_overwrites_prior(db: Database) -> None:
    """A second post overwrites the stored review so only the latest remains."""
    await set_posted_review(
        db,
        repo_full_name="owner/repo",
        pr_number=4,
        review_id=1,
        node_id="A",
        verdict="COMMENT",
    )
    await set_posted_review(
        db,
        repo_full_name="owner/repo",
        pr_number=4,
        review_id=2,
        node_id="B",
        verdict="REQUEST_CHANGES",
    )
    review = await get_posted_review(db, repo_full_name="owner/repo", pr_number=4)
    assert review is not None
    assert review["review_id"] == 2
    assert review["node_id"] == "B"
    assert review["verdict"] == "REQUEST_CHANGES"


@pytest.mark.asyncio
async def test_posted_review_survives_reopen(tmp_path: Path) -> None:
    """The stored review id survives a service restart (DB reopened from disk)."""
    db_file = str(tmp_path / "heimdall-test.db")
    first = Database(db_file)
    await first.initialize()
    await set_posted_review(
        first,
        repo_full_name="owner/repo",
        pr_number=7,
        review_id=555,
        node_id="NODE7",
        verdict="COMMENT",
    )
    await first.close()

    reopened = Database(db_file)
    await reopened.initialize()
    review = await get_posted_review(reopened, repo_full_name="owner/repo", pr_number=7)
    await reopened.close()

    assert review is not None
    assert review["review_id"] == 555
    assert review["node_id"] == "NODE7"
    assert review["verdict"] == "COMMENT"
