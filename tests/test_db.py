"""Tests for SQLite persistence: job tracking and last-reviewed SHA."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from heimdall.db import (
    Database,
    count_recent_reviews,
    get_inflight_count,
    get_job,
    get_last_reviewed_sha,
    get_posted_review,
    prune_review_events,
    record_review_event,
    release_inflight,
    set_last_reviewed_sha,
    set_posted_review,
    try_acquire_inflight,
    try_record_review_event,
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


# ---------------------------------------------------------------------------
# Guardrail: per-repo rate/budget via review_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_recent_reviews_counts_within_window(db: Database) -> None:
    """count_recent_reviews counts only events at/after the cutoff."""
    await record_review_event(db, repo_full_name="owner/repo", occurred_at=100.0)
    await record_review_event(db, repo_full_name="owner/repo", occurred_at=200.0)
    await record_review_event(db, repo_full_name="owner/repo", occurred_at=300.0)

    # Cutoff 150 keeps the events at 200 and 300.
    assert await count_recent_reviews(db, repo_full_name="owner/repo", since=150.0) == 2
    # A cutoff before all of them keeps all three.
    assert await count_recent_reviews(db, repo_full_name="owner/repo", since=0.0) == 3


@pytest.mark.asyncio
async def test_review_events_are_per_repo(db: Database) -> None:
    """Events are scoped per repo: one repo's count never includes another's."""
    await record_review_event(db, repo_full_name="owner/a", occurred_at=10.0)
    await record_review_event(db, repo_full_name="owner/b", occurred_at=10.0)
    await record_review_event(db, repo_full_name="owner/b", occurred_at=11.0)

    assert await count_recent_reviews(db, repo_full_name="owner/a", since=0.0) == 1
    assert await count_recent_reviews(db, repo_full_name="owner/b", since=0.0) == 2


@pytest.mark.asyncio
async def test_prune_review_events_drops_old_rows(db: Database) -> None:
    """prune_review_events deletes events older than the cutoff, keeps newer ones."""
    await record_review_event(db, repo_full_name="owner/repo", occurred_at=100.0)
    await record_review_event(db, repo_full_name="owner/repo", occurred_at=300.0)

    await prune_review_events(db, repo_full_name="owner/repo", before=200.0)

    assert await count_recent_reviews(db, repo_full_name="owner/repo", since=0.0) == 1


@pytest.mark.asyncio
async def test_try_record_review_event_reserves_until_cap(db: Database) -> None:
    """try_record_review_event records a slot only while under the window budget."""
    repo = "owner/repo"
    # Cap 2 in the window [cutoff=0]: the first two reserve, the third is refused.
    assert await try_record_review_event(
        db, repo_full_name=repo, occurred_at=10.0, cutoff=0.0, max_reviews=2
    )
    assert await try_record_review_event(
        db, repo_full_name=repo, occurred_at=11.0, cutoff=0.0, max_reviews=2
    )
    assert not await try_record_review_event(
        db, repo_full_name=repo, occurred_at=12.0, cutoff=0.0, max_reviews=2
    )
    # The refused attempt recorded nothing — exactly two events exist.
    assert await count_recent_reviews(db, repo_full_name=repo, since=0.0) == 2


@pytest.mark.asyncio
async def test_try_record_review_event_ignores_events_outside_window(db: Database) -> None:
    """Events older than the cutoff don't count toward the budget."""
    repo = "owner/repo"
    # Two events outside the window (< cutoff 100) must not consume the budget.
    await record_review_event(db, repo_full_name=repo, occurred_at=10.0)
    await record_review_event(db, repo_full_name=repo, occurred_at=20.0)
    assert await try_record_review_event(
        db, repo_full_name=repo, occurred_at=150.0, cutoff=100.0, max_reviews=1
    )
    # Now one in-window event exists -> at cap -> the next is refused.
    assert not await try_record_review_event(
        db, repo_full_name=repo, occurred_at=160.0, cutoff=100.0, max_reviews=1
    )


@pytest.mark.asyncio
async def test_review_events_survive_reopen(tmp_path: Path) -> None:
    """Recorded review events survive a service restart (DB reopened from disk)."""
    db_file = str(tmp_path / "rate.db")
    first = Database(db_file)
    await first.initialize()
    await record_review_event(first, repo_full_name="owner/repo", occurred_at=42.0)
    await first.close()

    reopened = Database(db_file)
    await reopened.initialize()
    count = await count_recent_reviews(reopened, repo_full_name="owner/repo", since=0.0)
    await reopened.close()

    assert count == 1


# ---------------------------------------------------------------------------
# Guardrail: per-installation concurrency via inflight_reviews
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_inflight_increments_until_cap(db: Database) -> None:
    """Acquire succeeds up to the cap, then refuses; the count never exceeds it."""
    assert await try_acquire_inflight(db, installation_id=1, cap=2) is True
    assert await get_inflight_count(db, installation_id=1) == 1
    assert await try_acquire_inflight(db, installation_id=1, cap=2) is True
    assert await get_inflight_count(db, installation_id=1) == 2
    # At cap: the third acquire is refused and the counter stays at the cap.
    assert await try_acquire_inflight(db, installation_id=1, cap=2) is False
    assert await get_inflight_count(db, installation_id=1) == 2


@pytest.mark.asyncio
async def test_release_inflight_frees_a_slot(db: Database) -> None:
    """Releasing a slot lets a subsequent acquire succeed again."""
    await try_acquire_inflight(db, installation_id=5, cap=1)
    assert await try_acquire_inflight(db, installation_id=5, cap=1) is False
    await release_inflight(db, installation_id=5)
    assert await get_inflight_count(db, installation_id=5) == 0
    assert await try_acquire_inflight(db, installation_id=5, cap=1) is True


@pytest.mark.asyncio
async def test_release_inflight_clamps_at_zero(db: Database) -> None:
    """A release on a zero/absent counter never drives it negative."""
    await release_inflight(db, installation_id=99)
    assert await get_inflight_count(db, installation_id=99) == 0


@pytest.mark.asyncio
async def test_inflight_is_per_installation(db: Database) -> None:
    """One installation at its cap does not block a different installation."""
    assert await try_acquire_inflight(db, installation_id=1, cap=1) is True
    assert await try_acquire_inflight(db, installation_id=1, cap=1) is False
    # A different installation has its own independent counter.
    assert await try_acquire_inflight(db, installation_id=2, cap=1) is True


@pytest.mark.asyncio
async def test_concurrent_acquire_never_exceeds_cap(db: Database) -> None:
    """N concurrent acquirers against a cap admit at most ``cap`` of them.

    Drives the acquire under asyncio.gather to simulate concurrent reviews and
    asserts the DB-backed counter never overshoots the cap.
    """
    import asyncio

    cap = 3
    results = await asyncio.gather(
        *(try_acquire_inflight(db, installation_id=7, cap=cap) for _ in range(10))
    )
    assert sum(1 for granted in results if granted) == cap
    assert await get_inflight_count(db, installation_id=7) == cap


@pytest.mark.asyncio
async def test_inflight_count_survives_reopen(tmp_path: Path) -> None:
    """The in-flight counter survives a restart (it is DB-backed, not in-process)."""
    db_file = str(tmp_path / "inflight.db")
    first = Database(db_file)
    await first.initialize()
    await try_acquire_inflight(first, installation_id=3, cap=5)
    await try_acquire_inflight(first, installation_id=3, cap=5)
    await first.close()

    reopened = Database(db_file)
    await reopened.initialize()
    count = await get_inflight_count(reopened, installation_id=3)
    await reopened.close()

    assert count == 2
