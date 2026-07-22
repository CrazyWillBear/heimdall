"""Tests for the default claude subprocess invoker: kills on timeout / token cap.

The real claude binary is never spawned — bwrap resolution is patched to a fixed
path and asyncio.create_subprocess_exec is patched so we drive process behaviour
(slow completion, oversized usage) and assert the subprocess is killed and the
failure surfaces cleanly.  These tests exercise the timeout/token-cap enforcement
that runs after the sandbox wrap; the sandbox shape itself is covered in
test_lens_sandbox.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    LensTimeoutError,
    LensTokenCapError,
    parse_findings,
    run_claude_subprocess,
)


def _fake_proc(
    *,
    stdout: bytes,
    exhausts_wait: bool = False,
    returncode: int = 0,
    stderr: bytes = b"",
) -> MagicMock:
    """Build a fake asyncio subprocess.

    When ``exhausts_wait`` is True, communicate() never returns (simulating a
    hung claude run) so wait_for must time out.  ``returncode``/``stderr`` drive the
    failure-diagnostic path (a non-zero exit or an error written to stderr).

    Mirrors real subprocess semantics: ``returncode`` is ``None`` while the process
    is still running (the ``exhausts_wait`` case, until something kills it) and only
    becomes non-``None`` once it has actually exited — killing it sets it, same as
    asyncio's real process-reaping does.
    """
    proc = MagicMock()
    proc.returncode = None if exhausts_wait else returncode

    def _kill(*, _proc: MagicMock = proc) -> None:
        _proc.returncode = -9  # SIGKILL, once the process has actually exited

    proc.kill = MagicMock(side_effect=_kill)
    proc.wait = AsyncMock()

    if exhausts_wait:
        async def _never() -> tuple[bytes, bytes]:
            await asyncio.Event().wait()  # blocks forever
            return b"", b""

        proc.communicate = _never
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _claude_json(findings: list[dict[str, Any]], *, total_tokens: int) -> bytes:
    """Emulate `claude -p --output-format json` envelope on stdout."""
    envelope = {
        "type": "result",
        "result": json.dumps({"findings": findings}),
        "usage": {
            "input_tokens": total_tokens // 2,
            "output_tokens": total_tokens - total_tokens // 2,
        },
    }
    return json.dumps(envelope).encode("utf-8")


def _claude_json_stream_array(
    findings: list[dict[str, Any]], *, total_tokens: int
) -> bytes:
    """Emulate newer `claude -p --output-format json`: a JSON *array* of stream events.

    The real claude (>=2.1) emits the whole run as an array — system init, assistant
    turns, then a terminal ``type=="result"`` event carrying the final text + usage —
    rather than the single result object older runs (and the mocks above) produce.
    """
    findings_text = json.dumps({"findings": findings})
    events = [
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": findings_text}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": findings_text,
            "usage": {
                "input_tokens": total_tokens // 2,
                "output_tokens": total_tokens - total_tokens // 2,
            },
        },
    ]
    return json.dumps(events).encode("utf-8")


@pytest.mark.asyncio
async def test_subprocess_parses_stream_array_envelope() -> None:
    """A stream-array envelope is reduced to its terminal result: text + summed tokens."""
    finding = {"severity": "high", "title": "SQLi", "message": "concat"}
    stdout = _claude_json_stream_array([finding], total_tokens=2468)
    proc = _fake_proc(stdout=stdout)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    # Tokens come from the terminal result event's usage (not 0 from a missing
    # top-level usage), and the result text parses to the lens's findings.
    assert result.total_tokens == 2468
    assert result.killed is False
    parsed = parse_findings(result.stdout)
    assert [(f.title, f.severity.value) for f in parsed] == [("SQLi", "high")]


@pytest.mark.asyncio
async def test_subprocess_killed_when_stream_array_token_cap_exceeded() -> None:
    """A stream-array run reporting usage over the cap is killed (cap honoured)."""
    stdout = _claude_json_stream_array(
        [{"severity": "high", "title": "x", "message": "y"}],
        total_tokens=500_000,
    )
    proc = _fake_proc(stdout=stdout)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensTokenCapError):
        await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_subprocess_returns_parsed_result_on_success() -> None:
    """A successful run returns stdout and the summed token usage."""
    stdout = _claude_json(
        [{"severity": "low", "title": "nit", "message": "style"}],
        total_tokens=1234,
    )
    proc = _fake_proc(stdout=stdout)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    assert result.total_tokens == 1234
    assert result.killed is False
    assert "findings" in result.stdout


@pytest.mark.asyncio
async def test_subprocess_killed_on_wall_clock_timeout() -> None:
    """A run past the wall-clock timeout is killed and raises LensTimeoutError."""
    proc = _fake_proc(stdout=b"", exhausts_wait=True)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensTimeoutError):
        await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=0.05, token_cap=400_000, cwd="/srv/seed"
        )

    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_subprocess_killed_on_outer_cancellation() -> None:
    """Cancelling the awaiting task mid-communicate() still kills + reaps the child.

    This is the outer-timeout case: `_run_pipeline_with_retry` wraps the whole
    pipeline in its own `asyncio.wait_for`, so when *that* budget expires first, a
    `CancelledError` (not our `TimeoutError`) lands inside the suspended
    `communicate()` await.  Without a cleanup path keyed on cancellation the child
    is orphaned — never killed, never reaped.
    """
    proc = _fake_proc(stdout=b"", exhausts_wait=True)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        task = asyncio.ensure_future(
            run_claude_subprocess(
                ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
            )
        )
        await asyncio.sleep(0)  # let the task reach the communicate() await
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.kill.assert_called_once()
    proc.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_subprocess_killed_when_token_cap_exceeded() -> None:
    """A run reporting usage over the cap is killed and raises LensTokenCapError."""
    stdout = _claude_json(
        [{"severity": "high", "title": "x", "message": "y"}],
        total_tokens=500_000,
    )
    proc = _fake_proc(stdout=stdout)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensTokenCapError):
        await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_subprocess_logs_returncode_and_stderr_on_failed_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-zero exit / empty (0-token) run logs claude's returncode + stderr tail.

    This is the diagnostic seam: without it a failed claude run collapses to the
    generic "produced 0 tokens" guard downstream, hiding the real cause (an API
    overload/rate-limit, an oversized prompt, a crash).
    """
    proc = _fake_proc(
        stdout=b"",
        returncode=1,
        stderr=b"API Error: 529 overloaded_error",
    )

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), caplog.at_level(logging.WARNING, logger="heimdall.lens"):
        result = await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    # The run still returns (the 0-token guard fires downstream), but the cause is
    # now in the logs: the exit code and the stderr tail.
    assert result.total_tokens == 0
    assert "529 overloaded_error" in caplog.text
    assert "exit 1" in caplog.text


@pytest.mark.asyncio
async def test_subprocess_does_not_warn_on_successful_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A clean run (exit 0, tokens > 0) emits no failure-diagnostic warning."""
    stdout = _claude_json(
        [{"severity": "low", "title": "nit", "message": "style"}],
        total_tokens=1234,
    )
    proc = _fake_proc(stdout=stdout, stderr=b"some benign stderr chatter")

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), caplog.at_level(logging.WARNING, logger="heimdall.lens"):
        result = await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000, cwd="/srv/seed"
        )

    assert result.total_tokens == 1234
    assert caplog.text == ""
