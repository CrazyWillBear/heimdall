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
    LensOutputCapError,
    LensTimeoutError,
    LensTokenCapError,
    parse_findings,
    run_claude_subprocess,
)


class _FakeStreamReader:
    """Minimal stand-in for ``asyncio.StreamReader``: chunked ``read()`` + EOF.

    ``run_claude_subprocess`` reads stdout/stderr incrementally (not via
    ``communicate()``) so the byte cap can be enforced mid-read; this fake feeds
    the given payload back in fixed-size chunks, or blocks forever when
    ``hangs`` is set (simulating a stalled/never-closing pipe).
    """

    def __init__(self, data: bytes, *, chunk_size: int = 4096, hangs: bool = False) -> None:
        self._data = data
        self._offset = 0
        self._chunk_size = chunk_size
        self._hangs = hangs

    async def read(self, n: int = -1) -> bytes:
        if self._hangs and self._offset >= len(self._data):
            await asyncio.Event().wait()  # blocks forever past the fed payload
        size = self._chunk_size if n < 0 else min(n, self._chunk_size)
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _fake_proc(
    *,
    stdout: bytes,
    exhausts_wait: bool = False,
    returncode: int = 0,
    stderr: bytes = b"",
) -> MagicMock:
    """Build a fake asyncio subprocess with streaming stdout/stderr readers.

    When ``exhausts_wait`` is True, stdout never reaches EOF (simulating a hung
    claude run) so wait_for must time out.  ``returncode``/``stderr`` drive the
    failure-diagnostic path (a non-zero exit or an error written to stderr).
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.stdout = _FakeStreamReader(stdout, hangs=exhausts_wait)
    proc.stderr = _FakeStreamReader(stderr)
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
async def test_subprocess_killed_when_stdout_exceeds_byte_cap() -> None:
    """A child streaming past the byte ceiling is killed, not buffered unbounded.

    Regression for the runaway-transcript risk: proc.communicate() would hold the
    child's entire stdout in memory before the token cap is ever checked.  Feed a
    stream that hangs after emitting well past a tiny cap; the process must be
    killed as soon as the ceiling is crossed rather than blocking forever waiting
    for EOF.
    """
    stdout = b"x" * 10_000
    proc = _fake_proc(stdout=stdout, exhausts_wait=True)

    with patch(
        "heimdall.lens._resolve_bwrap", return_value="/usr/bin/bwrap"
    ), patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensOutputCapError):
        await run_claude_subprocess(
            ["claude", "-p"],
            timeout_seconds=900,
            token_cap=400_000,
            cwd="/srv/seed",
            output_byte_cap=1_000,
        )

    proc.kill.assert_called_once()


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
