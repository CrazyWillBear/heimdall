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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    LensTimeoutError,
    LensTokenCapError,
    parse_findings,
    run_claude_subprocess,
)


def _fake_proc(*, stdout: bytes, exhausts_wait: bool = False) -> MagicMock:
    """Build a fake asyncio subprocess.

    When ``exhausts_wait`` is True, communicate() never returns (simulating a
    hung claude run) so wait_for must time out.
    """
    proc = MagicMock()
    proc.returncode = 0
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    if exhausts_wait:
        async def _never() -> tuple[bytes, bytes]:
            await asyncio.Event().wait()  # blocks forever
            return b"", b""

        proc.communicate = _never
    else:
        proc.communicate = AsyncMock(return_value=(stdout, b""))
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
