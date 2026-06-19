"""Tests for the default claude subprocess invoker: kills on timeout / token cap.

The real claude binary is never spawned — asyncio.create_subprocess_exec is
patched so we drive process behaviour (slow completion, oversized usage) and
assert the subprocess is killed and the failure surfaces cleanly.
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


@pytest.mark.asyncio
async def test_subprocess_returns_parsed_result_on_success() -> None:
    """A successful run returns stdout and the summed token usage."""
    stdout = _claude_json(
        [{"severity": "low", "title": "nit", "message": "style"}],
        total_tokens=1234,
    )
    proc = _fake_proc(stdout=stdout)

    with patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ):
        result = await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000
        )

    assert result.total_tokens == 1234
    assert result.killed is False
    assert "findings" in result.stdout


@pytest.mark.asyncio
async def test_subprocess_killed_on_wall_clock_timeout() -> None:
    """A run past the wall-clock timeout is killed and raises LensTimeoutError."""
    proc = _fake_proc(stdout=b"", exhausts_wait=True)

    with patch(
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensTimeoutError):
        await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=0.05, token_cap=400_000
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
        "heimdall.lens.asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=proc),
    ), pytest.raises(LensTokenCapError):
        await run_claude_subprocess(
            ["claude", "-p"], timeout_seconds=900, token_cap=400_000
        )

    proc.kill.assert_called_once()
