"""Tests for the bubblewrap (bwrap) OS sandbox wrapping each lens subprocess.

The real ``bwrap``/``claude`` binaries are never spawned — bwrap resolution is
patched to a fixed path and ``asyncio.create_subprocess_exec`` is patched so we
capture the argv the invoker would have spawned and assert on the sandbox shape.

These tests pin the security boundary: the worker project dir is never bound into
the sandbox, the seed is bound read-only at the fixed ``/workspace`` mount, ``/tmp``
is a private tmpfs, ``~/.claude`` is read-only, PID/IPC are unshared, the network is
kept, and an unresolvable bwrap fails closed (the lens never runs unsandboxed).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from heimdall.lens import (
    SANDBOX_WORKSPACE_PATH,
    LensError,
    SandboxError,
    build_bwrap_prefix,
    build_sandbox_probe_argv,
    run_claude_subprocess,
    sandbox_exec_probe,
)

_FAKE_BWRAP = "/usr/bin/bwrap"


def test_sandbox_error_is_an_infra_fault_not_a_lens_error() -> None:
    """SandboxError is a distinct infra/deploy fault, NOT a per-lens LensError.

    Keeping it off the LensError tree is what lets the worker surface a misconfigured
    sandbox distinctly instead of swallowing it like a routine failed lens.
    """
    assert issubclass(SandboxError, Exception)
    assert not issubclass(SandboxError, LensError)


def _patch_bwrap_found() -> AbstractContextManager[object]:
    """Patch bwrap resolution so the prefix builder finds a fixed bwrap path."""
    return patch("heimdall.lens._resolve_bwrap", return_value=_FAKE_BWRAP)


def _ro_bind_targets(argv: list[str]) -> list[str]:
    """Return the host source path of every ``--ro-bind`` triple in ``argv``."""
    sources: list[str] = []
    for index, token in enumerate(argv):
        if token == "--ro-bind":
            sources.append(argv[index + 1])
    return sources


def test_prefix_binds_seed_readonly_at_fixed_workspace() -> None:
    """The seed dir is bound read-only at the fixed in-sandbox /workspace path."""
    with _patch_bwrap_found():
        prefix = build_bwrap_prefix(workspace_dir="/srv/heimdall-lens-abc")

    assert prefix[0] == _FAKE_BWRAP
    assert "--ro-bind" in prefix
    for index, token in enumerate(prefix):
        if token == "--ro-bind" and prefix[index + 1] == "/srv/heimdall-lens-abc":
            assert prefix[index + 2] == SANDBOX_WORKSPACE_PATH
            break
    else:  # pragma: no cover - defensive
        pytest.fail("seed not bound read-only at /workspace")


def test_prefix_has_tmpfs_unshare_and_shared_net() -> None:
    """The sandbox gets a private /tmp tmpfs, unshared PID/IPC, and kept network."""
    with _patch_bwrap_found():
        prefix = build_bwrap_prefix(workspace_dir="/srv/seed")

    assert "--tmpfs" in prefix
    assert prefix[prefix.index("--tmpfs") + 1] == "/tmp"
    assert "--unshare-pid" in prefix
    assert "--unshare-ipc" in prefix
    assert "--share-net" in prefix


def test_prefix_mounts_claude_home_readonly() -> None:
    """~/.claude is bound read-only into the sandbox at the same path."""
    with _patch_bwrap_found(), patch.dict(
        "os.environ", {"HOME": "/home/worker"}, clear=False
    ):
        prefix = build_bwrap_prefix(workspace_dir="/srv/seed")

    assert "/home/worker/.claude" in _ro_bind_targets(prefix)


def test_prefix_never_binds_worker_project_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The worker project dir (holding .env / heimdall.db) is never bound in."""
    project_dir = str(tmp_path)
    with _patch_bwrap_found(), patch("os.getcwd", return_value=project_dir):
        prefix = build_bwrap_prefix(workspace_dir="/srv/seed")

    sources = _ro_bind_targets(prefix)
    assert project_dir not in sources
    assert not any(src.startswith(project_dir + "/") for src in sources)


def test_extra_read_only_binds_surface_as_ro_bind_flags() -> None:
    """SANDBOX_EXTRA_READ_ONLY_BINDS entries surface as --ro-bind flags."""
    extras = ["/opt/node", "/etc/custom-ca"]
    with _patch_bwrap_found():
        prefix = build_bwrap_prefix(
            workspace_dir="/srv/seed", extra_read_only_binds=extras
        )

    sources = _ro_bind_targets(prefix)
    for extra in extras:
        assert extra in sources


def test_prefix_fails_closed_when_bwrap_unavailable() -> None:
    """An unresolvable bwrap raises SandboxError so the lens cannot run unsandboxed."""
    with patch("heimdall.lens._resolve_bwrap", return_value=None), pytest.raises(
        SandboxError
    ):
        build_bwrap_prefix(workspace_dir="/srv/seed")


@pytest.mark.asyncio
async def test_subprocess_spawns_bwrap_prefixed_argv() -> None:
    """run_claude_subprocess spawns bwrap-prefixed argv with the seed-only bind set."""
    captured: dict[str, object] = {}

    async def _fake_exec(*argv: str, **kwargs: object) -> MagicMock:
        captured["argv"] = list(argv)
        captured["cwd"] = kwargs.get("cwd")
        proc = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b'{"result": "{}"}', b""))
        return proc

    claude_argv = ["/real/claude", "-p", "review", "--add-dir", SANDBOX_WORKSPACE_PATH]
    with _patch_bwrap_found(), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=_fake_exec
    ):
        await run_claude_subprocess(
            claude_argv,
            timeout_seconds=900,
            token_cap=400_000,
            cwd="/srv/heimdall-lens-xyz",
        )

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[0] == _FAKE_BWRAP
    assert argv[-len(claude_argv):] == claude_argv
    assert "/srv/heimdall-lens-xyz" in _ro_bind_targets(argv)
    # Host cwd for the bwrap process must be the real seed dir (it exists on the host).
    # The in-sandbox cwd is /workspace, set by bwrap's --chdir flag (asserted below) —
    # passing the in-sandbox path as the host cwd would fail every spawn with ENOENT.
    assert captured["cwd"] == "/srv/heimdall-lens-xyz"
    assert argv[argv.index("--chdir") + 1] == SANDBOX_WORKSPACE_PATH


@pytest.mark.asyncio
async def test_subprocess_fails_closed_when_bwrap_unavailable() -> None:
    """When bwrap cannot be resolved the invoker raises and never spawns claude."""
    exec_mock = AsyncMock()
    with patch("heimdall.lens._resolve_bwrap", return_value=None), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=exec_mock
    ), pytest.raises(SandboxError):
        await run_claude_subprocess(
            ["/real/claude", "-p"],
            timeout_seconds=900,
            token_cap=400_000,
            cwd="/srv/seed",
        )

    exec_mock.assert_not_called()


@pytest.mark.asyncio
async def test_subprocess_fails_closed_on_empty_cwd() -> None:
    """A falsy/empty cwd fails closed before any spawn (defence-in-depth guard).

    cwd is required by the type, but an empty string would bind nothing read-only at
    /workspace and silently widen the sandbox, so the invoker rejects it rather than
    spawn an effectively unconfined lens.
    """
    exec_mock = AsyncMock()
    with _patch_bwrap_found(), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=exec_mock
    ), pytest.raises(SandboxError):
        await run_claude_subprocess(
            ["/real/claude", "-p"],
            timeout_seconds=900,
            token_cap=400_000,
            cwd="",
        )

    exec_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Startup sandbox exec-probe (issue #29)
# ---------------------------------------------------------------------------


def test_probe_argv_unshares_and_shares_net() -> None:
    """The probe exercises the same unshare/share-net machinery a lens wrap uses."""
    with _patch_bwrap_found():
        argv = build_sandbox_probe_argv()

    assert argv[0] == _FAKE_BWRAP
    assert argv[1:4] == ["--ro-bind", "/", "/"]
    assert "--unshare-all" in argv
    assert "--share-net" in argv
    # Everything after the -- separator is the command run inside the sandbox.
    assert argv[argv.index("--") + 1:] == ["true"]


def test_probe_argv_fails_closed_when_bwrap_unavailable() -> None:
    """An unresolvable bwrap raises SandboxError before any probe is spawned."""
    with patch("heimdall.lens._resolve_bwrap", return_value=None), pytest.raises(
        SandboxError
    ):
        build_sandbox_probe_argv()


def _fake_probe_proc(returncode: int, stderr: bytes = b"") -> MagicMock:
    """Build a fake subprocess whose communicate() yields the given exit/stderr."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return proc


@pytest.mark.asyncio
async def test_exec_probe_passes_on_zero_exit() -> None:
    """A probe that exits 0 returns without raising."""
    proc = _fake_probe_proc(returncode=0)
    with _patch_bwrap_found(), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
    ):
        await sandbox_exec_probe()  # must not raise


@pytest.mark.asyncio
async def test_exec_probe_aborts_on_nonzero_exit() -> None:
    """A probe that exits non-zero raises SandboxError so the worker refuses to boot."""
    proc = _fake_probe_proc(returncode=1, stderr=b"bwrap: setting up uid map failed")
    with _patch_bwrap_found(), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
    ), pytest.raises(SandboxError):
        await sandbox_exec_probe()


@pytest.mark.asyncio
async def test_exec_probe_fails_closed_when_bwrap_unavailable() -> None:
    """A missing bwrap raises SandboxError and never spawns a probe."""
    exec_mock = AsyncMock()
    with patch("heimdall.lens._resolve_bwrap", return_value=None), patch(
        "heimdall.lens.asyncio.create_subprocess_exec", new=exec_mock
    ), pytest.raises(SandboxError):
        await sandbox_exec_probe()

    exec_mock.assert_not_called()
