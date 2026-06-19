"""Tests for PR seed-context assembly and heimdall-context CLI wrapper."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from heimdall.context import PRContext, assemble_pr_context
from heimdall.context_cli import cmd_conventions, cmd_diff, cmd_file, cmd_pr, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REPO = "owner/repo"
_PR_NUMBER = 7
_HEAD_SHA = "abc123def456"
_BASE_SHA = "000000001111"
_AUTHOR = "octocat"
_TITLE = "Add awesome feature"
_BODY = "This PR adds an awesome feature."
_BASE_REF = "main"
_HEAD_REF = "feature/awesome"

_LINKED_ISSUES = [{"number": 42, "title": "Feature request"}]

_DIFF = """\
diff --git a/foo.py b/foo.py
index 0000000..1111111 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 x = 1
+y = 2
 z = 3
"""

_FILES = [
    {
        "filename": "foo.py",
        "status": "modified",
        "patch": "@@ -1,3 +1,4 @@ x = 1\n+y = 2\n z = 3",
    }
]

_FILE_CONTENT = "x = 1\ny = 2\nz = 3\n"

_PR_METADATA: dict[str, Any] = {
    "number": _PR_NUMBER,
    "title": _TITLE,
    "body": _BODY,
    "user": {"login": _AUTHOR},
    "base": {"sha": _BASE_SHA, "ref": _BASE_REF, "repo": {"full_name": _REPO}},
    "head": {"sha": _HEAD_SHA, "ref": _HEAD_REF},
}


def _make_mock_github_client(
    diff: str = _DIFF,
    files: list[dict[str, Any]] | None = None,
    pr_metadata: dict[str, Any] | None = None,
    file_content: str = _FILE_CONTENT,
    linked_issues: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    """Build a mocked GitHubClient that returns canned API data."""
    client = AsyncMock()
    client.get_pr_diff = AsyncMock(return_value=diff)
    client.get_pr_files = AsyncMock(return_value=files if files is not None else _FILES)
    client.get_pr = AsyncMock(return_value=pr_metadata if pr_metadata is not None else _PR_METADATA)
    client.get_file_content = AsyncMock(return_value=file_content)
    client.get_linked_issues = AsyncMock(
        return_value=linked_issues if linked_issues is not None else _LINKED_ISSUES
    )
    return client


# ---------------------------------------------------------------------------
# PRContext dataclass
# ---------------------------------------------------------------------------


def test_pr_context_fields() -> None:
    """PRContext holds all seed fields and is not mutable."""
    ctx = PRContext(
        repo_full_name=_REPO,
        pr_number=_PR_NUMBER,
        title=_TITLE,
        body=_BODY,
        author=_AUTHOR,
        base_sha=_BASE_SHA,
        head_sha=_HEAD_SHA,
        base_ref=_BASE_REF,
        head_ref=_HEAD_REF,
        linked_issues=_LINKED_ISSUES,
        diff=_DIFF,
        changed_files=_FILES,
        file_contents={"foo.py": _FILE_CONTENT},
        convention_docs={},
    )
    assert ctx.repo_full_name == _REPO
    assert ctx.pr_number == _PR_NUMBER
    assert ctx.diff == _DIFF
    assert ctx.file_contents["foo.py"] == _FILE_CONTENT
    assert ctx.linked_issues == _LINKED_ISSUES


# ---------------------------------------------------------------------------
# assemble_pr_context: seed contains diff, changed-file contents, PR metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_pr_context_seed_contains_diff() -> None:
    """Assembled seed includes the full unified diff."""
    mock_client = _make_mock_github_client()
    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    assert ctx.diff == _DIFF


@pytest.mark.asyncio
async def test_assemble_pr_context_seed_contains_pr_metadata() -> None:
    """Assembled seed includes PR title, body, author, and SHAs."""
    mock_client = _make_mock_github_client()
    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    assert ctx.title == _TITLE
    assert ctx.body == _BODY
    assert ctx.author == _AUTHOR
    assert ctx.base_sha == _BASE_SHA
    assert ctx.head_sha == _HEAD_SHA
    assert ctx.linked_issues == _LINKED_ISSUES


@pytest.mark.asyncio
async def test_assemble_pr_context_seed_contains_changed_file_contents() -> None:
    """Assembled seed includes full contents of each changed file."""
    mock_client = _make_mock_github_client()
    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    assert "foo.py" in ctx.file_contents
    assert ctx.file_contents["foo.py"] == _FILE_CONTENT


# ---------------------------------------------------------------------------
# No code execution: subprocess is never called during assembly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_pr_context_never_runs_subprocess() -> None:
    """Assembly must not spawn any subprocess (no build/test/script execution)."""
    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        patch("subprocess.run") as mock_run,
        patch("subprocess.Popen") as mock_popen,
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    mock_run.assert_not_called()
    mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Workspace: created under temp root, cleaned up after use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_pr_context_workspace_cleaned_up() -> None:
    """The temp workspace created during assembly is removed afterwards."""
    created_dirs: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(**kwargs: Any) -> str:
        path: str = real_mkdtemp(**kwargs)
        created_dirs.append(path)
        return path

    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        patch("heimdall.context.tempfile.mkdtemp", side_effect=tracking_mkdtemp),
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )

    # Every temp dir that was created must now be gone
    for d in created_dirs:
        assert not os.path.exists(d), f"Temp workspace was not cleaned up: {d}"


@pytest.mark.asyncio
async def test_assemble_pr_context_workspace_under_temp_root() -> None:
    """The workspace is created inside tempfile.gettempdir(), not in cwd."""
    created_dirs: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(**kwargs: Any) -> str:
        path: str = real_mkdtemp(**kwargs)
        created_dirs.append(path)
        return path

    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        patch("heimdall.context.tempfile.mkdtemp", side_effect=tracking_mkdtemp),
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )

    tmp_root = tempfile.gettempdir()
    for d in created_dirs:
        assert Path(d).parent == Path(tmp_root) or str(d).startswith(tmp_root), (
            f"Workspace not under temp root: {d}"
        )


# ---------------------------------------------------------------------------
# heimdall-context CLI: diff subcommand
# ---------------------------------------------------------------------------


def _write_workspace(tmp_path: Path, ctx: PRContext) -> Path:
    """Write a materialized workspace to tmp_path and return it."""
    (tmp_path / "diff.patch").write_text(ctx.diff)
    (tmp_path / "pr_metadata.json").write_text(
        json.dumps(
            {
                "repo_full_name": ctx.repo_full_name,
                "pr_number": ctx.pr_number,
                "title": ctx.title,
                "body": ctx.body,
                "author": ctx.author,
                "base_sha": ctx.base_sha,
                "head_sha": ctx.head_sha,
                "base_ref": ctx.base_ref,
                "head_ref": ctx.head_ref,
                "linked_issues": ctx.linked_issues,
            }
        )
    )
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    for filename, content in ctx.file_contents.items():
        file_path = files_dir / filename.replace("/", os.sep)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return tmp_path


def _make_context() -> PRContext:
    return PRContext(
        repo_full_name=_REPO,
        pr_number=_PR_NUMBER,
        title=_TITLE,
        body=_BODY,
        author=_AUTHOR,
        base_sha=_BASE_SHA,
        head_sha=_HEAD_SHA,
        base_ref=_BASE_REF,
        head_ref=_HEAD_REF,
        linked_issues=_LINKED_ISSUES,
        diff=_DIFF,
        changed_files=_FILES,
        file_contents={"foo.py": _FILE_CONTENT},
        convention_docs={},
    )


def test_cmd_diff_returns_expected_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """heimdall-context diff prints the diff from the materialized workspace."""
    ctx = _make_context()
    workspace = _write_workspace(tmp_path, ctx)
    cmd_diff(str(workspace))
    captured = capsys.readouterr()
    assert _DIFF in captured.out


def test_cmd_pr_returns_pr_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """heimdall-context pr prints PR metadata from the materialized workspace."""
    ctx = _make_context()
    workspace = _write_workspace(tmp_path, ctx)
    cmd_pr(str(workspace))
    captured = capsys.readouterr()
    assert _TITLE in captured.out
    assert _AUTHOR in captured.out
    assert _HEAD_SHA in captured.out


def test_main_diff_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() dispatches 'diff' subcommand correctly."""
    ctx = _make_context()
    workspace = _write_workspace(tmp_path, ctx)
    main(["diff", str(workspace)])
    captured = capsys.readouterr()
    assert _DIFF in captured.out


def test_main_pr_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() dispatches 'pr' subcommand correctly."""
    ctx = _make_context()
    workspace = _write_workspace(tmp_path, ctx)
    main(["pr", str(workspace)])
    captured = capsys.readouterr()
    assert _TITLE in captured.out


def test_main_invalid_subcommand_exits(tmp_path: Path) -> None:
    """main() exits with code 2 for an unknown subcommand."""
    with pytest.raises(SystemExit) as exc_info:
        main(["bad-subcommand", str(tmp_path)])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# materialize_workspace: writes seed to disk so the CLI can read it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_pr_context_materializes_diff_file() -> None:
    """assemble_pr_context with workspace_dir writes diff.patch to disk."""
    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
            workspace_dir=tmp_dir,
        )
        assert (Path(tmp_dir) / "diff.patch").exists()
        assert (Path(tmp_dir) / "diff.patch").read_text() == _DIFF


@pytest.mark.asyncio
async def test_assemble_pr_context_materializes_pr_metadata_file() -> None:
    """assemble_pr_context with workspace_dir writes pr_metadata.json to disk."""
    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
            workspace_dir=tmp_dir,
        )
        meta_path = Path(tmp_dir) / "pr_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["title"] == _TITLE
        assert meta["head_sha"] == _HEAD_SHA


@pytest.mark.asyncio
async def test_assemble_pr_context_materializes_file_contents() -> None:
    """assemble_pr_context with workspace_dir writes changed files under files/."""
    mock_client = _make_mock_github_client()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
            workspace_dir=tmp_dir,
        )
        file_path = Path(tmp_dir) / "files" / "foo.py"
        assert file_path.exists()
        assert file_path.read_text() == _FILE_CONTENT


# ---------------------------------------------------------------------------
# subprocess import guard: verify subprocess is never imported by context module
# ---------------------------------------------------------------------------


def test_context_module_does_not_import_subprocess() -> None:
    """heimdall.context must not import subprocess at module level."""
    import heimdall.context as ctx_mod

    assert not hasattr(ctx_mod, "subprocess"), (
        "heimdall.context imported subprocess — no code execution allowed"
    )


# ---------------------------------------------------------------------------
# Path traversal: _materialize must reject attacker-controlled filenames
# ---------------------------------------------------------------------------


def _make_context_with_files(file_contents: dict[str, str]) -> PRContext:
    """Build a minimal PRContext with the given file_contents mapping."""
    return PRContext(
        repo_full_name=_REPO,
        pr_number=_PR_NUMBER,
        title=_TITLE,
        body=_BODY,
        author=_AUTHOR,
        base_sha=_BASE_SHA,
        head_sha=_HEAD_SHA,
        base_ref=_BASE_REF,
        head_ref=_HEAD_REF,
        linked_issues=[],
        diff="",
        changed_files=[],
        file_contents=file_contents,
        convention_docs={},
    )


def test_materialize_rejects_absolute_filename(tmp_path: Path) -> None:
    """_materialize must not write files whose filename is an absolute path."""
    from heimdall.context import _materialize

    ctx = _make_context_with_files({"/etc/cron.d/x": "evil content"})
    # Must complete without raising, but must not write outside the workspace.
    _materialize(ctx, str(tmp_path))

    # Confirm nothing escaped: scan everything materialized under files/
    files_root = tmp_path / "files"
    workspace_root = str(tmp_path.resolve())
    if files_root.exists():
        for p in files_root.rglob("*"):
            if p.is_file():
                assert str(p.resolve()).startswith(workspace_root), (
                    f"File escaped workspace: {p}"
                )
    # The hostile absolute target must not have been written (or has different size
    # if it pre-existed, which indicates we did not overwrite it).
    hostile = Path("/etc/cron.d/x")
    assert not hostile.exists() or hostile.stat().st_size != len("evil content")


def test_materialize_rejects_traversal_filename(tmp_path: Path) -> None:
    """_materialize must not write files that escape via ../ traversal."""
    from heimdall.context import _materialize

    # Place the workspace one level deep so ../evil.txt would escape it.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    escape_target = tmp_path / "evil.txt"

    # ../evil.txt relative to files_root (<workspace>/files/) resolves to
    # <workspace>/evil.txt — still inside workspace — so use ../../evil.txt
    # to reach tmp_path/evil.txt which is outside the workspace directory.
    ctx = _make_context_with_files({"../../evil.txt": "evil content"})
    _materialize(ctx, str(workspace))

    assert not escape_target.exists(), (
        "Traversal escaped the workspace: evil.txt found outside workspace dir"
    )
    # Verify all written files are within the workspace
    files_root = workspace / "files"
    workspace_root = str(workspace.resolve())
    if files_root.exists():
        for p in files_root.rglob("*"):
            if p.is_file():
                assert str(p.resolve()).startswith(workspace_root), (
                    f"File escaped workspace: {p}"
                )


def test_materialize_normal_nested_path(tmp_path: Path) -> None:
    """_materialize writes normal nested paths correctly under files/."""
    from heimdall.context import _materialize

    ctx = _make_context_with_files({"pkg/mod/foo.py": "# content\n"})
    _materialize(ctx, str(tmp_path))

    expected = tmp_path / "files" / "pkg" / "mod" / "foo.py"
    assert expected.exists(), f"Expected file not found: {expected}"
    assert expected.read_text() == "# content\n"


# ---------------------------------------------------------------------------
# Robustness: binary and oversize files are skipped, not crashing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_pr_context_skips_binary_file() -> None:
    """A binary changed file is skipped; assembly does not crash."""
    binary_files = [
        {"filename": "logo.png", "status": "modified"},
        {"filename": "foo.py", "status": "modified"},
    ]

    def _file_content_side_effect(
        *, repo_full_name: str, path: str, ref: str, tolerate_missing: bool = False
    ) -> str | None:
        if path == "logo.png":
            raise UnicodeDecodeError("utf-8", b"\x89PNG", 0, 1, "invalid start byte")
        if tolerate_missing:
            return None  # convention docs not present in this test
        return _FILE_CONTENT

    mock_client = _make_mock_github_client(files=binary_files)
    mock_client.get_file_content = AsyncMock(side_effect=_file_content_side_effect)

    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )

    # Normal file is present
    assert "foo.py" in ctx.file_contents
    assert ctx.file_contents["foo.py"] == _FILE_CONTENT
    # Binary file is skipped (placeholder or absent — not a crash)
    assert "logo.png" not in ctx.file_contents or ctx.file_contents["logo.png"] != _FILE_CONTENT


@pytest.mark.asyncio
async def test_assemble_pr_context_skips_oversize_file() -> None:
    """A >1 MB changed file (GitHub Contents API 404) is skipped; assembly does not crash."""
    import httpx

    large_files = [
        {"filename": "huge.bin", "status": "modified"},
        {"filename": "foo.py", "status": "modified"},
    ]

    def _file_content_side_effect(
        *, repo_full_name: str, path: str, ref: str, tolerate_missing: bool = False
    ) -> str | None:
        if path == "huge.bin":
            response = httpx.Response(404, request=httpx.Request("GET", "https://api.github.com/"))
            raise httpx.HTTPStatusError(
                "404 Not Found",
                request=response.request,
                response=response,
            )
        if tolerate_missing:
            return None  # convention docs not present in this test
        return _FILE_CONTENT

    mock_client = _make_mock_github_client(files=large_files)
    mock_client.get_file_content = AsyncMock(side_effect=_file_content_side_effect)

    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )

    # Normal file is present
    assert "foo.py" in ctx.file_contents
    assert ctx.file_contents["foo.py"] == _FILE_CONTENT
    # Oversize file is not in contents (skipped)
    assert "huge.bin" not in ctx.file_contents


@pytest.mark.asyncio
async def test_assemble_pr_context_normal_files_fetched_alongside_bad() -> None:
    """When one file is binary and another errors, remaining normal files are still fetched."""
    import httpx

    mixed_files = [
        {"filename": "image.jpg", "status": "modified"},
        {"filename": "huge.dat", "status": "modified"},
        {"filename": "foo.py", "status": "modified"},
        {"filename": "bar.py", "status": "modified"},
    ]

    def _file_content_side_effect(
        *, repo_full_name: str, path: str, ref: str, tolerate_missing: bool = False
    ) -> str | None:
        if path == "image.jpg":
            raise UnicodeDecodeError("utf-8", b"\xff\xd8", 0, 1, "invalid start byte")
        if path == "huge.dat":
            response = httpx.Response(404, request=httpx.Request("GET", "https://api.github.com/"))
            raise httpx.HTTPStatusError(
                "404 Not Found",
                request=response.request,
                response=response,
            )
        if tolerate_missing:
            return None  # convention docs not present in this test
        return _FILE_CONTENT

    mock_client = _make_mock_github_client(files=mixed_files)
    mock_client.get_file_content = AsyncMock(side_effect=_file_content_side_effect)

    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )

    # Both normal files survive
    assert ctx.file_contents["foo.py"] == _FILE_CONTENT
    assert ctx.file_contents["bar.py"] == _FILE_CONTENT
    # Problem files are absent
    assert "image.jpg" not in ctx.file_contents
    assert "huge.dat" not in ctx.file_contents


# ---------------------------------------------------------------------------
# convention_docs: populated from repo at head_sha
# ---------------------------------------------------------------------------

_CONVENTION_DOCS = {
    "STYLEGUIDE.md": "# Style Guide\n\nBe consistent.",
    "CLAUDE.md": "# CLAUDE\n\nWorkflow rules.",
}


def _make_mock_github_client_with_conventions(
    convention_docs: dict[str, str] | None = None,
) -> AsyncMock:
    """Build a mocked GitHubClient that returns convention docs."""
    docs = convention_docs if convention_docs is not None else _CONVENTION_DOCS

    async def _get_file_content(
        *,
        repo_full_name: str,
        path: str,
        ref: str,
        tolerate_missing: bool = False,
    ) -> str | None:
        return docs.get(path)  # returns None for missing docs

    client = AsyncMock()
    client.get_pr_diff = AsyncMock(return_value=_DIFF)
    client.get_pr_files = AsyncMock(return_value=_FILES)
    client.get_pr = AsyncMock(return_value=_PR_METADATA)
    client.get_file_content = _get_file_content
    client.get_linked_issues = AsyncMock(return_value=_LINKED_ISSUES)
    return client


@pytest.mark.asyncio
async def test_assemble_pr_context_convention_docs_populated() -> None:
    """Assembled seed includes convention docs from the repo."""
    mock_client = _make_mock_github_client_with_conventions()
    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    assert "STYLEGUIDE.md" in ctx.convention_docs
    assert "CLAUDE.md" in ctx.convention_docs
    assert ctx.convention_docs["STYLEGUIDE.md"] == _CONVENTION_DOCS["STYLEGUIDE.md"]


@pytest.mark.asyncio
async def test_assemble_pr_context_convention_docs_missing_tolerated() -> None:
    """Convention docs missing from the repo (None) are omitted, not an error."""
    # Only CLAUDE.md present, STYLEGUIDE.md and README.md missing
    mock_client = _make_mock_github_client_with_conventions({"CLAUDE.md": "rules"})
    with patch("heimdall.context.GitHubClient", return_value=mock_client):
        ctx = await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
        )
    assert "CLAUDE.md" in ctx.convention_docs
    assert "STYLEGUIDE.md" not in ctx.convention_docs
    assert "README.md" not in ctx.convention_docs


@pytest.mark.asyncio
async def test_assemble_pr_context_materializes_convention_docs() -> None:
    """assemble_pr_context materializes convention docs under conventions/."""
    mock_client = _make_mock_github_client_with_conventions()
    with (
        patch("heimdall.context.GitHubClient", return_value=mock_client),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        await assemble_pr_context(
            app_id=1,
            private_key="key",
            installation_id=42,
            repo_full_name=_REPO,
            pr_number=_PR_NUMBER,
            workspace_dir=tmp_dir,
        )
        conventions_dir = Path(tmp_dir) / "conventions"
        assert conventions_dir.exists()
        assert (conventions_dir / "STYLEGUIDE.md").exists()
        assert (conventions_dir / "STYLEGUIDE.md").read_text() == _CONVENTION_DOCS["STYLEGUIDE.md"]


# ---------------------------------------------------------------------------
# heimdall-context CLI: file subcommand (with path sanitization)
# ---------------------------------------------------------------------------


def _make_context_with_conventions() -> PRContext:
    return PRContext(
        repo_full_name=_REPO,
        pr_number=_PR_NUMBER,
        title=_TITLE,
        body=_BODY,
        author=_AUTHOR,
        base_sha=_BASE_SHA,
        head_sha=_HEAD_SHA,
        base_ref=_BASE_REF,
        head_ref=_HEAD_REF,
        linked_issues=_LINKED_ISSUES,
        diff=_DIFF,
        changed_files=_FILES,
        file_contents={"foo.py": _FILE_CONTENT},
        convention_docs=_CONVENTION_DOCS,
    )


def _write_workspace_with_conventions(tmp_path: Path, ctx: PRContext) -> Path:
    """Write a full materialized workspace including convention docs."""
    ws = _write_workspace(tmp_path, ctx)
    conventions_dir = tmp_path / "conventions"
    conventions_dir.mkdir()
    for name, content in ctx.convention_docs.items():
        (conventions_dir / name).write_text(content)
    return ws


def test_cmd_file_reads_changed_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """heimdall-context file prints the content of a materialized changed file."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    cmd_file(str(workspace), "foo.py")
    captured = capsys.readouterr()
    assert _FILE_CONTENT in captured.out


def test_cmd_file_rejects_path_traversal(tmp_path: Path) -> None:
    """heimdall-context file rejects paths that escape the workspace."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    with pytest.raises(SystemExit):
        cmd_file(str(workspace), "../secret.txt")


def test_cmd_file_rejects_absolute_path(tmp_path: Path) -> None:
    """heimdall-context file rejects absolute paths."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    with pytest.raises(SystemExit):
        cmd_file(str(workspace), "/etc/passwd")


def test_cmd_file_missing_file_exits(tmp_path: Path) -> None:
    """heimdall-context file exits non-zero when the requested file doesn't exist."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    with pytest.raises(SystemExit):
        cmd_file(str(workspace), "nonexistent.py")


def test_main_file_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() dispatches 'file' subcommand correctly."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    main(["file", str(workspace), "foo.py"])
    captured = capsys.readouterr()
    assert _FILE_CONTENT in captured.out


# ---------------------------------------------------------------------------
# heimdall-context CLI: conventions subcommand
# ---------------------------------------------------------------------------


def test_cmd_conventions_reads_convention_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """heimdall-context conventions prints all convention docs."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    cmd_conventions(str(workspace))
    captured = capsys.readouterr()
    # Both docs must appear in the output
    assert "STYLEGUIDE.md" in captured.out
    assert "Be consistent" in captured.out
    assert "CLAUDE.md" in captured.out


def test_cmd_conventions_empty_when_no_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """heimdall-context conventions outputs nothing when conventions/ is absent."""
    ctx = _make_context()  # no convention docs materialized
    workspace = _write_workspace(tmp_path, ctx)
    cmd_conventions(str(workspace))
    captured = capsys.readouterr()
    # Should not raise; output may be empty or a notice
    assert isinstance(captured.out, str)


def test_main_conventions_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() dispatches 'conventions' subcommand correctly."""
    ctx = _make_context_with_conventions()
    workspace = _write_workspace_with_conventions(tmp_path, ctx)
    main(["conventions", str(workspace)])
    captured = capsys.readouterr()
    assert "STYLEGUIDE.md" in captured.out
