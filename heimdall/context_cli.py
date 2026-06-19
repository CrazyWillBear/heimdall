"""heimdall-context — CLI wrapper for reading materialized PR seed context.

Provides subcommands:
  heimdall-context diff  <workspace>            — print the unified diff
  heimdall-context pr    <workspace>            — print PR metadata as JSON
  heimdall-context file  <workspace> <path>     — print a materialized file's content
  heimdall-context conventions <workspace>      — print all repo convention docs

The workspace must be a directory previously produced by assemble_pr_context()
(i.e. it contains diff.patch, pr_metadata.json, files/, and optionally conventions/).

This wrapper is the ONLY allowlisted Bash command used during AI-driven lens review
sessions — it reads from pre-materialized data and executes nothing.  The ``file``
subcommand sanitizes the requested path so it cannot escape the workspace root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_diff(workspace: str) -> None:
    """Print the unified diff from a materialized workspace.

    Args:
        workspace: Path to the directory written by assemble_pr_context().
    """
    diff_path = Path(workspace) / "diff.patch"
    print(diff_path.read_text(encoding="utf-8"))


def cmd_pr(workspace: str) -> None:
    """Print formatted PR metadata from a materialized workspace.

    Args:
        workspace: Path to the directory written by assemble_pr_context().
    """
    meta_path = Path(workspace) / "pr_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(json.dumps(meta, indent=2))


def cmd_file(workspace: str, path: str) -> None:
    """Print the content of a materialized changed file from the workspace.

    The requested path is resolved inside ``<workspace>/files/`` and checked
    against the files root so that ``..`` traversal or absolute paths cannot
    escape the workspace.  Exits with code 1 on any path violation or if the
    file is absent.

    Args:
        workspace: Path to the directory written by assemble_pr_context().
        path: Relative path of the file within the ``files/`` subdirectory.
    """
    files_root = (Path(workspace) / "files").resolve()

    # Reject absolute paths before resolution — they would silently bypass the
    # containment check below if the absolute path happened to be inside files_root.
    if Path(path).is_absolute():
        print(f"error: absolute path not allowed: {path}", file=sys.stderr)
        sys.exit(1)

    target = (files_root / path).resolve()

    # Ensure the resolved target is still inside files_root (catches ".." etc.)
    try:
        target.relative_to(files_root)
    except ValueError:
        print(f"error: path escapes workspace: {path}", file=sys.stderr)
        sys.exit(1)

    if not target.exists():
        print(f"error: file not found in workspace: {path}", file=sys.stderr)
        sys.exit(1)

    print(target.read_text(encoding="utf-8"))


def cmd_conventions(workspace: str) -> None:
    """Print all materialized repo convention docs from the workspace.

    Each doc is printed with a header showing its filename followed by its
    content.  When no ``conventions/`` directory is present the command exits
    cleanly with no output.

    Args:
        workspace: Path to the directory written by assemble_pr_context().
    """
    conventions_root = Path(workspace) / "conventions"
    if not conventions_root.exists():
        return

    for doc_path in sorted(conventions_root.iterdir()):
        if doc_path.is_file():
            print(f"=== {doc_path.name} ===")
            print(doc_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    """Entry point for the heimdall-context CLI.

    Args:
        argv: Argument list; defaults to sys.argv[1:] when None.
    """
    parser = argparse.ArgumentParser(
        prog="heimdall-context",
        description="Read from a materialized Heimdall PR seed-context workspace.",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    diff_parser = sub.add_parser("diff", help="Print the unified diff")
    diff_parser.add_argument("workspace", help="Path to the materialized workspace")

    pr_parser = sub.add_parser("pr", help="Print PR metadata as JSON")
    pr_parser.add_argument("workspace", help="Path to the materialized workspace")

    file_parser = sub.add_parser(
        "file", help="Print a materialized changed-file content"
    )
    file_parser.add_argument("workspace", help="Path to the materialized workspace")
    file_parser.add_argument("path", help="Relative path within the files/ subdirectory")

    conventions_parser = sub.add_parser(
        "conventions", help="Print all repo convention docs"
    )
    conventions_parser.add_argument("workspace", help="Path to the materialized workspace")

    args = parser.parse_args(argv)

    if args.subcommand == "diff":
        cmd_diff(args.workspace)
    elif args.subcommand == "pr":
        cmd_pr(args.workspace)
    elif args.subcommand == "file":
        cmd_file(args.workspace, args.path)
    elif args.subcommand == "conventions":
        cmd_conventions(args.workspace)
    else:
        # argparse makes this unreachable, but keeps mypy happy
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
