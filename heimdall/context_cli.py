"""heimdall-context — CLI wrapper for reading materialized PR seed context.

Provides two subcommands:
  heimdall-context diff <workspace>   — print the unified diff
  heimdall-context pr   <workspace>   — print PR metadata as JSON

The workspace must be a directory previously produced by assemble_pr_context()
(i.e. it contains diff.patch and pr_metadata.json).

This wrapper is the only allowlisted Bash command used during AI-driven lens
review sessions — it reads from pre-materialized data and executes nothing.
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

    args = parser.parse_args(argv)

    if args.subcommand == "diff":
        cmd_diff(args.workspace)
    elif args.subcommand == "pr":
        cmd_pr(args.workspace)
    else:
        # argparse makes this unreachable, but keeps mypy happy
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
