"""Anchor lens findings to changed diff lines for inline review comments.

Splits the synthesized findings into two buckets:

  * **inline** — findings whose ``path:line`` location lands on a line ADDED by the
    PR (a right-side line in the unified diff). These become inline comments
    attached to the review submission at that exact ``file:line``.
  * **body** — every other finding (off-diff line, unknown file, or a missing /
    unparseable location). These are rendered in the review body instead, since
    GitHub only accepts inline comments anchored to a line that is part of the diff.

The single source of truth for "is this line commentable" is the unified diff:
:func:`commentable_lines` parses its hunks into the set of ``(path, new_line)`` pairs
on the right side. The mapping is deliberately robust — a finding with no location,
a non-numeric line, or a path outside the diff always falls back to the body rather
than producing an invalid inline comment that GitHub would reject.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from heimdall.lens import Finding, TaggedFinding, format_synthesis_body

logger = logging.getLogger(__name__)

# GitHub's create-review `comments` array anchors each comment on the RIGHT side
# (the PR head / new file) of the diff.
_DIFF_SIDE = "RIGHT"

# Matches a unified-diff hunk header: @@ -old_start,old_count +new_start,new_count @@
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# A finding location is "path:line" (line may be a single number or a start-end
# range, e.g. "foo.py:12" or "foo.py:12-18"); we anchor on the start line.
_LOCATION = re.compile(r"^(?P<path>.+):(?P<line>\d+)(?:-\d+)?$")


@dataclass(frozen=True)
class InlineComment:
    """An anchorable finding bound to a concrete ``(path, line)`` in the diff.

    Attributes:
        path: Repository-relative file path the comment anchors to.
        line: 1-based line number in the new (right-side) file.
        finding: The underlying lens :class:`~heimdall.lens.Finding`.
    """

    path: str
    line: int
    finding: Finding


def commentable_lines(diff: str) -> set[tuple[str, int]]:
    """Return the ``(path, new_line)`` pairs that are commentable in a unified diff.

    A line is commentable only when the PR ADDS it (a ``+`` line in a hunk); these
    are the right-side lines GitHub accepts inline comments on. Context lines and
    removed lines are excluded. The new-file line counter is advanced for added and
    context lines (and held for removed lines) exactly as the diff format prescribes.

    Args:
        diff: The raw unified diff text (e.g. ``diff.patch`` from the seed).

    Returns:
        The set of ``(path, line)`` pairs that can carry an inline comment. Empty
        when the diff is blank or has no added lines.
    """
    commentable: set[tuple[str, int]] = set()
    current_path: str | None = None
    new_line = 0
    in_hunk = False

    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            current_path = _strip_diff_path(raw[len("+++ ") :])
            in_hunk = False
            continue
        header = _HUNK_HEADER.match(raw)
        if header is not None:
            new_line = int(header.group(1))
            in_hunk = True
            continue
        if not in_hunk or current_path is None:
            continue
        if raw.startswith("+"):
            commentable.add((current_path, new_line))
            new_line += 1
        elif raw.startswith("-"):
            # Removed line: present only on the old side; new-line counter unchanged.
            continue
        elif raw.startswith("\\"):
            # "\ No newline at end of file" marker — not a real line.
            continue
        else:
            # Context line (leading space, or a blank line inside the hunk).
            new_line += 1

    return commentable


def _strip_diff_path(path: str) -> str:
    """Strip the ``b/`` prefix git puts on the new-file path of a diff header."""
    return path[2:] if path.startswith("b/") else path


def split_findings(
    tagged: list[TaggedFinding],
    commentable: set[tuple[str, int]],
) -> tuple[list[InlineComment], list[TaggedFinding]]:
    """Split findings into anchorable inline comments and body-fallback findings.

    A finding is anchorable when its ``location`` parses to ``path:line`` and that
    pair is in ``commentable`` (i.e. the line was added by the PR). Everything else —
    an off-diff line, an unknown file, or a missing / unparseable location — falls
    back to the body so no invalid inline comment is ever produced.

    Args:
        tagged: The synthesis survivors to route.
        commentable: The ``(path, line)`` pairs from :func:`commentable_lines`.

    Returns:
        ``(inline, body)`` where ``inline`` are anchorable findings bound to their
        ``(path, line)`` and ``body`` are the off-diff survivors to render in the body.
    """
    inline: list[InlineComment] = []
    body: list[TaggedFinding] = []
    for item in tagged:
        anchor = _anchor(item.finding.location)
        if anchor is not None and anchor in commentable:
            path, line = anchor
            inline.append(InlineComment(path=path, line=line, finding=item.finding))
        else:
            body.append(item)
    return inline, body


def _anchor(location: str | None) -> tuple[str, int] | None:
    """Parse a finding ``location`` into ``(path, line)``, or None if unusable.

    Tolerates a missing location, a path with no ``:line`` suffix, a non-numeric
    line, an empty path, and a ``start-end`` range (anchored on the start line).
    """
    if not location:
        return None
    match = _LOCATION.match(location.strip())
    if match is None:
        logger.debug("Finding location %r is not path:line; body fallback", location)
        return None
    path = match.group("path").strip()
    if not path:
        return None
    return path, int(match.group("line"))


def build_inline_comments(inline: list[InlineComment]) -> list[dict[str, Any]]:
    """Render anchorable findings into GitHub create-review ``comments`` entries.

    Each entry anchors on the RIGHT side at ``path:line`` and carries the finding's
    severity-tagged text as the comment body. The list is attached to the SAME review
    submission (not posted as separate calls), so the inline comments and the review
    body land atomically.

    Args:
        inline: The anchorable findings from :func:`split_findings`.

    Returns:
        The ``comments`` array for ``post_review`` (empty when there are none).
    """
    return [
        {
            "path": item.path,
            "line": item.line,
            "side": _DIFF_SIDE,
            "body": _inline_body(item.finding),
        }
        for item in inline
    ]


def _inline_body(finding: Finding) -> str:
    """Render a single finding's inline-comment body: a bold headline + message."""
    headline = f"**[{finding.severity.value.upper()}] {finding.title}**"
    return f"{headline}\n\n{finding.message}" if finding.message else headline


def render_body_for_offdiff(body_findings: list[TaggedFinding]) -> str:
    """Render the off-diff findings into the review body Markdown.

    Reuses :func:`~heimdall.lens.format_synthesis_body` so the body keeps the same
    severity-grouped, lens-tagged format — now carrying only the survivors that
    could not be anchored to a changed line (the rest live as inline comments).

    Args:
        body_findings: The off-diff survivors from :func:`split_findings`.

    Returns:
        The Markdown review body for the off-diff findings.
    """
    return format_synthesis_body(body_findings)
