"""Tests for diff-anchoring: split findings into inline comments vs body fallback."""

from __future__ import annotations

from heimdall.diff_anchor import (
    InlineComment,
    build_inline_comments,
    commentable_lines,
    split_findings,
)
from heimdall.lens import Finding, Severity, TaggedFinding

# A small unified diff touching two files. Added (right-side) lines are commentable.
_DIFF = """\
diff --git a/heimdall/foo.py b/heimdall/foo.py
index 1111111..2222222 100644
--- a/heimdall/foo.py
+++ b/heimdall/foo.py
@@ -1,4 +1,6 @@
 import os
 import sys
+import json
+import re

 def foo():
@@ -10,3 +12,4 @@ def foo():
     return 1
+    # trailing
 def bar():
     pass
diff --git a/heimdall/bar.py b/heimdall/bar.py
index 3333333..4444444 100644
--- a/heimdall/bar.py
+++ b/heimdall/bar.py
@@ -5,2 +5,3 @@ class C:
     x = 1
+    y = 2
     z = 3
"""


def _tagged(location: str | None, *, lens: str = "security") -> TaggedFinding:
    return TaggedFinding(
        lens=lens,
        finding=Finding(
            severity=Severity.HIGH, title="t", message="m", location=location
        ),
    )


# ---------------------------------------------------------------------------
# commentable_lines: which (path, line) pairs the diff allows comments on
# ---------------------------------------------------------------------------


def test_commentable_lines_collects_added_lines_per_file() -> None:
    """Only added (right-side, '+') lines are commentable, keyed by new-file line."""
    pairs = commentable_lines(_DIFF)
    # First hunk adds new lines 3 and 4 (import json / import re).
    assert ("heimdall/foo.py", 3) in pairs
    assert ("heimdall/foo.py", 4) in pairs
    # Second hunk adds new line 13 (# trailing) — context shifts the counter.
    assert ("heimdall/foo.py", 13) in pairs
    # bar.py adds new line 6 (y = 2).
    assert ("heimdall/bar.py", 6) in pairs


def test_commentable_lines_excludes_context_and_other_files() -> None:
    """Unchanged context lines and lines outside the diff are not commentable."""
    pairs = commentable_lines(_DIFF)
    # Line 1 (import os) is context, not added.
    assert ("heimdall/foo.py", 1) not in pairs
    # A file not in the diff is never commentable.
    assert ("heimdall/other.py", 3) not in pairs


def test_commentable_lines_tolerates_empty_diff() -> None:
    """An empty or whitespace-only diff yields no commentable lines."""
    assert commentable_lines("") == set()
    assert commentable_lines("   \n\n") == set()


def test_commentable_lines_added_content_starting_with_plus_plus() -> None:
    """An added line whose content begins with '++ ' is not a file header.

    The diff line ``+++ x`` is an ADDED content line (its text is ``++ x``), not a
    new-file ``+++ `` header. It and every later added line in the same hunk must
    stay commentable — a header misread would silently drop them.
    """
    diff = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1,1 +1,3 @@\n a\n+++ x\n+after\n"
    )
    pairs = commentable_lines(diff)
    # The '++ x' content line is new-file line 2; 'after' is line 3 — both added.
    assert pairs == {("f.py", 2), ("f.py", 3)}


# ---------------------------------------------------------------------------
# split_findings: anchorable -> inline; off-diff / unparseable -> body
# ---------------------------------------------------------------------------


def test_split_findings_routes_changed_line_to_inline() -> None:
    """A finding on a changed line goes to the inline bucket."""
    commentable = commentable_lines(_DIFF)
    inline, body = split_findings([_tagged("heimdall/foo.py:3")], commentable)
    assert len(inline) == 1
    assert body == []
    assert inline[0].finding.location == "heimdall/foo.py:3"


def test_split_findings_routes_off_diff_line_to_body() -> None:
    """A finding on a line not present in the diff falls back to the body."""
    commentable = commentable_lines(_DIFF)
    # Line 1 of foo.py is context (not added) -> not commentable -> body.
    inline, body = split_findings([_tagged("heimdall/foo.py:1")], commentable)
    assert inline == []
    assert len(body) == 1


def test_split_findings_routes_unparseable_location_to_body() -> None:
    """A finding with a missing or unparseable location falls back to the body."""
    commentable = commentable_lines(_DIFF)
    cases = [
        _tagged(None),
        _tagged("no-colon-here"),
        _tagged("heimdall/foo.py:notanumber"),
        _tagged("heimdall/foo.py"),
        _tagged(":42"),
    ]
    inline, body = split_findings(cases, commentable)
    assert inline == []
    assert len(body) == len(cases)


def test_split_findings_routes_unknown_file_to_body() -> None:
    """A finding on a file absent from the diff falls back to the body."""
    commentable = commentable_lines(_DIFF)
    inline, body = split_findings([_tagged("heimdall/missing.py:3")], commentable)
    assert inline == []
    assert len(body) == 1


def test_split_findings_handles_line_range_location() -> None:
    """A 'path:start-end' location anchors on the start line when it is commentable."""
    commentable = commentable_lines(_DIFF)
    inline, body = split_findings([_tagged("heimdall/foo.py:3-5")], commentable)
    assert len(inline) == 1
    assert body == []


# ---------------------------------------------------------------------------
# build_inline_comments: shape of the GitHub review `comments` array entries
# ---------------------------------------------------------------------------


def test_build_inline_comments_shape() -> None:
    """Each inline comment carries path, line, RIGHT side, and a rendered body."""
    inline = [
        InlineComment(path="heimdall/foo.py", line=3, finding=_tagged("x:3").finding),
    ]
    comments = build_inline_comments(
        [InlineComment(path="heimdall/foo.py", line=3, finding=inline[0].finding)],
    )
    assert comments == [
        {
            "path": "heimdall/foo.py",
            "line": 3,
            "side": "RIGHT",
            "body": comments[0]["body"],
        }
    ]
    assert "t" in comments[0]["body"]


def test_build_inline_comments_empty() -> None:
    """No anchorable findings yields an empty comments array."""
    assert build_inline_comments([]) == []
