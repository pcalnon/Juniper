#!/usr/bin/env python3
"""2026-09-10_fence_walker_crosscheck.py -- does the screen's fence walker agree with CommonMark?

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc validation (instrument conformance)
Author: Paul Calnon
License: MIT License

WHY THIS EXISTS

`util/ad-hoc/2026-09-05_markdown_structure_check.py` feeds a REQUIRED status context
(`util/markdown_structure_delta.py` imports it; the step runs in the `docs` job, whose name
`Documentation Links` is the required check). On 2026-09-10 its boolean fence toggle was
replaced with a length- and character-aware walk.

Relaxing an instrument that gates merges is the direction in which mistakes are invisible:
a walker that under-reports looks exactly like a clean tree. The screen's own unit tests
pin its behaviour on fixtures the author chose, which cannot answer "is the model right in
general".

So this compares the screen's line-level fence model against `markdown-it-py`'s -- a
CommonMark 0.31 reference implementation, and an INDEPENDENT one -- over every markdown file
handed to it, and reports every line where they disagree.

Agreement is not proof of correctness, but a disagreement is proof of a defect in one of
them, and the disagreements are what the old toggle was full of: run this against
`git stash`-ed old code and it lights up on every ```bash-inside-a-block and every
four-backtick sample in the tree.

EXIT CODES

  * 0 -- the two models agree on every line of every file examined;
  * 1 -- at least one disagreement (each is printed);
  * 2 -- could not run honestly: no paths, an unreadable path, zero files examined, or
    markdown-it missing. A conformance check that examined nothing must not report success.

Usage:
    python util/ad-hoc/2026-09-10_fence_walker_crosscheck.py $(git ls-files '*.md')
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCREEN = Path(__file__).with_name("2026-09-05_markdown_structure_check.py")


def _load_screen():
    spec = importlib.util.spec_from_file_location("md_structure_screen", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module: a @dataclass (or any decorator resolving __module__)
    # in a path-loaded module dies at import without it.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference_fence_lines(text: str) -> set:
    """0-based line indices markdown-it places inside a fenced code block, delimiters included."""
    from markdown_it import MarkdownIt

    out = set()
    for tok in MarkdownIt("commonmark").parse(text):
        if tok.type == "fence" and tok.map:
            start, end = tok.map
            out.update(range(start, end))
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: 2026-09-10_fence_walker_crosscheck.py FILE.md ...", file=sys.stderr)
        return 2
    try:
        import markdown_it  # noqa: F401
    except ImportError:
        print("markdown-it-py is required: pip install markdown-it-py", file=sys.stderr)
        return 2

    screen = _load_screen()
    examined = 0
    unreadable: list = []
    disagreements = 0

    for arg in argv:
        p = Path(arg)
        if p.suffix.lower() != ".md":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            unreadable.append(f"{arg}: {exc}")
            continue
        examined += 1
        lines = text.splitlines()
        spans, _unclosed = screen._fence_spans(lines)
        mine = {i for i, s in enumerate(spans) if s is not None}
        theirs = _reference_fence_lines(text)
        # markdown-it does not emit a `fence` token for an UNCLOSED fence run to EOF in
        # every case; compare only where both have an opinion about a closed block by
        # taking the symmetric difference and reporting it in full.
        diff = sorted(mine ^ theirs)
        if diff:
            disagreements += len(diff)
            print(f"=== {arg} === ({len(diff)} line(s) disagree)")
            for i in diff[:12]:
                who = "screen-only" if i in mine else "markdown-it-only"
                body = lines[i][:60] if i < len(lines) else ""
                print(f"   line {i+1:5d} {who:17} | {body}")
            if len(diff) > 12:
                print(f"   ... and {len(diff)-12} more")

    if unreadable:
        print(f"could not read {len(unreadable)} path(s):", file=sys.stderr)
        for u in unreadable:
            print(f"    {u}", file=sys.stderr)
        return 2
    if examined == 0:
        print("examined 0 markdown file(s) -- refusing to report success", file=sys.stderr)
        return 2

    print(f"\nexamined {examined} file(s); {disagreements} disagreeing line(s)")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
