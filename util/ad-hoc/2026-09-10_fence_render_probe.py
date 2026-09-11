#!/usr/bin/env python3
"""2026-09-10_fence_render_probe.py -- what does a markdown renderer ACTUALLY do with these lines?

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc analysis (documentation structural integrity)
Author: Paul Calnon
License: MIT License

WHY THIS EXISTS

`util/ad-hoc/2026-09-05_markdown_structure_check.py` walks fences with a BOOLEAN TOGGLE:
every line starting ``` flips it. CommonMark does not work that way.

  * a closing fence may NOT carry an info string, so ```bash cannot close anything;
  * a closing fence must be AT LEAST AS LONG as the opener, so ``` cannot close ````;
  * ``` and ~~~ are different fence characters and never close each other.

The toggle therefore disagrees with every real renderer wherever those cases occur, and the
disagreement is silent -- it produces a plausible, wrong answer rather than an error. The
2026-09-09 handoff records the cost: two round-1 agents reported "three unclosed fences" on
`main`, the reviewer "verified" it by re-running the screen, and the true answer is two. The
screen was the instrument under suspicion and was used as its own witness.

This probe asks the question the other way round. It hands the file to `markdown-it-py`
(CommonMark 0.31) and reports, per line, whether the RENDERER places it in a fenced code
block or in prose. A line the author believes is inside a ```jinja2 sample, and which the
renderer emits as a visible paragraph, is a rendering defect regardless of what any
hand-written walker says.

WHAT IT REPORTS

For each line in the requested range: `CODE` (inside a fence, with the opening line number
and info string) or `prose`. `--diff-against` re-parses a second file and prints only the
lines whose classification CHANGED -- which is how a fence repair is verified: the lines
that were wrongly prose must become CODE, and nothing else may move.

EXIT CODES

  * 0 -- parsed and reported;
  * 2 -- the file could not be read, the range is empty, or markdown-it is not installed.
    A probe that cannot parse must not print a clean-looking report.

Usage:
    python util/ad-hoc/2026-09-10_fence_render_probe.py FILE [--range 800,880]
    python util/ad-hoc/2026-09-10_fence_render_probe.py FILE --diff-against FILE.fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _classify(text: str) -> dict:
    """Map 1-based line number -> None (prose) or (opener_line, info_string).

    `markdown-it` gives every block token a `map` of [start, end) 0-based line numbers.
    A `fence` token's map spans the OPENING fence line through the line after the closing
    one, so the content is map[0]+1 .. map[1]-1 when the fence is closed, and runs to EOF
    when it is not. We record the whole span including its delimiters, because "is this
    line part of a code block" is the question a reader of the rendered page is asking.
    """
    try:
        from markdown_it import MarkdownIt
    except ImportError:  # pragma: no cover - environment guard
        print("markdown-it-py is required: pip install markdown-it-py", file=sys.stderr)
        raise SystemExit(2)

    md = MarkdownIt("commonmark")
    tokens = md.parse(text)
    out: dict = {}
    for tok in tokens:
        if tok.type != "fence" or not tok.map:
            continue
        start, end = tok.map
        info = tok.info.strip() or "(bare)"
        for ln in range(start + 1, end + 1):
            out[ln] = (start + 1, info)
    return out


def _read(path: Path) -> tuple:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return text, text.splitlines()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--range", dest="rng", help="FIRST,LAST (1-based, inclusive)")
    ap.add_argument("--diff-against", dest="other", help="report only lines whose class CHANGED")
    args = ap.parse_args(argv)

    text, lines = _read(Path(args.file))
    cls = _classify(text)

    if args.rng:
        first, last = (int(x) for x in args.rng.split(","))
    else:
        first, last = 1, len(lines)
    first, last = max(1, first), min(len(lines), last)
    if first > last:
        print("empty range -- refusing to report", file=sys.stderr)
        return 2

    if args.other:
        other_text, other_lines = _read(Path(args.other))
        other_cls = _classify(other_text)
        changed = 0
        for ln in range(first, last + 1):
            a, b = cls.get(ln), other_cls.get(ln)
            if a == b:
                continue
            changed += 1
            body = (lines[ln - 1] if ln <= len(lines) else "")[:64]
            print(f"{ln:5d}  {_label(a):<22} -> {_label(b):<22} | {body}")
        print(f"\n{changed} line(s) changed classification in {first}..{last}")
        return 0

    for ln in range(first, last + 1):
        print(f"{ln:5d}  {_label(cls.get(ln)):<22} | {lines[ln - 1][:72]}")
    return 0


def _label(entry) -> str:
    if entry is None:
        return "prose"
    return f"CODE(@{entry[0]} {entry[1]})"


if __name__ == "__main__":
    raise SystemExit(main())
