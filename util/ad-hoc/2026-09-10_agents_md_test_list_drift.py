#!/usr/bin/env python3
"""2026-09-10_agents_md_test_list_drift.py -- which suites does the doc test-list fail to name?

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc analysis (documentation / CI consistency)
Author: Paul Calnon
License: MIT License

WHY THIS EXISTS

juniper-ml runs an explicit, HAND-WRITTEN list of suites in `ci.yml` -- there is no
discovery -- and the docs carry a SECOND hand-written list, the one a developer (or an
agent) reads to learn how to run the tests locally.

`tests/test_ci_test_wiring_drift.py` already gates one direction: every `tests/test_*.py`
must be INVOKED by ci.yml. Nothing gates the other two edges of the triangle, so the doc
list drifts silently and a suite CI runs is one the local instructions never mention. When
this was first written the doc list lived in `AGENTS.md` and named 115 of 164; the 2026-09-10 structure repair
relocated it to `docs/REFERENCE.md` § Test Suite Reference ("Running every suite") and
completed it, because finishing it in place would have left the always-loaded `AGENTS.md`
534 chars under its 38000-char ceiling. The doc list is read from there now.

This reports all three gaps rather than the one that prompted it:

  * run by CI, not named in the docs  -- the local instructions are incomplete;
  * named in the docs, not run by CI  -- the instructions promise a gate that does not gate;
  * on disk, in neither               -- an orphan (the class test_ci_test_wiring_drift.py
    exists to catch; a hit here means that gate is itself broken).

Both lists are read as INVOCATIONS (`python3 -m unittest -v tests/...`), never as substring
mentions. The regression step in ci.yml is a single `run: |` block scalar, so its `#` lines
are literal string content, not YAML comments -- a substring search matches a suite that is
only DISCUSSED in a comment and never runs, which is the vacuous-pass shape these gates
exist to refuse. For the same reason the doc side skips comment lines too.

EXIT CODES

  * 0 -- no gaps in any direction;
  * 1 -- at least one gap (all are printed);
  * 2 -- a list could not be read, or either list came back EMPTY (a rename upstream would
    otherwise present as "every suite is missing").

Usage:
    python util/ad-hoc/2026-09-10_agents_md_test_list_drift.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INVOCATION = re.compile(r"python3 -m unittest -v\s+(tests/test_[A-Za-z0-9_]+\.py)")


def _repo_root() -> Path:
    for cand in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (cand / ".github" / "workflows").is_dir():
            return cand
    print("could not locate repo root", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    root = _repo_root()
    try:
        ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        doc = (root / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
    except OSError as exc:
        print(f"could not read a list: {exc}", file=sys.stderr)
        return 2

    def _invocations(text: str) -> set:
        return {
            m.group(1)
            for line in text.splitlines()
            if not line.lstrip().startswith("#")
            for m in [INVOCATION.search(line)]
            if m
        }

    ci_set = _invocations(ci)
    ag_set = _invocations(doc)
    on_disk = {f"tests/{p.name}" for p in (root / "tests").glob("test_*.py")}

    if not ci_set or not ag_set:
        print(f"a list came back EMPTY (ci={len(ci_set)}, docs={len(ag_set)}) -- refusing",
              file=sys.stderr)
        return 2

    ci_only = sorted(ci_set - ag_set)
    ag_only = sorted(ag_set - ci_set)
    orphans = sorted(on_disk - ci_set - ag_set)

    print(f"ci.yml invokes      : {len(ci_set)}")
    print(f"REFERENCE.md names  : {len(ag_set)}")
    print(f"tests/ on disk      : {len(on_disk)}\n")

    for label, rows in (
        ("run by CI, NOT named in docs/REFERENCE.md", ci_only),
        ("named in docs/REFERENCE.md, NOT run by CI", ag_only),
        ("on disk, in NEITHER list", orphans),
    ):
        print(f"--- {label} ({len(rows)}) ---")
        for r in rows:
            print(f"    {r}")
        print()

    return 1 if (ci_only or ag_only or orphans) else 0


if __name__ == "__main__":
    raise SystemExit(main())
