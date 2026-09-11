#!/usr/bin/env python3
"""2026-09-10_retarget_dangling_note_links.py -- repoint the ten notes symlinks 432ed644 broke.

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc repair (documentation cross-reference recovery)
Author: Paul Calnon
License: MIT License

WHAT BROKE

`432ed644` ("rename 251 notes files to JUNIPER_<DATE>_JUNIPER-<REPO>_<PHRASE>", 2026-07-04)
renamed both ends of ten symlinks: the LINKS were renamed with everything else, and so were
their TARGETS -- but a symlink's body is opaque text, so nothing rewrote it. All ten have
pointed at pre-rename basenames ever since.

Every target is a tracked file on `main` today. Nothing is lost; the links just name the old
spelling. The rename map below is not inferred from the numbering -- it is read out of
`git show 432ed644 --diff-filter=R`, where every one of these is an R100 (100% similarity)
rename.

WHY THIS IS NOT ONE RECIPE

A predecessor's first reading was "one repairable, nine permanently broken", from grepping
`notes/legacy/regressions/` -- the links' RESOLVED path, which has never existed on `main`.
The targets live in `notes/regressions/`, one level UP. Searching a resolved path proves
only that the path is wrong.

The correction then over-generalised the other way, to "all ten take
`../regressions/<renamed basename>`". That is true of NINE. The tenth
(`notes/development/...V7-IMPLEMENTATION-ROADMAP.md`) is not in `notes/regressions/` at all,
sits one level up in `notes/`, and its DATE CHANGED in the rename (2026-04-24 in the link's
own name, 2026-05-25 in the target's) -- so its basename is not mechanically derivable from
the link's, and the nine-link recipe would leave it dangling.

Hence two explicit maps, and a post-condition that resolves every link on disk.

Usage:
    python util/ad-hoc/2026-09-10_retarget_dangling_note_links.py          # dry run
    python util/ad-hoc/2026-09-10_retarget_dangling_note_links.py --apply
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- the NINE under notes/legacy/ -> ../regressions/<renamed basename> -----------------
# key: link path; value: the renamed basename in notes/regressions/ (verified R100).
LEGACY = {
    "notes/legacy/CANOPY_REGRESSION_ANALYSIS.md":
        "JUNIPER_2026-04-02_JUNIPER-CANOPY_REGRESSION-ANALYSIS-06.md",
    "notes/legacy/CASCOR_TRAINING_FAILURE_ANALYSIS.md":
        "JUNIPER_2026-04-02_JUNIPER-CASCOR_REGRESSION-ANALYSIS-07.md",
    "notes/legacy/JUNIPER_REGRESSION_ANALYSIS_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-ANALYSIS-08.md",
    "notes/legacy/JUNIPER_REGRESSION_DEVELOPMENT_ROADMAP.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-DEVELOPMENT-ROADMAP-03.md",
    "notes/legacy/JUNIPER_REGRESSION_DEVELOPMENT_ROADMAP_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-DEVELOPMENT-ROADMAP-04.md",
    "notes/legacy/JUNIPER_REGRESSION_REMEDIATION_PLAN_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-REMEDIATION-PLAN-03.md",
    "notes/legacy/REGRESSION_ANALYSIS_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-ANALYSIS-09.md",
    "notes/legacy/REGRESSION_DEVELOPMENT_ROADMAP_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-DEVELOPMENT-ROADMAP-04.md",
    "notes/legacy/REGRESSION_REMEDIATION_PLAN_2026-04-02.md":
        "JUNIPER_2026-04-02_JUNIPER-ECOSYSTEM_REGRESSION-REMEDIATION-PLAN-04.md",
}

# --- the TENTH: a different directory, a different level, and a different DATE ---------
TENTH_LINK = ("notes/development/"
              "JUNIPER_2026-04-24_JUNIPER-ECOSYSTEM_OUTSTANDING-DEVELOPMENT-ITEMS-V7-IMPLEMENTATION-ROADMAP.md")
TENTH_TARGET = ("../"
                "JUNIPER_2026-05-25_JUNIPER-ECOSYSTEM_OUTSTANDING-DEVELOPMENT-ITEMS-V7-IMPLEMENTATION-ROADMAP.md")

# The pre-rename bodies, asserted before each rewrite so a link someone already fixed, or
# one pointing somewhere unexpected, refuses rather than being silently overwritten.
EXPECTED_LEGACY_PREFIX = "regressions/"
EXPECTED_TENTH = "../JUNIPER_OUTSTANDING_DEVELOPMENT_ITEMS_V7_IMPLEMENTATION_ROADMAP.md"


def _plan() -> list:
    out = []
    for link, basename in LEGACY.items():
        out.append((link, f"../regressions/{basename}", EXPECTED_LEGACY_PREFIX))
    out.append((TENTH_LINK, TENTH_TARGET, EXPECTED_TENTH))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="rewrite the links (default: dry run)")
    args = ap.parse_args(argv)

    plan = _plan()
    errors = []
    actions = []

    for link, new_target, expected in plan:
        p = Path(link)
        if not p.is_symlink():
            errors.append(f"{link}: not a symlink")
            continue
        current = os.readlink(p)
        if expected.endswith("/"):
            if not current.startswith(expected):
                errors.append(f"{link}: body {current!r} does not start {expected!r}")
                continue
        elif current != expected:
            errors.append(f"{link}: body {current!r} != expected {expected!r}")
            continue
        # The new target must be a real file, resolved RELATIVE TO THE LINK'S directory.
        resolved = (p.parent / new_target).resolve()
        if not resolved.is_file():
            errors.append(f"{link}: new target does not resolve to a file: {resolved}")
            continue
        actions.append((p, current, new_target))

    if errors:
        print("refusing -- preconditions failed:", file=sys.stderr)
        for e in errors:
            print(f"    {e}", file=sys.stderr)
        return 2

    print(f"{len(actions)} link(s) to retarget:\n")
    for p, current, new_target in actions:
        print(f"  {p}\n      {current}\n   -> {new_target}")

    if not args.apply:
        print("\ndry run -- pass --apply to rewrite")
        return 0

    for p, _current, new_target in actions:
        p.unlink()
        p.symlink_to(new_target)

    # Post-condition: every link resolves. `is_file()` FOLLOWS symlinks, which is exactly
    # why a dangling one scored clean for months -- here that is the property we want.
    still_broken = [str(p) for p, _c, _t in actions if not p.is_file()]
    if still_broken:
        print("APPLIED BUT STILL DANGLING:", file=sys.stderr)
        for s in still_broken:
            print(f"    {s}", file=sys.stderr)
        return 2
    print(f"\napplied: {len(actions)} link(s) retargeted; all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
