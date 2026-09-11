#!/usr/bin/env python3
"""2026-09-10_restore_recurse_model_suites.py -- put back the four §3.2 suites 4da40fe9 dropped.

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc repair (documentation content recovery)
Author: Paul Calnon
License: MIT License

WHAT WAS LOST

`notes/JUNIPER_2026-05-31_JUNIPER-ECOSYSTEM_MODEL-MIDDLEWARE-REFACTOR-DESIGN-AND-PLAN.md:360`
assigns EIGHT suites to the companion model document:

    "Moved to the companion model document (§3.2) -- numerical-correctness,
     known-answer/golden, regression-metric, time-series-leakage, growth-loop,
     determinism, overfit-tiny, and architecture-specific stability suites"

`notes/JUNIPER_2026-05-31_JUNIPER-RECURRENCE_RECURSE-MODEL-DESIGN-AND-PLAN.md` §3.2 carries
four. The other four are a live cross-document inconsistency, not a design choice.

WHY THIS IS A RESTORE AND NOT A RATIFICATION

`4da40fe9` ("minor formatting changes to design docs", 2026-06-07) PADDED table columns
across the whole file. In the §3.2 hunk it padded four rows and, in the SAME hunk, deleted
the table header, the separator, and the other four rows. A deliberate removal of four
suites does not also delete the header the surviving four still need -- that is a reformat
that ate the top of a table.

`b264c2a2` (2026-09-09) then restored the header and separator and left the rows out,
reading one deletion two ways. This closes that.

Two further tells, both checked rather than assumed:

  * companion §3.3 does NOT hold the four (zero hits for `Numerical correctness` there);
  * the survivor/deleted split is INVERTED against any "keep the model-specific ones"
    theory -- deleted `Known-answer / golden` names the Reber grammar, the most
    model-specific content in the table, while surviving `Determinism` is the one the
    companion's §3.1 already carries.

SOURCE OF TRUTH

`22c32bd1:notes/JUNIPER_RECURSE_MODEL_DESIGN_AND_PLAN_2026-05-31.md` -- the PRE-RENAME
path. The current path exits 128 at that commit (432ed644 renamed 251 notes files on
2026-07-04). Rows are restored BYTE-IDENTICAL to what `4da40fe9` removed; nothing is
rewritten or re-padded, so the restore is verifiable by `git show` rather than by reading.

Every target line is asserted before the write. A repair script that cannot find what it
expects must refuse, not guess -- `2026-09-08_repair_notes_tables.py` caught an off-by-one
that way.

Usage:
    python util/ad-hoc/2026-09-10_restore_recurse_model_suites.py          # dry run
    python util/ad-hoc/2026-09-10_restore_recurse_model_suites.py --apply
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TARGET = Path("notes/JUNIPER_2026-05-31_JUNIPER-RECURRENCE_RECURSE-MODEL-DESIGN-AND-PLAN.md")
SOURCE_REV = "22c32bd1"
SOURCE_PATH = "notes/JUNIPER_RECURSE_MODEL_DESIGN_AND_PLAN_2026-05-31.md"

# The four suites named by the companion and absent from the target.
RESTORE_KEYS = (
    "**Numerical correctness**",
    "**Known-answer / golden**",
    "**Regression metrics**",
    "**Time-series correctness**",
)
# The row the four must precede -- the first survivor, so original order is preserved.
ANCHOR = "| **Growth-loop correctness** (GrowableModel) |"


def _source_rows() -> list:
    """The four rows, read out of the pre-rename blob at SOURCE_REV."""
    blob = subprocess.run(
        ["git", "show", f"{SOURCE_REV}:{SOURCE_PATH}"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    rows = []
    for key in RESTORE_KEYS:
        hits = [ln for ln in blob if ln.startswith(f"| {key} |")]
        if len(hits) != 1:
            raise AssertionError(f"{key!r}: expected exactly 1 row at {SOURCE_REV}, found {len(hits)}")
        rows.append(hits[0])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the file (default: dry run)")
    args = ap.parse_args(argv)

    lines = TARGET.read_text(encoding="utf-8").split("\n")

    # Assert the damage is still exactly as described before touching anything.
    for key in RESTORE_KEYS:
        present = [i for i, ln in enumerate(lines) if ln.startswith(f"| {key} |")]
        if present:
            print(f"{key} is ALREADY present at line {present[0]+1} -- refusing", file=sys.stderr)
            return 2
    anchors = [i for i, ln in enumerate(lines) if ln.startswith(ANCHOR)]
    if len(anchors) != 1:
        print(f"anchor row found {len(anchors)} times, expected 1 -- refusing", file=sys.stderr)
        return 2
    at = anchors[0]
    if not lines[at - 1].startswith("|---") or not lines[at - 2].startswith("| Suite |"):
        print(f"line {at} / {at-1} are not the separator / header -- refusing", file=sys.stderr)
        return 2

    rows = _source_rows()
    print(f"restoring {len(rows)} row(s) into {TARGET} before line {at+1}:\n")
    for r in rows:
        print(f"  + {r[:100]}")

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return 0

    lines[at:at] = rows
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"\napplied: {TARGET} now has {len(lines)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
