#!/usr/bin/env python3
"""2026-09-10_relocate_test_command_list.py -- move the 164 run-the-tests commands out of AGENTS.md.

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc repair (memory budget / documentation consistency)
Author: Paul Calnon
License: MIT License

THE PROBLEM

`AGENTS.md` carries a hand-written list of `python3 -m unittest -v tests/...` commands, and
`ci.yml` carries a second one. Nothing gates the two against each other, so they drifted:
ci.yml invokes **164** suites and AGENTS.md named **115**. The 2026-09-09 handoff reported
this as ONE missing suite (`tests/test_markdown_structure_screen.py`); it is 49, that one
among them. Nothing is named in AGENTS.md that CI does not run, and nothing on disk is
absent from both -- the drift is one-directional incompleteness.

ml#1883 (merged 2026-09-10, while this was being written) added three soak suites --
`test_soak_ledger`, `test_soak_next_probe`, `test_soak_run_probe` -- and recorded "Set
difference against ci.yml is now empty." Measured at its own merge commit `83556f26`,
AGENTS.md named 118 of 164 and the set difference was **46**. The three it was looking at
were real and are kept here; the generalisation from "the suites I checked" to "the set
difference" is the reason this tool computes the difference rather than asserting it.

WHY THE LIST MOVES INSTEAD OF BEING COMPLETED

AGENTS.md is an always-loaded memory file under a character ceiling
(`conf/memory_budget.json`: 38000, enforced by the `Memory Budget` job). Measured, not
estimated: the file is 34585 chars, the 49 missing lines cost 2881, and completing the list
in place lands at 37466 -- **534 chars of headroom**, below the +605 that a single
documentation PR has already been measured to cost. Completing it in place would leave the
file effectively at its ceiling and fail the next author.

Relocating instead continues a migration this file has already made SEVEN times -- Shared
Observability Helpers, Shared Service-Core Contracts, Repository Structure, Utility Scripts,
Test Suite descriptions, CI/CD Workflow Inventory, Pre-commit Hooks -- each replaced by a
one-line pointer into `docs/REFERENCE.md`, which is deliberately NOT budget-governed because
it is the migration DESTINATION. The per-suite DESCRIPTIONS are already there; only the raw
command list stayed behind, which is the inconsistency this closes.

SOURCE OF TRUTH

The commands are regenerated from ci.yml's regression step, not copied from AGENTS.md --
that is what makes the relocated list complete rather than a move of a stale one. Comment
lines inside the step are skipped: the step is a `run: |` block scalar, so its `#` lines are
literal content and a substring match would pick up suites that are only discussed.

Usage:
    python util/ad-hoc/2026-09-10_relocate_test_command_list.py          # dry run
    python util/ad-hoc/2026-09-10_relocate_test_command_list.py --apply
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INVOCATION = re.compile(r"^\s*python3 -m unittest -v\s+(tests/test_[A-Za-z0-9_]+\.py)\s*$")

AGENTS = Path("AGENTS.md")
REFERENCE = Path("docs/REFERENCE.md")
CI = Path(".github/workflows/ci.yml")

ANCHOR_HEADING = "## Test Suite Reference"
POINTER = (
    "# Run every regression suite. The full ordered list of 164 `python3 -m unittest`\n"
    "# commands lives in docs/REFERENCE.md -- see \"Running every suite\" under\n"
    "# § Test Suite Reference. It is generated from ci.yml's regression step, which is\n"
    "# the authoritative list; tests/test_ci_test_wiring_drift.py gates that every suite\n"
    "# on disk is invoked there, and util/ad-hoc/2026-09-10_agents_md_test_list_drift.py\n"
    "# reports any suite CI runs that the reference does not name.\n"
)


def _ci_commands() -> list:
    """Ordered suite paths as ci.yml actually invokes them (comment lines skipped)."""
    out, seen = [], set()
    for line in CI.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = INVOCATION.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args(argv)

    suites = _ci_commands()
    if len(suites) < 100:
        print(f"only {len(suites)} suites parsed from ci.yml -- refusing", file=sys.stderr)
        return 2

    ag_lines = AGENTS.read_text(encoding="utf-8").split("\n")
    first = next((i for i, l in enumerate(ag_lines) if INVOCATION.match(l)), None)
    if first is None:
        print("no unittest invocations found in AGENTS.md -- already relocated?", file=sys.stderr)
        return 2
    last = max(i for i, l in enumerate(ag_lines) if INVOCATION.match(l))
    block = [l for l in ag_lines[first:last + 1] if INVOCATION.match(l)]
    if len(block) != (last - first + 1):
        strays = [l for l in ag_lines[first:last + 1] if not INVOCATION.match(l)]
        print(f"the AGENTS.md block is not contiguous; {len(strays)} stray line(s):", file=sys.stderr)
        for s in strays[:8]:
            print(f"    {s!r}", file=sys.stderr)
        return 2

    before_chars = len(AGENTS.read_text(encoding="utf-8"))
    print(f"AGENTS.md lines {first+1}-{last+1}: {len(block)} unittest command(s)")
    print(f"ci.yml invokes {len(suites)}; the relocated list will name all of them")
    print(f"AGENTS.md {before_chars} chars -> ~{before_chars - sum(len(l)+1 for l in block) + len(POINTER)}")

    ref_lines = REFERENCE.read_text(encoding="utf-8").split("\n")
    anchor = next((i for i, l in enumerate(ref_lines) if l.strip() == ANCHOR_HEADING), None)
    if anchor is None:
        print(f"{ANCHOR_HEADING!r} not found in {REFERENCE} -- refusing", file=sys.stderr)
        return 2
    if any("### Running every suite" in l for l in ref_lines):
        print("REFERENCE.md already has 'Running every suite' -- refusing", file=sys.stderr)
        return 2

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return 0

    # 1. AGENTS.md: the block becomes a pointer.
    ag_lines[first:last + 1] = POINTER.rstrip("\n").split("\n")
    AGENTS.write_text("\n".join(ag_lines), encoding="utf-8")

    # 2. REFERENCE.md: insert the complete list right after the section heading's preamble.
    insert_at = anchor + 1
    # Built as named paragraphs rather than adjacent string literals inside the list: in a
    # list literal, implicit concatenation is indistinguishable from a MISSING COMMA, which
    # silently welds two entries into one. CodeQL flags it for that reason (code-scanning
    # alert 683 on this file), and in a list whose entries become markdown LINES a missing
    # comma would merge two lines with no error anywhere.
    intro = (
        "The complete ordered list, generated from `.github/workflows/ci.yml`'s "
        f"`Run Python regression tests` step on 2026-09-10 -- **{len(suites)} suites**. This "
        "relocated from `AGENTS.md` (the 2026-09-10 structure repair), where the "
        "hand-maintained copy had drifted to 115 and completing it in place would have left "
        "the always-loaded file 534 chars under its 38000-char ceiling."
    )
    provenance = (
        "ci.yml is the authoritative list. `tests/test_ci_test_wiring_drift.py` gates that "
        "every `tests/test_*.py` on disk is invoked there; "
        "`util/ad-hoc/2026-09-10_agents_md_test_list_drift.py` reports any suite CI runs "
        "that this list does not name."
    )
    body = [
        "",
        "### Running every suite",
        "",
        intro,
        "",
        provenance,
        "",
        "```bash",
    ]
    body += [f"python3 -m unittest -v {s}" for s in suites]
    body += [
        "bash scripts/test_resume_file_safety.bash",
        "# doc-link validator regression tests live in juniper-doc-tools/tests/",
        "# and run under the dedicated `CI -- juniper-doc-tools` workflow.",
        "```",
    ]
    ref_lines[insert_at:insert_at] = body
    REFERENCE.write_text("\n".join(ref_lines), encoding="utf-8")

    print(f"\napplied: AGENTS.md {before_chars} -> {len(AGENTS.read_text(encoding='utf-8'))} chars")
    print(f"         docs/REFERENCE.md gained {len(body)} lines naming {len(suites)} suites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
