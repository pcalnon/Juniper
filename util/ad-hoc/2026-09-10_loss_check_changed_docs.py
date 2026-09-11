#!/usr/bin/env python3
"""2026-09-10_loss_check_changed_docs.py -- run the loss check over every markdown file a PR touches.

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc validation (content-loss screening)
Author: Paul Calnon
License: MIT License

WHY THIS EXISTS

`util/ad-hoc/2026-09-08_section_loss_check.py` answers "what does the BEFORE have that the
AFTER lacks?" for ONE file pair. It exists because ml#1837 collapsed four duplicate soak
sections and deleted all THREE copies of a systemd install recipe, leaving zero -- and every
check on the way in passed.

Running it one file at a time is how a multi-file PR still ships a loss: the author checks
the file they were thinking about. This runs it over EVERY markdown file changed against a
base ref, so the question is asked of the whole change set rather than of the author's
attention.

It reports, never repairs. A reported loss is a prompt to ADJUDICATE -- deliberate drop or
genuine loss -- not a failure. That distinction is the whole point: the 2026-09-10 structure repair deliberately
dropped 115 test-command lines from `AGENTS.md` (they moved to `docs/REFERENCE.md`), and the
adjudication was to show all 115 present at the destination rather than to assert it.

A FENCE REPAIR ALWAYS TRIPS THIS, AND ALMOST NEVER MEANS WHAT IT SAYS

Half of `atoms()` is `fenced_lines()` -- every non-blank line INSIDE a fenced block. Repairing
a lost fence un-fences the lines the fence had swallowed, so every one of them stops being an
atom and is reported LOST while its text never moved. On the 2026-09-10 structure repair that read as 209 lost atoms
in one prompt file and 109 in another, with ZERO characters removed from either.

So adjudicate a fence repair by asking the SECOND question, which this tool deliberately does
not answer for you: is each reported atom still present as a raw non-blank line in the AFTER
file? If it is, the atom changed CLASSIFICATION rather than existence. On the 2026-09-10 structure repair exactly one
atom failed that test across four repaired files -- the welded line
`# In _reset_state_and_history() or a network reset method:# Juniper Cascor Concurrency`,
which the repair SPLIT in two, both halves present. That is the shape a real adjudication
has: a number, and a named exception with a reason.

EXIT CODES

  * 0 -- every changed markdown file lost nothing;
  * 1 -- at least one file lost atoms (each is listed, for adjudication);
  * 2 -- could not run: no changed files resolved, or the loss checker is missing. A screen
    that examined nothing must not report success.

Usage:
    python util/ad-hoc/2026-09-10_loss_check_changed_docs.py [--base HEAD]
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

from pathlib import Path

CHECKER = Path(__file__).with_name("2026-09-08_section_loss_check.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("section_loss_check", CHECKER)
    if not spec or not spec.loader:
        print(f"cannot load {CHECKER}", file=sys.stderr)
        raise SystemExit(2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="HEAD", help="base ref to compare against (default HEAD)")
    args = ap.parse_args(argv)

    changed = [p for p in _git("diff", "--name-only", args.base, "--", "*.md").split("\n") if p]
    if not changed:
        print(f"no markdown files changed against {args.base} -- nothing to screen", file=sys.stderr)
        return 2

    mod = _load_checker()
    # `atoms(text)` is exactly what the checker's own main() compares -- reuse it rather than
    # reimplement, so this screen cannot drift away from the tool it claims to run. Refuse
    # loudly if the name ever moves; a getattr fallback to None would screen nothing and
    # report zero losses, which is the vacuous pass this whole family of checks exists to
    # refuse.
    extract = getattr(mod, "atoms", None)
    if extract is None:
        print("loss checker has no atoms() -- refusing rather than screening nothing", file=sys.stderr)
        return 2

    losses = 0
    print(f"screening {len(changed)} changed markdown file(s) against {args.base}\n")
    for rel in changed:
        try:
            before_text = _git("show", f"{args.base}:{rel}")
        except subprocess.CalledProcessError:
            print(f"  {'ADDED':<9} {rel}  (no BEFORE -- nothing can be lost)")
            continue
        after = Path(rel)
        if not after.is_file():
            print(f"  {'DELETED':<9} {rel}  -- adjudicate the whole file", file=sys.stderr)
            losses += 1
            continue
        lost = sorted(extract(before_text) - extract(after.read_text(encoding="utf-8")))
        if lost:
            losses += 1
            print(f"  {'LOST ' + str(len(lost)):<9} {rel}")
            for item in lost[:10]:
                print(f"              {item[:88]}")
            if len(lost) > 10:
                print(f"              ... and {len(lost)-10} more")
        else:
            print(f"  {'ok':<9} {rel}")

    print(f"\n{losses} of {len(changed)} file(s) lost atoms")
    if losses:
        print("Each must be ADJUDICATED: a deliberate drop (say where it went) or a genuine loss.")
    return 1 if losses else 0


if __name__ == "__main__":
    raise SystemExit(main())
