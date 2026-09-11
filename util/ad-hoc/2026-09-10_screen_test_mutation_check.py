#!/usr/bin/env python3
"""2026-09-10_screen_test_mutation_check.py -- do the new screen tests discriminate?

Project: juniper-ml
Sub-Project: fleet triage / Cursor-fleet PR-flood remediation (round 2)
Application: ad-hoc validation (test quality)
Author: Paul Calnon
License: MIT License

WHY THIS EXISTS

2026-09-10 relaxed `util/ad-hoc/2026-09-05_markdown_structure_check.py` in two ways -- the
table SEPARATOR now accepts GFM's single hyphen, and the fence walk follows CommonMark's
character/length/info-string rules instead of a boolean toggle -- and added tests for both.

A test written against already-fixed code proves nothing: it passes before the fix too. This
runs the new tests' exact fixtures through the PRE-FIX screen, which must FAIL on each one.

Point `--old` at the previous revision of the screen, e.g.

    git show HEAD:util/ad-hoc/2026-09-05_markdown_structure_check.py > /tmp/old_screen.py

EXIT CODES

  * 0 -- every case discriminates (fails against the old code, clean against the new);
  * 1 -- at least one case is VACUOUS: it passes under the old code, so it tests nothing;
  * 2 -- could not run (missing --old, unreadable module).

Usage:
    python util/ad-hoc/2026-09-10_screen_test_mutation_check.py --old /tmp/old_screen.py
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import tempfile

# The exact fixtures asserted to yield [] by tests/test_markdown_structure_screen.py --
# SeparatorAcceptsValidGfmDelimitersTest and FenceWalkerFollowsCommonMarkTest.
CASES = {
    "single_hyphen_delimiter": "# T\n\n| A | B |\n| - | - |\n| 1 | 2 |\n",
    "alignment_colons": "# T\n\n| C | D |\n|:-:|:-:|\n| 3 | 4 |\n",
    "info_string_cannot_close": "# T\n\n````markdown\n```bash\necho hi\n```\n## Sample heading\n````\n",
    "short_cannot_close_longer": "# T\n\n````markdown\n## A\n```\n## B\n````\n",
    "tilde_vs_backtick": "# T\n\n~~~\n```\n~~~\n\nprose\n",
}

# Behaviour the OLD code did not model AT ALL, so it was clean by omission rather than by
# being right. "Old is clean" therefore proves nothing here, and a bare exemption assertion
# would be indistinguishable from the screen ignoring tilde fences -- which is what the old
# one did. Each entry pairs the exemption with a POSITIVE control that must still fire under
# the new screen; that pair is what gives the exemption meaning.
NEW_BEHAVIOUR = {
    "tilde_markdown_exempt": ("# T\n\n~~~markdown\n## Sample heading\n~~~\n", 0),
    "tilde_bash_still_swallows": ("# T\n\n~~~bash\n## Swallowed\n~~~\n", 1),
}

# Negative controls: the screen must STILL report these, before and after.
CONTROLS = {
    "missing_separator": "# Title\n\nprose\n\n| a | b |\n| 1 | 2 |\n",
    "really_unclosed": "# T\n\n```bash\necho hi\n",
    "h2_swallowed": "# T\n\n```bash\necho hi\n\n## Swallowed\n",
}

NEW = pathlib.Path(__file__).with_name("2026-09-05_markdown_structure_check.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        print(f"cannot load {path}", file=sys.stderr)
        raise SystemExit(2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _check(mod, body, td):
    p = pathlib.Path(td, "t.md")
    p.write_text(body, encoding="utf-8")
    return mod.check(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True, help="path to the PRE-FIX screen")
    args = ap.parse_args(argv)

    old = _load(args.old, "old_screen")
    new = _load(NEW, "new_screen")

    vacuous, broken_controls = [], []
    with tempfile.TemporaryDirectory() as td:
        print("CASES -- must be clean under NEW, and must FAIL under OLD:\n")
        for name, body in CASES.items():
            o, n = _check(old, body, td), _check(new, body, td)
            ok = bool(o) and not n
            if not ok:
                vacuous.append(name)
            print(f"  {name:28} old={len(o)} new={len(n)}  {'discriminates' if ok else 'VACUOUS'}")
            for f in o:
                print(f"        old said: {f[:88]}")

        print("\nNEW BEHAVIOUR -- old code silent by OMISSION; the pair is the evidence:\n")
        for name, (body, want) in NEW_BEHAVIOUR.items():
            o, n = _check(old, body, td), _check(new, body, td)
            ok = (len(n) == 0) if want == 0 else (len(n) >= 1)
            if not ok:
                broken_controls.append(name)
            note = "exempt" if want == 0 else "must fire"
            print(f"  {name:28} old={len(o)} new={len(n)}  {note}: {'ok' if ok else 'WRONG'}")

        print("\nCONTROLS -- must be reported by BOTH (the fix must not blind the screen):\n")
        for name, body in CONTROLS.items():
            o, n = _check(old, body, td), _check(new, body, td)
            ok = bool(o) and bool(n)
            if not ok:
                broken_controls.append(name)
            print(f"  {name:28} old={len(o)} new={len(n)}  {'held' if ok else 'REGRESSED'}")

    if vacuous or broken_controls:
        if vacuous:
            print(f"\nVACUOUS case(s): {', '.join(vacuous)}", file=sys.stderr)
        if broken_controls:
            print(f"REGRESSED control(s): {', '.join(broken_controls)}", file=sys.stderr)
        return 1
    print(f"\nall {len(CASES)} case(s) discriminate; all {len(CONTROLS)} control(s) held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
