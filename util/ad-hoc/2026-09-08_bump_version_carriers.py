#!/usr/bin/env python3
"""Copy listed files out of a checkout with ONE exact string replaced, and count the hits.

Project:     Juniper
Sub-Project: juniper-ml
Application: cross-repo tooling (ad-hoc)
Author:      Paul Calnon
Created:     2026-09-08
Status:      ad-hoc (decision-11 release train)

Why this exists
---------------
A release bump has more carriers than ``pyproject.toml``: juniper-data-client pins twelve
``Version: x.y.z`` file headers to ``__version__`` (``tests/test_file_header_versions.py``),
plus three docs headers; juniper-data and juniper-canopy keep a fallback literal in
``__init__.py``. The release train's ``propose.py`` does not touch those, so they are landed
as a follow-up signed commit built from edited copies of the files. A ``sed -i`` over a
list is the obvious tool and the wrong one here: it edits the primary checkout in place
(which must stay a clean read-only mirror of ``origin/main`` for the train), and it is
silent when a pattern matches zero times or twice. This script never touches the source
tree -- it writes each edited copy under ``--out-root`` at the same relative path -- and
it FAILS (exit 1) unless every file matched ``--from`` exactly ``--expect`` times, which
is how a renamed header or a stale file list surfaces before the commit does.

Usage
-----
    python3 util/ad-hoc/2026-09-08_bump_version_carriers.py \
        --src-root /path/to/checkout --out-root /scratch/out \
        --from "Version: 0.4.2" --to "Version: 0.5.0" [--expect 1] \
        rel/path/one.py rel/path/two.py ...

Prints one ``<count>  <relpath>`` line per file. Exit 0 = every file matched as expected,
1 = a count mismatch (nothing is written for that file), 2 = usage / IO error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description="Edit copies of version-carrier files with an exact-count guard")
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--expect", type=int, default=1, help="required number of matches per file (default 1)")
    ap.add_argument("files", nargs="+", help="paths relative to --src-root")
    args = ap.parse_args(argv)

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    if not src_root.is_dir():
        print(f"error: --src-root {src_root} is not a directory", file=sys.stderr)
        return 2

    bad = 0
    for rel in args.files:
        src = src_root / rel
        try:
            text = src.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {src}: {exc}", file=sys.stderr)
            return 2
        count = text.count(args.old)
        print(f"{count:>3}  {rel}")
        if count != args.expect:
            bad += 1
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text.replace(args.old, args.new), encoding="utf-8")

    if bad:
        print(f"refused: {bad} file(s) did not match {args.old!r} exactly {args.expect} time(s); their copies were NOT written", file=sys.stderr)
        return 1
    print(f"wrote {len(args.files)} edited copies under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
