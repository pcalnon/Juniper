#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# Created     : 2026-09-09
# Status      : single-use (Phase 5 provenance proof)
# Retire when : the :8202 leg is relaunched from a clean juniper-cascor main
# Related     : notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md (Phase 5),
#               juniper-cascor#632
# ---------------------------------------------------------------------------
"""Does the cascor leg that served window 3 run the code that merged in #632?

THE PROBLEM. The `:8202` leg reports `git_sha d39d537e…` on `/v1/health`, because
the stamp is `git rev-parse HEAD` and the leg runs a worktree whose FIX IS
UNCOMMITTED ON TOP OF THAT BASE. `d39d537e…` is a dependency bump; it contains
none of the fix. So the arc's serving-commit stamp -- the thing Phase 5 says
closes still-owed item 7 -- names, for this leg, a commit that is NOT what ran.

THE PROOF THAT ACTUALLY WORKS. Compare the CONTENT of the served files against
the content GitHub holds at the merge commit. This asks GitHub for each blob at
the merge sha and sha256s it against the file on disk in the running tree; no git
command is issued against the sibling repository (this session is worktree-isolated).

Usage:
    python3 util/ad-hoc/2026-09-09_verify_served_cascor_matches_merge.py \\
        [--merge 5eb6f14497b332b0c7d473aba557a5b6d8d5f8a6]
"""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TREE = Path(
    "/home/pcalnon/Development/python/Juniper/worktrees/"
    "juniper-cascor--fix--snapshot-restore-seed-numpy-scalar--20260908-0725--d39d5370"
)
FILES = [
    "src/snapshots/snapshot_common.py",
    "src/snapshots/snapshot_serializer.py",
    "src/tests/unit/test_snapshot_serializer.py",
]


def gh_blob(repo: str, sha: str, path: str) -> bytes:
    # The ref MUST ride in the query string: `-f` makes `gh api` issue a POST.
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/{path}?ref={sha}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return base64.b64decode(json.loads(out)["content"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", default="5eb6f14497b332b0c7d473aba557a5b6d8d5f8a6", help="the merge commit of juniper-cascor#632")
    ap.add_argument("--repo", default="pcalnon/juniper-cascor")
    ap.add_argument("--base", default="d39d537e985ba329290142c123f576b4ece9dba3", help="the sha /v1/health reports")
    args = ap.parse_args()

    print(f"served tree : {TREE}")
    print(f"merge commit: {args.merge}")
    try:
        subj = subprocess.run(
            ["gh", "api", f"repos/{args.repo}/commits/{args.base}", "--jq", ".commit.message"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
        print(f"the sha /v1/health reports ({args.base[:12]}…) is: {subj!r}")
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        print(f"could not resolve base sha: {exc}")

    all_same = True
    print()
    for rel in FILES:
        local = (TREE / rel).read_bytes()
        remote = gh_blob(args.repo, args.merge, rel)
        lh, rh = hashlib.sha256(local).hexdigest(), hashlib.sha256(remote).hexdigest()
        same = lh == rh
        all_same &= same
        print(f"{'MATCH  ' if same else 'DIFFER '} {rel}")
        print(f"         served {lh}")
        print(f"         merged {rh}")

    print()
    if all_same:
        print("VERDICT: the running leg's fix files are BYTE-IDENTICAL to what merged in #632.")
        print("         The /v1/health git_sha does NOT prove this and names a commit without the fix;")
        print("         this content comparison is the provenance record for window 3.")
    else:
        print("VERDICT: the running leg DIFFERS from what merged — every window-3 verdict needs re-reading.")
    return 0 if all_same else 1


if __name__ == "__main__":
    sys.exit(main())
