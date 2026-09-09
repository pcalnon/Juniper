#!/usr/bin/env python3
"""Append a GitHub-signed commit to an EXISTING branch, without opening a PR.

Project     : Juniper
Sub-Project : juniper-ml
Application : cross-repo tooling (ad-hoc)
Author      : Paul Calnon
License     : MIT License
Created     : 2026-09-08

Why this exists
---------------
`util/open_signed_pr.py` does branch + signed commit + PR in one shot, and its
DUP-GUARD returns 1 the moment an open PR already exists for the branch. That
guard is correct -- it stops a second PR being opened for the same branch -- but
it fires BEFORE the commit, so there is no supported way to add a follow-up
commit to a PR that tool opened. Signing locally is not an option in a headless
session, and an unsigned commit anywhere in a branch's history blocks the merge
under `required_signatures` (squash does not rescue it).

This reuses `open_signed_pr`'s own `create_signed_commit` against the branch's
current head, so the follow-up commit is signed by GitHub exactly like the first.

The `expected_head_oid` is read live rather than passed in: that is the
optimistic-concurrency token, so a concurrent push to the same branch makes this
fail loudly instead of clobbering. Note the addition is a WHOLE-FILE upload --
read the branch's current copy before editing, or you silently revert whatever
landed on it since.

THE ADDITIONS CONTRACT, AND THE BUG THAT MADE IT WORTH DOCUMENTING
------------------------------------------------------------------
`create_signed_commit` takes `additions` as a list of **2-tuples**
`(repo_path, base64_contents)` -- it does `for path, contents in additions`.

The first version of this script passed a list of **dicts**
(`{"path": ..., "contents": ...}`). Unpacking a two-key dict yields its KEYS, so
every commit silently wrote a file literally named `path` whose body was
`base64.b64decode("contents")` = `b'r\\x89\\xedz{l'` -- and **never uploaded the
real file at all**. It printed `signed commit <sha>` and looked like a success.
Nine branches took that commit before CI caught it, via `end-of-file-fixer`
failing on the stray binary rather than via anything that understood the problem.

That is why `--verify` exists and defaults ON: after committing, this re-reads
each path from the branch and compares bytes. A write tool that cannot confirm
its own write is a vacuous success waiting to happen.

Usage
-----
    python3 util/ad-hoc/append_signed_commit.py \\
        --repo juniper-ml --branch docs/some-branch \\
        --add notes/FOO.md:notes/FOO.md \\
        --delete path \\
        --message "docs: follow-up" --commit-body-file /tmp/body.md
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess  # nosec B404 - shells out to the authenticated `gh` CLI only
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_signed_pr import create_signed_commit, parse_add  # noqa: E402


def _gh_json(*args: str) -> dict:
    out = subprocess.run(  # nosec B603 B607 - fixed argv, authenticated gh CLI
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def branch_head_oid(owner: str, repo: str, branch: str) -> str:
    """Live head OID of ``branch`` -- the optimistic-concurrency token."""
    return _gh_json("api", f"repos/{owner}/{repo}/git/ref/heads/{branch}")["object"]["sha"]


class ReadError(Exception):
    """The branch could not be read -- distinct from 'the content differs'."""


def blob_at(owner: str, repo: str, ref: str, repo_path: str) -> bytes | None:
    """Raw bytes of ``repo_path`` at ``ref``; None when genuinely ABSENT (404).

    ``ref`` goes in the QUERY STRING, not through ``-f``: ``gh api`` switches the
    request to POST as soon as any ``-f`` parameter is supplied, and the contents
    endpoint rejects that. The first version of this verifier did exactly that, so
    every read failed and every path was reported as content-mismatch -- a checker
    that cries wolf on a commit that was in fact correct.

    A 404 is a real absence and returns None. Anything else raises, because
    "I could not look" must never be reported as "the content is wrong".
    """
    proc = subprocess.run(  # nosec B603 B607 - fixed argv, authenticated gh CLI
        ["gh", "api", "-X", "GET", f"repos/{owner}/{repo}/contents/{repo_path}?ref={ref}", "--jq", ".content"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if "404" in proc.stderr or "Not Found" in proc.stderr:
            return None
        raise ReadError(f"could not read {repo_path}@{ref}: {proc.stderr.strip()[:200]}")
    return base64.b64decode(proc.stdout.strip())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default="pcalnon")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--add", action="append", default=[], metavar="LOCAL:REPOPATH")
    ap.add_argument("--delete", action="append", default=[], metavar="REPOPATH")
    ap.add_argument("--message", required=True)
    ap.add_argument("--commit-body-file")
    ap.add_argument("--no-verify", action="store_true", help="skip the post-commit read-back (do not)")
    args = ap.parse_args(argv)

    if not args.add and not args.delete:
        ap.error("nothing to do: pass --add and/or --delete")

    # (repo_path, base64_contents) TUPLES -- see the contract note in the docstring.
    additions: list[tuple[str, str]] = []
    wanted: dict[str, bytes] = {}
    for spec in args.add:
        local, repo_path = parse_add(spec)
        raw = Path(local).read_bytes()
        additions.append((repo_path, base64.b64encode(raw).decode("ascii")))
        wanted[repo_path] = raw

    body = Path(args.commit_body_file).read_text(encoding="utf-8") if args.commit_body_file else None
    head = branch_head_oid(args.owner, args.repo, args.branch)
    print(f"branch head is {head[:12]}; +{len(additions)} file(s), -{len(args.delete)} file(s)")

    sha = create_signed_commit(
        args.owner,
        args.repo,
        args.branch,
        args.message,
        additions,
        head,
        deletions=args.delete or None,
        commit_body=body,
    )
    print(f"signed commit {sha[:12]} on {args.owner}/{args.repo}:{args.branch}")

    if args.no_verify:
        return 0

    bad = 0
    for repo_path, raw in wanted.items():
        try:
            got = blob_at(args.owner, args.repo, args.branch, repo_path)
        except ReadError as exc:
            print(f"  VERIFY INCONCLUSIVE {repo_path}: {exc}", file=sys.stderr)
            bad += 1
            continue
        if got is None:
            print(f"  VERIFY FAIL {repo_path}: ABSENT from the branch after the commit", file=sys.stderr)
            bad += 1
        elif got != raw:
            print(f"  VERIFY FAIL {repo_path}: branch has {len(got)} bytes, expected {len(raw)}", file=sys.stderr)
            bad += 1
        else:
            print(f"  verified {repo_path} ({len(raw)} bytes)")
    for repo_path in args.delete:
        try:
            still = blob_at(args.owner, args.repo, args.branch, repo_path)
        except ReadError as exc:
            print(f"  VERIFY INCONCLUSIVE {repo_path}: {exc}", file=sys.stderr)
            bad += 1
            continue
        if still is not None:
            print(f"  VERIFY FAIL {repo_path}: still present after deletion", file=sys.stderr)
            bad += 1
        else:
            print(f"  verified {repo_path} is gone")

    if bad:
        print(f"VERIFY FAILED for {bad} path(s) -- the commit landed but its CONTENT is wrong", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
