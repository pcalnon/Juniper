#!/usr/bin/env python3
"""
Report, for every ``pkg==version`` pin in a requirements lock, whether PyPI serves a wheel
that a given CPython tag can install on a given platform without a compiler.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-08
Status: ad-hoc — investigation
Retire when: every image-bearing repo's publish-image.yml has built arm64 at least once on
             a real release (the PR arm then IS the wheel check, and this pre-flight is moot).
Related: juniper-ml prompts/thread-handoff_automated-prompts/
         HANDOFF_2026-09-07_container-registry-rollout-wave-1-complete.md item 3 ("arm64
         evidence does not transfer to recurrence -- re-check cp313 aarch64 wheels").

Why: the container images are ``python:3.14-slim`` (recurrence: ``3.13-slim``) -- no gcc, no
Rust -- so a pinned native-extension package with NO matching binary wheel is a hard build
failure on that arch, discovered only when the arm64 runner tries. Pure-Python packages
(``py3-none-any``) and stable-ABI wheels (``abi3``) are fine on any arch. This script asks PyPI
for the pinned release's file list and classifies each pin BEFORE a 15-minute CI cycle does.

Usage::

    python3 util/ad-hoc/2026-09-08_lock_wheel_availability.py LOCKFILE --python 3.14 --arch aarch64
    python3 util/ad-hoc/2026-09-08_lock_wheel_availability.py LOCKFILE --python 3.13 --arch aarch64 --also x86_64

Exit 0 when every pin has an installable wheel (or is pure Python); exit 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

PIN_RE = re.compile(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s;#]+)")


def parse_lock(path: str) -> list[tuple[str, str]]:
    pins = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith((" ", "\t", "#")) or not line.strip():
                continue
            match = PIN_RE.match(line)
            if match:
                name = re.sub(r"\[.*\]", "", match.group(1))
                pins.append((name, match.group(2)))
    return pins


def release_files(name: str, version: str) -> list[str]:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310 - fixed https host
        data = json.load(resp)
    return [f["filename"] for f in data["urls"]]


def classify(files: list[str], py_tag: str, arch: str) -> tuple[str, str]:
    """Return (status, evidence) for one pinned release."""
    wheels = [f for f in files if f.endswith(".whl")]
    if not wheels:
        return ("SDIST-ONLY", "no wheels at all; needs a compiler if it has C extensions")
    pure = [w for w in wheels if "-none-any.whl" in w]
    if pure:
        return ("PURE", pure[0])
    for w in wheels:
        if arch in w and ("manylinux" in w or "musllinux" not in w) and (f"-{py_tag}-" in w or "-abi3-" in w):
            return ("WHEEL", w)
    abi3_other = [w for w in wheels if "-abi3-" in w and arch in w]
    if abi3_other:
        return ("WHEEL", abi3_other[0])
    same_arch = sorted({w.split("-")[2] for w in wheels if arch in w})
    return ("MISSING", f"wheels for {arch} exist only for {same_arch or 'no python tag'}; all: {sorted({w.split('-')[2] for w in wheels})}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lockfile")
    ap.add_argument("--python", default="3.14", help="CPython version, e.g. 3.14")
    ap.add_argument("--arch", default="aarch64")
    ap.add_argument("--also", default=None, help="a second arch to report alongside")
    args = ap.parse_args()

    py_tag = "cp" + args.python.replace(".", "")
    arches = [args.arch] + ([args.also] if args.also else [])
    pins = parse_lock(args.lockfile)
    print(f"{len(pins)} pins in {args.lockfile}; checking {py_tag} on {', '.join(arches)}")
    bad = 0
    for name, version in pins:
        try:
            files = release_files(name, version)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ERROR    {name}=={version}: {exc}")
            bad += 1
            continue
        for arch in arches:
            status, evidence = classify(files, py_tag, arch)
            flag = "  " if status in ("PURE", "WHEEL") else "!!"
            if status not in ("PURE", "WHEEL"):
                bad += 1
            if status != "PURE" or arch == arches[0]:
                print(f"{flag} {status:<10} {name}=={version} [{arch}] {evidence}")
    print(f"\n{'OK' if bad == 0 else 'PROBLEMS: ' + str(bad)} -- {len(pins)} pins")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
