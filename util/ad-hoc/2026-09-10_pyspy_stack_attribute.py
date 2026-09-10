#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml
Application: Performance lane -- PF-8 follow-up
Author:      Paul Calnon
License:     MIT License

Attribute a ``py-spy record --native -f raw`` folded-stack file to native threading libraries
and to the autograd engine, by SAMPLE COUNT rather than by line count.

Why by sample count: the folded format is ``<frame>;<frame>;... <count>``, so one line can carry
hundreds of samples. A ``grep -c`` over the file counts LINES and silently mis-weights the
answer. It also substring-matches -- "omp" hits "compiled", "component" and
"Compute" -- so the library patterns here are anchored to real ``lib*.so`` names.

Used to settle notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md
section 4.2: which library carries the initial output pass's multi-core burst.

    python3 <this script> --stacks /path/to/raw.txt
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

#: Anchored to shared-object names so a substring like "omp" in "compiled" cannot match.
LIBRARY_PATTERNS = {
    "libgomp (GNU OpenMP)": re.compile(r"libgomp"),
    "libiomp5 / libomp (LLVM/Intel OpenMP)": re.compile(r"lib(iomp5|omp)\.so"),
    "libopenblas / scipy_openblas": re.compile(r"(libopenblas|libscipy_openblas)"),
    "libmkl": re.compile(r"libmkl"),
    "libtorch_cpu": re.compile(r"libtorch_cpu"),
    "libtorch_python": re.compile(r"libtorch_python"),
}

SEMANTIC_PATTERNS = {
    "torch::autograd::Engine": re.compile(r"autograd::Engine"),
    "autograd ReadyQueue": re.compile(r"ReadyQueue"),
    "at::parallel_for / ATen threading": re.compile(r"(at::parallel_for|ATen.*[Tt]hread)"),
    "GOMP parallel region": re.compile(r"GOMP_parallel|gomp_thread"),
}

THREAD_RE = re.compile(r"^thread \((\d+)\)")
LINE_RE = re.compile(r"^(.*?)\s+(\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stacks", type=Path, required=True, help="py-spy raw folded-stack file")
    parser.add_argument("--top", type=int, default=10, help="how many threads to list")
    args = parser.parse_args()

    total = 0
    by_thread: collections.Counter = collections.Counter()
    by_library: collections.Counter = collections.Counter()
    by_semantic: collections.Counter = collections.Counter()
    thread_has_engine: set[str] = set()

    for raw in args.stacks.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        m = LINE_RE.match(raw.rstrip())
        if not m:
            continue
        stack, count = m.group(1), int(m.group(2))
        total += count
        tid_match = THREAD_RE.match(stack)
        tid = tid_match.group(1) if tid_match else "?"
        by_thread[tid] += count
        for name, pattern in LIBRARY_PATTERNS.items():
            if pattern.search(stack):
                by_library[name] += count
        for name, pattern in SEMANTIC_PATTERNS.items():
            if pattern.search(stack):
                by_semantic[name] += count
                if name == "torch::autograd::Engine":
                    thread_has_engine.add(tid)

    if total == 0:
        print("no samples parsed -- is this a py-spy '-f raw' file?")
        return 2

    print(f"total samples : {total}")
    print(f"distinct threads sampled : {len(by_thread)}")
    print(f"threads carrying autograd::Engine frames : {len(thread_has_engine)}")
    print("\nby native library (a stack may match several):")
    for name in LIBRARY_PATTERNS:
        got = by_library.get(name, 0)
        print(f"   {name:<40} {got:>6}  {100 * got / total:5.1f}%")
    print("\nby semantic marker:")
    for name in SEMANTIC_PATTERNS:
        got = by_semantic.get(name, 0)
        print(f"   {name:<40} {got:>6}  {100 * got / total:5.1f}%")
    print(f"\ntop {args.top} threads by samples:")
    for tid, count in by_thread.most_common(args.top):
        tag = "  [autograd engine]" if tid in thread_has_engine else ""
        print(f"   tid {tid:<10} {count:>6}  {100 * count / total:5.1f}%{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
