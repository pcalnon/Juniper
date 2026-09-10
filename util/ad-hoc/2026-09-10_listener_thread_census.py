#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml
Application: Performance lane -- PF-8 follow-up
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

WHAT THIS IS
------------
An EXTERNAL per-thread CPU census of a running process, used to attribute the ~11-core burst of
the cascor listener's initial output pass
(``notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md`` §4.2) to a
thread pool of a known WIDTH.

WHY WIDTH AND NOT NAME
----------------------
Thread names do not discriminate the two candidate pools in this build: a validated burn of
NumPy's OpenBLAS pool (``2026-09-10_first_pass_library_attribution.py --synthetic numpy``)
produced 12.65 cores across 16 threads, every one of them named ``python``. Width does
discriminate, because the pools have different maxima on this 16-core host:

  * torch intra-op, unpinned default ....... 8 threads  -> at most ~8 cores
  * torch intra-op, cascor's parent pin .... 2 threads  -> at most ~2 cores
  * NumPy / OpenBLAS, unpinned ............ 16 threads  -> ~12.6 cores measured

So a sustained burn above 8 cores cannot be carried by torch's intra-op pool at either setting,
and counting the threads that are actually burning in a given interval says which pool is live.

WHY EXTERNAL
------------
``/proc/<pid>/task/`` is readable for the user's own processes and needs no ptrace. That matters
here: ``/proc/sys/kernel/yama/ptrace_scope`` is 1 on this host, so py-spy can only attach to its
own descendants and cannot be pointed at a listener that was started by the launcher under
``nohup``. This census has no such restriction.

USAGE
-----
    python3 <this script> --pid <listener pid> --duration 120 --interval 0.5 \
        --out /path/census.json

Sampling is cheap (one small read per thread per tick) but is NOT free: it is reported in the
output as ``sampler_overhead_note`` so a reader can discount it. The census attributes only the
NAMED process, not its children -- a candidate-pool worker is a separate process and does not
appear here, which is deliberate: the burst under investigation is the listener's own, before
any pool exists.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


def read_threads(pid: int) -> dict[int, tuple[str, float]]:
    """{tid: (comm, cpu_seconds)} for every thread of ``pid``; empty if the process is gone."""
    out: dict[int, tuple[str, float]] = {}
    task_dir = Path(f"/proc/{pid}/task")
    try:
        entries = list(task_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue  # thread exited between listing and reading
        close = raw.rfind(")")
        open_paren = raw.find("(")
        if close < 0 or open_paren < 0:
            continue
        try:
            tid = int(raw[:open_paren].strip())
        except ValueError:
            continue
        comm = raw[open_paren + 1 : close]
        fields = raw[close + 2 :].split()
        if len(fields) < 13:
            continue
        try:
            cpu = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
        except ValueError:
            continue
        out[tid] = (comm, cpu)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, required=True, help="pid to census (the cascor listener)")
    parser.add_argument("--duration", type=float, default=120.0, help="seconds to sample")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between ticks")
    parser.add_argument("--burn-threshold", type=float, default=0.30, help="cores above which a thread counts as burning in an interval")
    parser.add_argument("--out", type=Path, required=True, help="write the census JSON here")
    args = parser.parse_args()

    if not Path(f"/proc/{args.pid}").exists():
        print(f"REFUSED: no such pid {args.pid}")
        return 2

    samples: list[dict] = []
    end = time.time() + args.duration
    while time.time() < end:
        threads = read_threads(args.pid)
        if not threads:
            break  # process exited -- stop rather than record empty ticks
        samples.append({"wall": time.time(), "threads": threads})
        time.sleep(args.interval)

    series = []
    for prev, cur in zip(samples, samples[1:]):
        dt = cur["wall"] - prev["wall"]
        if dt <= 0:
            continue
        total = 0.0
        burning = 0
        per_thread = []
        for tid, (comm, cpu) in cur["threads"].items():
            start = prev["threads"].get(tid)
            if start is None:
                continue  # thread appeared this interval; no baseline to difference against
            cores = (cpu - start[1]) / dt
            total += cores
            if cores > args.burn_threshold:
                burning += 1
                per_thread.append({"tid": tid, "comm": comm, "cores": round(cores, 3)})
        per_thread.sort(key=lambda r: -r["cores"])
        series.append(
            {
                "t": round(cur["wall"] - samples[0]["wall"], 2),
                "cores": round(total, 3),
                "threads_alive": len(cur["threads"]),
                "threads_burning": burning,
                "top": per_thread[:20],
            }
        )

    peak = max(series, key=lambda s: s["cores"], default=None)
    result = {
        "pid": args.pid,
        "interval": args.interval,
        "burn_threshold": args.burn_threshold,
        "samples": len(samples),
        "loadavg_at_end": os.getloadavg(),
        "peak_interval": peak,
        "max_threads_burning": max((s["threads_burning"] for s in series), default=None),
        "max_threads_alive": max((s["threads_alive"] for s in series), default=None),
        "sampler_overhead_note": (
            "this sampler runs in its own process; its CPU is NOT included in the figures above, "
            "which cover only the censused pid"
        ),
        "series": series,
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"censused pid {args.pid}: {len(samples)} samples over {args.duration}s at {args.interval}s")
    print(f"peak interval: {peak['cores'] if peak else None} cores, {peak['threads_burning'] if peak else None} threads burning (of {peak['threads_alive'] if peak else None} alive)")
    print(f"max threads burning in any interval: {result['max_threads_burning']}")
    print("first 24 intervals (t / cores / burning):")
    for s in series[:24]:
        print(f"   t={s['t']:>6}  cores={s['cores']:>7}  burning={s['threads_burning']:>3}  alive={s['threads_alive']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
