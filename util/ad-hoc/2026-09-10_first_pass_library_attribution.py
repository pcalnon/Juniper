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
The discriminating test that
``notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md`` §4.2 leaves
open: which library carries the ~11-core burst of cascor's *initial* output-layer pass, and
whether ``torch.get_num_threads()`` reads 2 inside the process while that burst runs.

The probe note could not discriminate because both candidate mechanisms bracket the observed
magnitude (unpinned torch at the real op shape = 7.5 cores, unpinned NumPy matmul = 11.5,
observed = 10.6-11.4), and because the probes that produced those two numbers were *hand-built
loops*, not cascor's real ``train_output_layer`` on a real constructed network. Trap 11 of
``prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-10_perf-lane-pf8-located-pair-run-runtime-block-binds-nothing.md``:
"A magnitude match is not an attribution ... the probe that 'confirmed' one of them was run at an
op shape the workload never uses."

So this runs the REAL method, on a REAL constructed network, and attributes the CPU burn
*per thread*, by thread name, from ``/proc/self/task/``. Attribution, not magnitude matching.

WHAT IT MEASURES
----------------
1. ``torch.get_num_threads()`` at import, after config construction, and after network
   construction -- the parent pin (``cascade_correlation.py:617`` -> ``:1179-1180``,
   ``max(2, worker_thread_count * 2)``) is applied in the constructor, so this is the direct
   read of "is the pin effective in the process running the pass".
2. Which native threading runtimes are loaded (``/proc/self/maps``: libgomp / libiomp5 /
   libopenblas / libmkl), so a burn can be attributed to a runtime that is actually present.
3. Per-thread cpu-seconds over the pass, by thread name, from
   ``/proc/self/task/<tid>/stat`` (utime+stime, fields 14/15) and ``/proc/self/task/<tid>/comm``.
   This is the discriminator: OpenBLAS names its pool threads, and a pool that burns cores
   shows up as N threads each burning ~1 core-second per wall-second.
4. Total process occupancy in cores over the pass, for comparison against the note's 10.6-11.4.

The sampler is a Python thread inside the same process; its own burn is reported separately and
is subtracted from nothing -- it is shown so the reader can discount it.

HOW TO RUN
----------
The three BLAS variables are read once at library load, so pinning is the CALLER's job::

    # unpinned arm (the service's default -- configure_blas_threads() is a no-op by default)
    env -C /home/pcalnon/Development/python/Juniper/juniper-cascor/src \
        /opt/miniforge3/envs/JuniperCascor1/bin/python \
        <this script> --arm unpinned --json /path/out.json

    # pinned arm
    env -C /home/pcalnon/Development/python/Juniper/juniper-cascor/src \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        /opt/miniforge3/envs/JuniperCascor1/bin/python \
        <this script> --arm pinned --json /path/out.json

``--arm`` is a LABEL ONLY -- it records what the caller claims to have exported and the script
verifies it against ``os.environ``; it never sets the variables itself (setting them after
import would be a silent no-op and would make the arm a lie).

WHAT IT DOES NOT DO
-------------------
- It does not drive the cascor SERVICE. It runs the same trainer object the service constructs
  (``api/lifecycle/manager.py:1578`` -> ``CascadeCorrelationNetwork(config=config)``) in a
  single process. If the burst reproduces here, the mechanism is in the trainer and not in the
  listener's plumbing; if it does NOT reproduce, that is itself the finding and the listener
  must be profiled instead.
- The spiral data is generated locally to the cell's parameters (2 spirals, 200 points each,
  2 rotations, noise 0.05, train_ratio 0.8 -> 320 rows), NOT fetched from juniper-data. The
  attribution question is about which thread pool burns cores at a given op shape; the shape and
  dtype are what matter and both are reproduced exactly. Do not read this script's wall-clock
  timings as run-tier figures -- they are not comparable to a suite cell.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

CLOCK_TICKS = os.sysconf("SC_CLK_TCK")

#: Native threading runtimes worth naming if they are mapped into the process.
_RUNTIME_PATTERNS = {
    "libgomp": re.compile(r"libgomp[^/\s]*\.so"),
    "libiomp5": re.compile(r"libiomp5[^/\s]*\.so"),
    "libomp": re.compile(r"/libomp[^/\s]*\.so"),
    "libopenblas": re.compile(r"(libopenblas|libscipy_openblas)[^/\s]*\.so"),
    "libmkl": re.compile(r"libmkl_[^/\s]*\.so"),
    "libtorch_cpu": re.compile(r"libtorch_cpu[^/\s]*\.so"),
}


def loaded_threading_runtimes() -> dict[str, str]:
    """Return {runtime_name: first matching mapped path} from /proc/self/maps."""
    found: dict[str, str] = {}
    try:
        text = Path("/proc/self/maps").read_text(errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        # the mapped path is the last field when present
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        for name, pattern in _RUNTIME_PATTERNS.items():
            if name not in found and pattern.search(path):
                found[name] = path
    return found


def read_thread_cpu() -> dict[int, tuple[str, float]]:
    """{tid: (comm, cpu_seconds)} for every thread of this process.

    ``/proc/<tid>/stat`` field 14 is utime and 15 is stime, in clock ticks. The comm field is
    parenthesised and may itself contain spaces and parentheses, so it is stripped by finding
    the LAST ')' rather than by splitting on whitespace.
    """
    out: dict[int, tuple[str, float]] = {}
    task_dir = Path("/proc/self/task")
    try:
        tids = list(task_dir.iterdir())
    except OSError:
        return out
    for entry in tids:
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue  # thread exited between listing and reading -- expected, not an error
        close = raw.rfind(")")
        if close < 0:
            continue
        try:
            tid = int(raw[: raw.index("(")].strip())
        except ValueError:
            continue
        comm = raw[raw.index("(") + 1 : close]
        fields = raw[close + 2 :].split()
        # after the comm and state fields, index 11 is utime and 12 is stime (0-based here)
        if len(fields) < 13:
            continue
        try:
            cpu = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS
        except ValueError:
            continue
        out[tid] = (comm, cpu)
    return out


def process_cpu_seconds() -> float:
    """Total cpu-seconds (user+sys) for this process, from /proc/self/stat."""
    raw = Path("/proc/self/stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    return (int(fields[11]) + int(fields[12])) / CLOCK_TICKS


class ThreadSampler:
    """Samples per-thread CPU and the process total on a fixed interval."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []
        self.sampler_tid: int | None = None

    def _run(self) -> None:
        self.sampler_tid = threading.get_native_id()
        while not self._stop.is_set():
            self.samples.append(
                {
                    "wall": time.time(),
                    "proc_cpu": process_cpu_seconds(),
                    "threads": read_thread_cpu(),
                }
            )
            self._stop.wait(self.interval)

    def __enter__(self) -> "ThreadSampler":
        self._thread = threading.Thread(target=self._run, name="pf8-sampler", daemon=True)
        self._thread.start()
        # let the first sample land before the workload starts
        time.sleep(self.interval)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.samples.append(
            {"wall": time.time(), "proc_cpu": process_cpu_seconds(), "threads": read_thread_cpu()}
        )

    def attribute(self) -> dict:
        """Cores over the sampled window, in total and per thread."""
        if len(self.samples) < 2:
            return {"error": "not enough samples"}
        first, last = self.samples[0], self.samples[-1]
        wall = last["wall"] - first["wall"]
        if wall <= 0:
            return {"error": "zero window"}
        per_thread = []
        seen = set(first["threads"]) | set(last["threads"])
        for tid in seen:
            start_comm, start_cpu = first["threads"].get(tid, ("<started>", 0.0))
            end_comm, end_cpu = last["threads"].get(tid, (None, None))
            if end_cpu is None:
                # thread exited mid-window: its final CPU is unknown, so report it as such
                per_thread.append(
                    {"tid": tid, "comm": start_comm, "cores": None, "note": "exited mid-window"}
                )
                continue
            per_thread.append(
                {
                    "tid": tid,
                    "comm": end_comm or start_comm,
                    "cores": round((end_cpu - start_cpu) / wall, 4),
                    "is_sampler": tid == self.sampler_tid,
                }
            )
        per_thread.sort(key=lambda r: (r["cores"] is None, -(r["cores"] or 0.0)))
        # Per-interval series. Trap 10 of the 2026-09-10 handoff: a mean over a bimodal window
        # is not the window's answer -- a burst that is one contiguous block at the start is a
        # PHASE, not a mixture, and only the series shows which it is.
        series = []
        for prev, cur in zip(self.samples, self.samples[1:]):
            dt = cur["wall"] - prev["wall"]
            if dt <= 0:
                continue
            burning = 0
            for tid, (_, cpu) in cur["threads"].items():
                if tid == self.sampler_tid:
                    continue
                start = prev["threads"].get(tid)
                if start is not None and (cpu - start[1]) / dt > 0.30:
                    burning += 1
            series.append(
                {
                    "t": round(cur["wall"] - self.samples[0]["wall"], 2),
                    "cores": round((cur["proc_cpu"] - prev["proc_cpu"]) / dt, 3),
                    "threads_over_0p30": burning,
                }
            )
        total = (last["proc_cpu"] - first["proc_cpu"]) / wall
        sampler = sum(r["cores"] or 0.0 for r in per_thread if r.get("is_sampler"))
        return {
            "window_seconds": round(wall, 3),
            "process_cores": round(total, 4),
            "sampler_cores": round(sampler, 4),
            "process_cores_excl_sampler": round(total - sampler, 4),
            "thread_count_end": len(last["threads"]),
            "threads_burning_over_0p05": sum(
                1 for r in per_thread if (r["cores"] or 0.0) > 0.05 and not r.get("is_sampler")
            ),
            "peak_interval_cores": max((s["cores"] for s in series), default=None),
            "peak_interval_threads": max((s["threads_over_0p30"] for s in series), default=None),
            "series": series,
            "per_thread": per_thread,
        }


def openblas_pool_size() -> int | None:
    """Ask NumPy's bundled OpenBLAS for its pool size, if it is reachable via ctypes."""
    for name, path in loaded_threading_runtimes().items():
        if name != "libopenblas":
            continue
        try:
            lib = ctypes.CDLL(path)
            fn = getattr(lib, "openblas_get_num_threads64_", None) or getattr(
                lib, "openblas_get_num_threads", None
            )
            if fn is None:
                return None
            fn.restype = ctypes.c_int
            return int(fn())
        except (OSError, AttributeError):
            return None
    return None


def make_spiral(n_spirals: int, n_points: int, n_rotations: float, noise: float, seed: int, radius_scale: float = 10.0):
    """A two-spiral set at the cell's parameters. Local, not a juniper-data artifact.

    ``radius_scale`` defaults to 10.0 because that is what juniper-data's SpiralParams and
    cascor's own ``spiral_problem`` both default to, and because a UNIT-radius spiral is
    degenerate for candidate training: cascor's own generator docstring
    (``api/routes/training.py``, ``_generate_spiral``) records that at unit scale every tanh
    candidate sits in its linear regime, best-of-pool correlation pins at ~2.7e-4 and
    ``grow_network`` terminates ``below_threshold`` with ZERO hidden units. A probe built on
    the unit spiral would measure a growth phase that never happens.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for spiral in range(n_spirals):
        t = np.linspace(0.0, 1.0, n_points)
        angle = 2.0 * np.pi * n_rotations * t + (2.0 * np.pi * spiral / n_spirals)
        radius = t * radius_scale
        x = radius * np.cos(angle) + rng.normal(0.0, noise, n_points)
        y = radius * np.sin(angle) + rng.normal(0.0, noise, n_points)
        xs.append(np.stack([x, y], axis=1))
        label = np.zeros((n_points, n_spirals), dtype=np.float32)
        label[:, spiral] = 1.0
        ys.append(label)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        choices=("unpinned", "pinned"),
        required=True,
        help="LABEL for what the caller exported; verified against os.environ, never set here.",
    )
    parser.add_argument("--epochs", type=int, default=4000, help="epochs for the initial output pass (default 4000 -- the cell's output_epochs; train_output_layer does NOT early-stop, and its INFO line every epoch_display_frequency=10 epochs is why the listener log shows 400 lines)")
    parser.add_argument(
        "--synthetic",
        choices=("numpy", "torch"),
        default=None,
        help=(
            "INSTRUMENT VALIDATION, not the workload: instead of the cascor pass, burn a known "
            "pool (a 1500^2 float32 matmul loop) so the /proc thread census can be checked "
            "against a pool whose identity is known in advance. Use to establish whether thread "
            "NAMES discriminate OpenBLAS from OpenMP in this build before trusting a census."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("output_pass", "fit"),
        default="output_pass",
        help="output_pass = the initial output-layer pass alone (the window the burst was seen in); fit = the whole real training path including the growth loop and candidate pool",
    )
    parser.add_argument("--max-iterations", type=int, default=10, help="max_iterations for --mode fit (default 10, the cell's value)")
    parser.add_argument(
        "--on-thread",
        action="store_true",
        help=(
            "Run the pass on a SEPARATE Python thread from the one that constructed the network. "
            "OpenMP's nthreads-var is a PER-THREAD control: omp_set_num_threads (which "
            "torch.set_num_threads calls) binds the calling thread only. cascor applies its "
            "parent pin in the CONSTRUCTOR, so if the service constructs on a request thread and "
            "trains on another, the training thread keeps the default ICV. This flag tests that "
            "mechanism directly in one process."
        ),
    )
    parser.add_argument(
        "--construct-on-thread",
        action="store_true",
        help="construct the network INSIDE the workload thread (use with --on-thread) to test whether the constructing thread, rather than the main thread, is what carries the pin",
    )
    parser.add_argument("--repeat", type=int, default=1, help="run the output pass N times back-to-back on the same thread (output_pass mode only)")
    parser.add_argument("--json", type=Path, default=None, help="write the full result as JSON here")
    parser.add_argument("--interval", type=float, default=0.2, help="sampler interval in seconds")
    args = parser.parse_args()

    blas_env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    all_set_to_2 = all(v == "2" for v in blas_env.values())
    if args.arm == "pinned" and not all_set_to_2:
        print(f"REFUSED: --arm pinned but the environment is {blas_env}; export all three to 2 in the shell.", file=sys.stderr)
        return 2
    if args.arm == "unpinned" and any(v is not None for v in blas_env.values()):
        print(f"REFUSED: --arm unpinned but the environment carries {blas_env}; unset them in the shell.", file=sys.stderr)
        return 2

    import torch  # imported AFTER the env check so the arm is honest

    threads_at_import = torch.get_num_threads()

    sys.path.insert(0, os.getcwd())
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation.cascade_correlation_config.cascade_correlation_config import (
        CascadeCorrelationConfig,
    )

    threads_after_import = torch.get_num_threads()

    # The cell's training params (pf8-occupancy-probe c000). ``train_output_layer`` is called
    # DIRECTLY rather than through ``fit()``, so the growth loop never runs and no candidate pool
    # is ever created -- exactly the state the burst was observed in (current_hidden_units 0, no
    # pool alive). ``max_iterations`` is therefore never read on this path; it is passed only so
    # the config matches the cell.
    config = CascadeCorrelationConfig(
        input_size=2,
        output_size=2,
        max_hidden_units=10,
        candidate_pool_size=4,
        max_iterations=10,
        output_epochs=args.epochs,
        worker_thread_count=1,
    )
    threads_after_config = torch.get_num_threads()

    # --construct-on-thread defers construction into the workload thread itself, which
    # discriminates "the MAIN thread is special" from "the CONSTRUCTING thread is special":
    # if the burst disappears when the SAME non-main thread both constructs and trains, then
    # the pin binds whichever thread ran the constructor and nothing else.
    network = None if args.construct_on_thread else CascadeCorrelationNetwork(config=config)
    threads_after_network = torch.get_num_threads()

    x_np, y_np = make_spiral(2, 200, 2.0, 0.05, 20260807)
    # train_ratio 0.8 of 400 -> 320 rows, the cell's real training width
    import numpy as np

    rng = np.random.default_rng(20260807)
    order = rng.permutation(len(x_np))
    keep = order[: int(0.8 * len(x_np))]
    x = torch.from_numpy(x_np[keep])
    y = torch.from_numpy(y_np[keep])

    runtimes = loaded_threading_runtimes()

    result = {
        "arm": args.arm,
        "blas_env": blas_env,
        "torch_num_threads": {
            "at_import": threads_at_import,
            "after_cascor_import": threads_after_import,
            "after_config": threads_after_config,
            "after_network_construction": threads_after_network,
        },
        "torch_interop_threads": torch.get_num_interop_threads(),
        "openblas_pool_size": openblas_pool_size(),
        "loaded_threading_runtimes": runtimes,
        "shape": {"rows": int(x.shape[0]), "in": int(x.shape[1]), "out": int(y.shape[1]), "epochs": args.epochs},
        "loadavg_before": os.getloadavg(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
    }

    def _synthetic_burn(kind: str, seconds: float = 4.0) -> None:
        """Burn a pool whose identity is known in advance, to validate the census."""
        if kind == "numpy":
            a = np.random.rand(1500, 1500).astype(np.float32)
            b = a.copy()
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                a @ b
        else:
            a = torch.rand(1500, 1500)
            b = a.clone()
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                a @ b

    def _run_workload() -> object:
        nonlocal network
        if network is None:
            network = CascadeCorrelationNetwork(config=config)
            result["torch_num_threads"]["after_network_construction_on_workload_thread"] = torch.get_num_threads()
        if args.synthetic:
            _synthetic_burn(args.synthetic)
            return None
        if args.mode == "fit":
            network.fit(
                x_train=x,
                y_train=y,
                max_epochs=args.epochs,
                max_iterations=args.max_iterations,
                early_stopping=True,
            )
            return None
        # --repeat runs the pass N times BACK TO BACK on the same thread. This discriminates
        # "the thread is wrong" from "the FIRST parallel workload on a thread is wrong": the
        # service's ten later output passes run on the same cascor-train thread as the initial
        # one, and only the initial one bursts, so if repeat 2 here does not burst the trigger is
        # the first workload on the thread and not the thread itself.
        last = None
        for index in range(args.repeat):
            marker = time.perf_counter()
            last = network.train_output_layer(x=x, y=y, epochs=args.epochs)
            result.setdefault("repeat_marks", []).append(
                {"pass": index + 1, "start_offset": round(marker - t0, 3), "end_offset": round(time.perf_counter() - t0, 3)}
            )
        return last

    result["synthetic"] = args.synthetic
    result["mode"] = args.mode
    result["on_thread"] = args.on_thread

    holder: dict[str, object] = {}
    with ThreadSampler(interval=args.interval) as sampler:
        t0 = time.perf_counter()
        if args.on_thread:
            worker = threading.Thread(
                target=lambda: holder.update(r=_run_workload()), name="pf8-workload"
            )
            worker.start()
            worker.join()
        else:
            holder["r"] = _run_workload()
        wall = time.perf_counter() - t0
    final_loss = holder.get("r")
    if args.on_thread and args.construct_on_thread:
        print("(the SAME non-main thread both constructed the network and ran the workload)")
    elif args.on_thread:
        print("(workload ran on a SEPARATE thread from the one that constructed the network)")

    result["pass_wall_seconds"] = round(wall, 4)
    result["ms_per_epoch"] = round(1000.0 * wall / args.epochs, 4)
    result["final_loss"] = float(final_loss) if final_loss is not None else None
    result["loadavg_after"] = os.getloadavg()
    result["attribution"] = sampler.attribute()

    attribution = result["attribution"]
    print(f"arm                     : {args.arm}  env={blas_env}")
    print(f"torch.get_num_threads() : import={threads_at_import} after-config={threads_after_config} AFTER-NETWORK={threads_after_network}")
    print(f"openblas pool size      : {result['openblas_pool_size']}")
    print(f"threading runtimes      : {sorted(runtimes)}")
    print(f"initial output pass     : {args.epochs} epochs over {x.shape[0]} rows, {wall:.2f} s ({result['ms_per_epoch']:.2f} ms/epoch)")
    print(f"process cores           : {attribution.get('process_cores')}  (excl sampler {attribution.get('process_cores_excl_sampler')})")
    print(f"PEAK interval           : {attribution.get('peak_interval_cores')} cores, {attribution.get('peak_interval_threads')} threads over 0.30")
    print(f"threads alive at end    : {attribution.get('thread_count_end')}   burning >0.05 cores: {attribution.get('threads_burning_over_0p05')}")
    series = attribution.get("series", [])
    if series:
        head = " ".join(f"{s['cores']:.1f}/{s['threads_over_0p30']}" for s in series[:16])
        print(f"series cores/threads    : {head}{' ...' if len(series) > 16 else ''}")
    print("top threads by cores:")
    for row in attribution.get("per_thread", [])[:12]:
        tag = " [sampler]" if row.get("is_sampler") else ""
        print(f"   {row['cores']!s:>8}  tid={row['tid']:<8} comm={row['comm']!r}{tag}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
