# Perf lane — PF-8 located: one run is 4.5 worker-equivalents unpinned and 2.2 pinned, a second run costs +11.3%, and the experiment YAML's `runtime:` block binds nothing

Successor to §1.3 of the re-scope note
([`JUNIPER_2026-09-08_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-RESCOPE-AND-MICRO-TIMING-REFERENCE.md`](JUNIPER_2026-09-08_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-RESCOPE-AND-MICRO-TIMING-REFERENCE.md))
and to §1 items 2 and 3 of the 2026-09-09 handoff
([`prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_perf-lane-pf8-needs-no-harness-micro-reference-cut-pf2-axis-inert.md`](../prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_perf-lane-pf8-needs-no-harness-micro-reference-cut-pf2-axis-inert.md)).
Item numbers refer to the P2 plan
([`JUNIPER_2026-09-02_JUNIPER-ECOSYSTEM_PERF-LANE-P2-PLAN.md`](JUNIPER_2026-09-02_JUNIPER-ECOSYSTEM_PERF-LANE-P2-PLAN.md))
unless stated otherwise; "the sweep note" is
[`JUNIPER_2026-09-02_JUNIPER-ECOSYSTEM_PF1-INSTRUMENT-RESOLUTION-AND-HEADROOM-SWEEP.md`](JUNIPER_2026-09-02_JUNIPER-ECOSYSTEM_PF1-INSTRUMENT-RESOLUTION-AND-HEADROOM-SWEEP.md).

**Four things happened, in the order §1.3 of the re-scope note required.** The micro timing
reference was re-cut at the quietest condition this arc has seen (§1). The occupancy probe — step 1
of §1.3 — ran, twice: once under cascor's default thread budget and once under the four-variable
budget `run_suite` pins for a parallel arm, and the two answers differ by 2× (§2). Because the
default-budget figure landed inside the ~4–8 band, step 2 ran too: the two-arm concurrency pair,
three parallel pairs bracketed by six sequential controls, all twelve cells at `step_count` 1770
(§3). And the probe's most consequential finding is not a number but a wiring fact: the
`runtime:` block every experiment YAML carries — `blas_threads`, `num_processes` — is validated by
the driver, accepted by the service, and **read by nothing on either path**; what the cascor
listener actually burns during output passes is NumPy's bundled OpenBLAS pool at the host's full
width, which `torch.set_num_threads` cannot touch (§4).

---

## 0. What this closes and what it leaves

| item | state after this document |
|---|---|
| **4.1 residue** (occupancy probe, S) | **DONE** — §2. One PF-1-shape run under the default budget occupies **4.52** sweep-worker equivalents during training (3 cells, 1.1% spread), **bimodal**: ~30% of the window at ~11, the rest at ~1.8. Under the pinned budget, **2.15**, never above the knee |
| **4.2** (the two-arm pair) | **EXECUTED from `util/ad-hoc/`** — §3. Parallel / control mean step = **1.1125**, i.e. a second run costs **+11.3%** at the pinned budget; inside the sweep's 20.5% quiet band, outside today's within-arm spread. Advisory, as 4.3 said it could only ever be. Where a *committed* instrument would live (the §1.5 CI hazard) is still the owner's call, and is now **optional** — §6 |
| **2.4 follow-up** (quiet re-cut `0003`) | **DONE at ambient, not idle** — §1. Cut at 1-minute load 5.3 → 7.1 against 9–10 for `0001` / `0002`; the micro tier does not see that difference (median ratio 0.99). `0003` is the cut to compare against |
| **2.1** (PF-2's inert axis) | Untouched — owner. §4 adds one constraint: fix the thread budget explicitly in any re-scope, because the budget moves step time by a third and `step_count` not at all |
| **2.2** (PF-3, ~6.7 h) | Untouched — no idle window (§5) |
| `epochs_completed` exact-match (handoff §1 item 1, third bullet) | Untouched — owner. §1 adds evidence: in every pairwise comparison of the three micro cuts, the benchmarks outside ±20.5% are dominated by the candidate tier, whose epoch count is emergent |
| **NEW owner question** | Should the launcher export the thread budget the `runtime:` block already declares? §4.3 — it changes what every existing YAML means, so it is a baseline-cutting decision, not a fix |

---

## 1. Micro timing reference `0003`

Cut 2026-09-10T09:35Z from the clean cascor primary at `a51b7c58` (`origin/main`; the holder check
the 2026-09-09 handoff's §5 requires came back empty), with `util/ad-hoc/2026-09-08_loadavg_sampler.py`
recording beside it (`~/.local/state/juniper-experiments/baselines/cascor-micro/loadavg-20260910.tsv`).
77 tests passed in 39.5 s; 71 benchmarks saved, the same 71 as `0001` and `0002`; `machine_info.juniper`
identical to both.

| run | sha | 1m / 5m / 15m load at save | trace during the run (1-minute, 5 s samples) |
|---|---|---|---|
| `0001` | `3286b758` (dirty) | 9.17 / 5.05 / 3.46 | 4.22 → 10.91, mean 7.41 |
| `0002` | `145fbe92` | 10.22 / 6.51 / 4.32 | 5.2 → 9.5 |
| **`0003`** | **`a51b7c58`** | **7.07 / 5.80 / 5.89** | **5.17 → 7.31**, back to 5.48 within a minute of the end |

**This is not a quiet host** — the sweep note's own quiet blocks ran at 5.9–18.6, and the
1-minute figure at save is the benchmarks' own load on top of a ~5.3 ambient. It is the
best-conditioned of the three cuts, and by the rule the cascor procedure states
(`juniper-cascor/docs/testing/REFERENCE.md` § Micro timing reference: *"re-cutting on a quiet host
supersedes it"*) it is the cut to compare against: `--benchmark-compare=0003`, never
`--benchmark-compare-fail` (owner, item 2.5).

**What the three cuts say about each other**, benchmark by benchmark from the saved JSON
(`util/ad-hoc/2026-09-10_micro_reference_compare.py`; median of each benchmark's rounds; no re-run):

| pair | 1-minute load at save | median ratio (other / base) | p10 – p90 | faster / slower | outside ±20.5% |
|---|---|---|---|---|---|
| `0003` vs `0002` | 7.1 vs 10.2 | **0.9945** | 0.874 – 1.166 | 37 / 34 | 7 |
| `0003` vs `0001` | 7.1 vs 9.2 | 0.9613 | 0.834 – 1.117 | 50 / 21 | 7 |
| `0002` vs `0001` | 10.2 vs 9.2, three minutes apart | 0.9542 | 0.846 – 1.102 | 47 / 24 | 4 |

Two readings. **The micro tier does not see the difference between a 1-minute load of 7 and one
of 10**: the quieter cut is not systematically faster (median ratio 0.99 against `0002`), and the
two loaded cuts differ from each other by as much as either differs from the quiet one. The
per-benchmark scatter between *any* two cuts is about ±15% at p10–p90 — the run tier's 13–20.5%
band, reproduced one tier down. And **the benchmarks that leave the band are the candidate ones**:
of the 18 out-of-band entries across the three comparisons, 12 are `test_micro_candidate.py` or
`test_baselines.py::TestCandidateTrainingBaseline` (the largest, `test_activation_comparison[sigmoid]`,
at 1.58× and 1.44×), against 2 correlation, 3 forward-pass / residual-error, 1 output-training. That is
the tier whose epoch count is emergent (2026-09-09 handoff §3.3), so a timing comparison of it is
partly a comparison of how many epochs each cut happened to run — evidence for, not a decision on,
the owner's `epochs_completed` exact-match question.

---

## 2. Step 1 — the occupancy probe

### 2.1 The instrument

`util/ad-hoc/2026-09-10_pf8_occupancy_sampler.py` watches the run root for run directories, reads
each run's `juniper-cascor.pid` / `juniper-data.pid` once the launcher writes them, finds the driver
by its `--run-dir` argument, and once a second walks `/proc/<pid>/stat` for the three process trees,
writing cpu-seconds per role per interval: `cascor_uvicorn` (the listener), `cascor_forkserver`,
`cascor_workers` (the forkserver's descendants — the candidate pool), `cascor_other`, `data`,
`driver`, plus the whole host's busy cores from `/proc/stat` and the 1-minute load. Divided by wall
seconds that is occupancy, and **1.0 is one sweep worker** (one saturated core, the unit §8.4 of the
sweep note is drawn in). It is `/proc` deltas, not `ps %CPU`, which is a lifetime average.
`util/ad-hoc/2026-09-10_pf8_occupancy_analyse.py` aligns the trace with each cell's drive window
(the first and last `ts_unix` of the driver's `metrics_series.csv`) and reduces it.
`tests/test_pf8_occupancy_probe.py` pins the arithmetic (26 tests, CI-wired).

What it cannot do: a pid that exits inside an interval takes its last partial second with it. The
`vanished` column counts those (3–5 per cell, all at teardown, none inside a drive window), so the
loss is bounded at under five cpu-seconds per run and falls outside the measured window anyway.

The workload is PF-1's cell shape exactly — `spiral-smoke.yaml` at `(10, 10)` / 4000 / 4000
(`util/ad-hoc/2026-09-10_pf8_occupancy_probe_suite.yaml`, three repeats as a matrix axis) — so that
`compare_baseline` can say whether the probe measured the baseline's workload. It did:
**`PASS`, `step_count` 1770 / 1770 against `pf1-2026-09-04b`** (speed reported −21.1%, not gated).

### 2.2 Under cascor's default thread budget: 4.52, and bimodal

Suite `pf8-occupancy-probe-20260910T093641Z`, no thread variable set (the manifests record all four
as `null`), so the listener ran with the runtime-default BLAS pool and the candidate pool with
cascor's own default of `min(candidate_pool_size 4, cores) − 1 = 3` processes.

| cell | `step_count` | mean step ms | `drive` s | cascor tree, worker-eq | p95 / max (1 s) | share of window ≥ 6 / ≥ 8 | occupancy ≥ 6 / < 6 | ambient cores | load 1m |
|---|---|---|---|---|---|---|---|---|---|
| c000 | 1770 | 28.36 | 55.20 | **4.524** | 11.57 / 12.38 | 0.31 / 0.31 | 10.64 / 1.80 | 3.63 | 7.54 |
| c001 | 1770 | 28.39 | 55.19 | **4.541** | 12.61 / 12.78 | 0.29 / 0.27 | 11.42 / 1.75 | 4.21 | 8.76 |
| c002 | 1770 | 27.89 | 55.21 | **4.490** | 12.33 / 12.66 | 0.29 / 0.27 | 11.20 / 1.77 | 3.71 | 8.93 |

**Mean 4.52 worker-equivalents, spread 1.1%** across three cells — the occupancy of this workload
is as reproducible as its step count. **But the mean is a bimodal trace's mean.** About 30% of each
training window sits at 10.6–11.4 cores, all of it in the *listener* process (the forkserver's
children never exceed ~2 in total), and the remaining 70% sits at ~1.8. The 1-second p95 is
11.6–12.6. §4 says what the high mode is.

Read on the sweep's axis, 4.52 is below the 6-worker point, where §8.4 measured +19.9% inside a
20.5% band — "below what this host can measure", not "free" — and it is inside the ~4–8 band §1.3
of the re-scope note set for running step 2. That is why §3 exists.

### 2.3 Under the budget `run_suite` pins for a parallel arm: 2.15, and a third faster

The same suite file with the four variables `thread_budget_env("cascor", 2)` would export on this
host — `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CASCOR_NUM_PROCESSES=4` — set in the
shell (suite `pf8-occupancy-probe-20260910T094028Z`; the manifests record all four).

| cell | `step_count` | mean step ms | `drive` s | cascor tree, worker-eq | p95 / max | share ≥ 6 | ambient cores | load 1m |
|---|---|---|---|---|---|---|---|---|
| c000 | 1770 | 18.44 | 35.12 | **2.150** | 3.03 / 3.11 | 0 | 4.40 | 6.63 |
| c001 | 1770 | 18.48 | 35.16 | **2.130** | 2.96 / 3.10 | 0 | 4.65 | 8.01 |
| c002 | 1770 | 18.88 | 35.13 | **2.169** | 3.04 / 3.21 | 0 | 4.95 | 6.95 |

Three things, each with its own weight:

1. **2.15 worker-equivalents, never above 3.3 in any second.** The bimodality is gone: with the
   listener's pool capped, the run is a near-constant load of about two cores. Below the band, so
   by §1.3's rule alone the pair would *not* have been worth running at this budget.
2. **The pinned run is faster — mean step 18.4–18.9 ms against 27.9–28.4, i.e. 34% faster per
   step, `drive` 35 s against 55.** Same day, same workload, n = 3 against 3, the direction the same
   in every cell, and the gap (34%) outside both the 20.5% quiet band and the 13% between-run
   drift. Report-only, as all speed is (decision 2 of §7 of the sweep note); but it means every
   service-path timing this lane has recorded, the PF-1 baseline included, was taken with the
   listener oversubscribed.
3. **`step_count` is 1770 in all six cells.** The thread budget moves the step *time* by a third
   and the work count by nothing — exactly the split the gate was designed around. And
   `compare_baseline` **REFUSED** this suite against `pf1-2026-09-04b` (exit 2: *"host identity
   differs from the baseline (thread_budget)"*), which is the identity precondition doing precisely
   what §2.2 of the P2 plan says it must: a pinned run is a different regime, not a 48% speed-up.

---

## 3. Step 2 — the two-run pair

### 3.1 Design

Per §1.3–§1.4 of the re-scope note, executed by `util/ad-hoc/2026-09-10_pf8_pair_driver.bash`
(one background launch, one log, the load average around every suite):

- **Parallel arm**: `util/ad-hoc/2026-09-10_pf8_two_run_parallel_suite.yaml` — two identical
  PF-1-shape cells, `execution: {mode: parallel, max_parallel: 2}`, run **three times** as three
  separate suites so the pairs are aligned (one six-cell suite would stagger cells 3–6 as the pool
  refilled). Each cell brings up its own juniper-data and cascor on its own ports. The suite lives
  under `util/ad-hoc/` for the reason §1.5 of the re-scope note gives: checked in under
  `suites/perf/` it turns `tests/test_experiment_suite_yamls.py` red in CI.
- **Control arm**: the probe suite of §2 — three sequential cells — run once **before** the
  parallel runs (§2.3's suite, `…T094028Z`) and once **after** (`…T094842Z`), with the same four
  variables exported by hand, as §1.3 requires; `run_suite` exports them only in parallel mode.
- **Instrument**: mean step duration (`step_sum / step_count`) from `read_run_metrics`, never
  `drive`. The sampler ran throughout (`suites/pf8-occupancy-trace-20260910.tsv`; a copy as
  `occupancy.tsv` in each suite directory; the reduced pair in `suites/pf8-pair-analysis-20260910.json`).

### 3.2 Result

| arm | cells | `step_count` | mean step ms | min – max | within-arm spread | cascor tree per cell, worker-eq | thread env |
|---|---|---|---|---|---|---|---|
| parallel (3 pairs) | 6 | 1770 in all | **20.72** | 20.15 – 21.91 | 8.7% | 2.02 | pinned, identical to control |
| control (3 before + 3 after) | 6 | 1770 in all | **18.63** | 18.30 – 19.15 | 4.7% | 2.11 | pinned |

**Identity holds before the comparison is read**: same four-variable budget in all twelve manifests,
same `step_count` in all twelve cells — the two checks `compare_baseline` would apply, applied.

**Parallel / control = 1.1125: a second concurrent run costs +11.3% per step**, at this budget, on
this day. The two arms do not overlap (the slowest control cell, 19.15 ms, is below the fastest
parallel cell, 20.15 ms), and the two runs together put **4.01–4.06** worker-equivalents on the host
— the pair total the parallel arm actually imposed, against the 4.52 a *single* unpinned run
imposes.

**Start alignment, which the re-scope note left as an assumption, is measured**: the two cells'
drive windows began 0.068 s, 0.046 s and 0.003 s apart and shared 40.1 s in every pair (100% of
both windows in two pairs; 89% of the longer one in the first, whose cell B ran one poll longer).
The parallel executor submits both cells back-to-back and the port lock's `mkdir` is the only
serialisation point; that is enough.

### 3.3 Two readings, both true

- **By the lane's standing rule** (§8.4 of the sweep note: an effect against a 20.5% quiet band
  "carries no information"; §1.3 of the re-scope note: inside the band is *"below what this host
  can measure"*): +11.3% is inside the band. Across sessions, days and ambient conditions, this
  host cannot distinguish a second run from drift.
- **By today's brackets**: the controls that bracket the parallel runs spread 4.7%, the parallel
  cells 8.7%, and +11.3% clears both with the arms disjoint. On the day, with n = 6 against 6, the
  cost is resolved and real.

PF-8 was never going to change a gate (item 4.3: the work half is settled, speed is ungated), so the
finding is what §1.3 said it would be — **advisory**: two pinned PF-1-shape runs may share this
host at a per-step cost of about a tenth, which is below the band the lane treats as noise between
sessions. Nothing here supports running two *unpinned* runs concurrently: each would burst to ~11
cores for a third of its window (§2.2), and two such bursts overlapping is 22 cores on 16 — the
plateau of §8.4 or past it — but that configuration was not run, because the parallel arm cannot be
unpinned (`run_suite` always exports the budget in parallel mode), and the note does not guess.

### 3.4 Host condition

Ambient (host busy cores minus this run's own trees) was 3.6–4.2 during the default probe,
4.4–5.7 during the controls, and about 6 during the parallel runs — where the *partner* run is
inside the ambient figure, so the true ambient there is ~4. Between cells the host ran 4.8 busy
cores at a 1-minute load of 6.6 (223 host-only samples). The sweep's quiet blocks ran at loads of
5.9–18.6. This was ambient, not idle, throughout; the load was recorded beside every measurement.

---

## 4. Why the listener burns eleven cores, and why nothing in the YAML can stop it

### 4.1 The `runtime:` block is read by nothing

`spiral-smoke.yaml` — the base config of PF-1, PF-2, PF-3 and both probes — declares
`runtime: {blas_threads: 2}`, and the P1 design and the re-scope note both describe the workload as
*"a candidate pool of 4 with `runtime.blas_threads: 2`"*. Traced through every consumer on
2026-09-10:

| layer | what it does with `runtime:` | where |
|---|---|---|
| driver `run_experiment.py` | validates the key set (`RUNTIME_KEYS = {num_processes, blas_threads, eval_metrics_enabled}`) and **reads none of them** | `util/experiments/run_experiment.py:171`, `:607-609`; no other reference in the file |
| launcher `experiment_stack.bash` | exports no thread variable; the cascor bring-up line carries `JUNIPER_CASCOR_CONFIG_FILE` and the run's provenance variables only | `util/experiment_stack.bash:647` (the announced command), `:633` |
| cascor service | accepts `runtime` as a known top-level block and projects **only `service:`** into Settings — *"the other blocks belong to the driver / launcher layers and are deliberately ignored here"* | `juniper-cascor/src/api/settings.py:142-144` |
| cascor direct CLI | reads `dataset.params` and `training.params` from the file, nothing else | `juniper-cascor/src/main.py:305-320` |
| cascor thread policy | *"Default: do nothing, leaving the runtime's own choice"*; `JUNIPER_CASCOR_BLAS_THREADS=<n>` opts in, and the launcher never sets it | `juniper-cascor/src/parallelism/blas_threads.py` module docstring |

So `runtime.blas_threads: 2` and `runtime.num_processes` are **decorative on both paths**: a
documented intent (`settings.py` calls the block "process-env territory") that no layer ever
implemented. The manifests are honest about it — `environment.thread_env` records all four
variables as `null` for every unpinned run — but nothing warns that the YAML asked for something
the run did not get.

### 4.2 What the listener's high mode is: NumPy's OpenBLAS, not torch

cascor pins its own torch pools: the parent to `max(2, 2 × worker_thread_count)` threads
(`cascade_correlation.py:1179-1180`) and each candidate worker to `worker_thread_count`
(`:4152-4153`). Those pins hold — the probe's candidate workers never exceeded ~2 cores in total.
The listener's 11-core bursts are something the pins do not reach. Isolated in a fresh interpreter
per case (`util/ad-hoc/2026-09-10_torch_thread_pin_probe.py`, 1024² float32 matmuls for 2 s,
cpu-seconds per wall-second from `/proc/self/stat`, JuniperCascor1: torch 2.11.0+cu130 on MKL
2024.2, numpy 2.4.4 on its bundled `scipy-openblas64`):

| case | `torch.get_num_threads()` | cores consumed |
|---|---|---|
| torch, variables unset, no pin | 8 | 6.7 |
| torch, variables unset, `torch.set_num_threads(2)` | 2 | **1.9** |
| torch, `OMP/MKL/OPENBLAS=2` | 2 | 1.9 |
| **NumPy, variables unset, no pin** | 8 | **11.5** |
| **NumPy, variables unset, `torch.set_num_threads(2)`** | 2 | **10.9** |
| NumPy, `OMP/MKL/OPENBLAS=2` | 2 | **2.0** |

`torch.set_num_threads(2)` bounds torch's own operators to two cores and leaves NumPy's OpenBLAS
pool at the host's full width; only the environment variable — read once, when the library loads —
reaches it. The listener's measured high mode (10.6–11.4) matches the NumPy figure, and pinning the
three variables at 2 removed it entirely (§2.3). And the NumPy loop at 16 threads completed
**fewer** matmuls in its two seconds (57) than at two threads (79): on a host with 8 physical
cores, hyper-threading, and a 4–5-core ambient load, a 16-thread OpenBLAS pool is oversubscribed
before the run's own candidate pool is counted — which is the 34% of §2.3.

This is attribution by magnitude and by a direct probe of the libraries, not by profiling the
listener's stack; which NumPy calls in cascor's output-layer path carry the work is not identified
here.

### 4.3 What follows, and what is deliberately not done

- **Every service-path figure in this lane — PF-1's calibration, its five-repeat baseline
  `pf1-2026-09-04b`, the 21-cell headroom sweep, the PF-2 probe — was taken with the listener's
  OpenBLAS pool at 16 threads.** They remain valid measurements of that regime, and their
  `step_count`s are untouched (1770 in all fifteen cells of this session, pinned or not). Their
  *speeds* describe an oversubscribed listener.
- **Fixing it is an owner decision, not a bug fix.** The obvious change — the launcher exporting
  `JUNIPER_CASCOR_BLAS_THREADS` (or the three variables) from `runtime.blas_threads` and
  `CASCOR_NUM_PROCESSES` from `runtime.num_processes` — would make every existing experiment YAML
  mean something different from what it has meant in every run to date, change the speed regime of
  every suite, and require a new run-tier baseline (`compare_baseline` would correctly refuse the
  old one on `thread_budget`). It touches the launcher, the driver's environment capture, cascor's
  policy module, and every document that quotes a service-path timing. Nothing here does it.
- The candidate pool's own budget is a second knob with a measured cost: cascor#531 found a
  BLAS cap *slowed* the candidate phase 1.52× and changed its epoch count. The output-layer phase
  measured here moves the other way. Any budget decision needs both phases measured under it; this
  document measured one.

---

## 5. Host condition during this session's measurements

| measurement | when (UTC) | 1-minute load at start → end | trace |
|---|---|---|---|
| micro reference `0003` | 09:34:29 – 09:36:34 | 5.19 → 7.31 → 5.48 | `baselines/cascor-micro/loadavg-20260910.tsv` |
| occupancy probe, default | 09:36:41 – 09:40:00 | 7.5 – 8.9 per cell | `suites/pf8-occupancy-trace-20260910.tsv` |
| occupancy probe, pinned (control before) | 09:40:28 – 09:43:20 | 6.6 – 8.0 | same |
| parallel arm ×3 | 09:45:18 – 09:48:27 | 5.50 → 8.05, 7.76 → 7.69, 6.65 → 7.91 | same + `suites/pf8-pair-driver-20260910T094518Z.log` |
| control after | 09:48:42 – 09:51:09 | 7.13 → 8.65 | same |

A `clamscan` had been holding one core for nineteen CPU-hours since before the session and a
browser content process burst to 170% throughout; seven peer sessions were live, one holding an idle
cascor stack on `:8202`. Ambient, not idle — the sweep's own standard (§8.2 of the sweep note).

---

## 6. Owner decisions — three carried, one new

1. **PF-2's axis** (2026-09-09 handoff §1 item 1, first bullet) — unchanged. One constraint added
   by §2.3: whatever the scenario becomes, its suite must fix the thread budget explicitly, because
   the budget moves step time by a third and the work count not at all; a scenario that inherits
   "whatever the shell had" measures the shell.
2. **Item 4.2's CI hazard** (second bullet) — now **optional**. The pair has run from `util/ad-hoc/`
   (the headroom sweep's precedent) and PF-8's advisory answer is on record. A committed
   `suites/perf/` pair is needed only if PF-8 is to be re-run routinely; if it is, the recommended
   fix stands (move the parallel floor check from `load_suite` to the execution path, still
   fail-closed, with its own negative tests). If it is not, the hazard stays a documented hazard
   (P2 plan §4) and no gate changes.
3. **`epochs_completed` exact-match** (third bullet) — unchanged. §1's evidence: the candidate
   benchmarks are the ones that leave the ±20.5% band between cuts.
4. **NEW — the `runtime:` block.** Implement it (launcher exports; new baseline; every YAML's
   meaning changes) or retire it (drop the keys from the schema so a config cannot ask for what it
   will not get). Either is a decision about what every experiment config *means*; §4.3 says why it
   is not taken here.

---

## 7. What this document does not settle

- **n is 3 per probe budget and 6 per arm, on one day.** The 1.1% / 1.8% occupancy spreads and the
  disjoint arms are strong for one day; §3.3's second reading does not survive a different day's
  ambient without repeats, and says so.
- **The pair ran only at the pinned budget.** Two unpinned runs — the configuration an operator who
  launches two ordinary PF-1 suites would actually create — were not measured, and §3.3 says what
  the occupancy trace predicts for them without claiming it.
- **The NumPy attribution is by magnitude and library probe**, not by a profile of the listener.
- **`0003` is ambient, not quiet.** A cut at a 1-minute load under 3 has still never been taken on
  this host.
- **The 34% pinned-vs-default speed difference is a between-suite comparison** taken forty minutes
  apart under similar ambient; it is consistent in all six cells and far outside the bands, but it
  was not interleaved cell-by-cell.

---

## 8. Reproduction

```bash
# Micro reference 0003 (cascor primary, JuniperCascor1), then compare the cuts offline
env -C /home/pcalnon/Development/python/Juniper/juniper-cascor/src /opt/miniforge3/envs/JuniperCascor1/bin/python -m pytest \
  tests/performance/test_baselines.py tests/performance/test_micro_autograd.py tests/performance/test_micro_candidate.py \
  tests/performance/test_micro_correlation.py tests/performance/test_micro_forward_pass.py tests/performance/test_micro_output_training.py \
  --run-performance --benchmark-storage=file:///home/pcalnon/.local/state/juniper-experiments/baselines/cascor-micro --benchmark-autosave
S=~/.local/state/juniper-experiments/baselines/cascor-micro/Linux-CPython-3.13-64bit
python3 util/ad-hoc/2026-09-10_micro_reference_compare.py --base $S/0002_*.json --other $S/0003_*.json

# Occupancy probe (start the sampler first; stop it by the pid it records)
python3 util/ad-hoc/2026-09-10_pf8_occupancy_sampler.py --out occupancy.tsv --pid-file occ.pid --max-seconds 1800 &
JUNIPER_EXP_PROJECT_DIR=/home/pcalnon/Development/python/Juniper python3 util/experiments/run_suite.py --suite util/ad-hoc/2026-09-10_pf8_occupancy_probe_suite.yaml
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CASCOR_NUM_PROCESSES=4 JUNIPER_EXP_PROJECT_DIR=/home/pcalnon/Development/python/Juniper \
  python3 util/experiments/run_suite.py --suite util/ad-hoc/2026-09-10_pf8_occupancy_probe_suite.yaml
python3 util/ad-hoc/2026-09-10_pf8_occupancy_analyse.py --suite ~/.local/state/juniper-experiments/suites/<probe suite> --trace occupancy.tsv

# The pair (three parallel suites, then one control run), and its reduction
bash util/ad-hoc/2026-09-10_pf8_pair_driver.bash
python3 util/ad-hoc/2026-09-10_pf8_occupancy_analyse.py --parallel <three pf8-two-run-parallel-* dirs> --control <two pf8-occupancy-probe-* dirs> --trace occupancy.tsv
kill "$(cat occ.pid)"

# Identity against the PF-1 baseline: PASS for the unpinned probe, REFUSED (exit 2, thread_budget) for the pinned one
python3 util/experiments/compare_baseline.py --baseline pf1-2026-09-04b --suite ~/.local/state/juniper-experiments/suites/pf8-occupancy-probe-20260910T093641Z
python3 util/experiments/compare_baseline.py --baseline pf1-2026-09-04b --suite ~/.local/state/juniper-experiments/suites/pf8-occupancy-probe-20260910T094028Z

# The mechanism: torch's pin bounds torch, only the environment variable bounds NumPy's OpenBLAS
/opt/miniforge3/envs/JuniperCascor1/bin/python util/ad-hoc/2026-09-10_torch_thread_pin_probe.py

# The arithmetic
python3 -m unittest -v tests/test_pf8_occupancy_probe.py
```

Retained on this host, untracked: `~/.local/state/juniper-experiments/suites/pf8-occupancy-probe-20260910T{093641,094028,094842}Z/`,
`suites/pf8-two-run-parallel-20260910T{094518,094629,094736}Z/`, `suites/pf8-occupancy-trace-20260910.tsv`,
`suites/pf8-pair-analysis-20260910.json`, `suites/pf8-pair-driver-20260910T094518Z.log`, and
`baselines/cascor-micro/` run `0003` with `loadavg-20260910.tsv`.
