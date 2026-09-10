# PF scenario suites (plan §12.3 — Wave 7.3)

Operator surface (PF-1 matched epoch pair, matrix-axis repeats, scrapeability, PF-3 stall/wall, PF-4/PF-8 not driver suites): [`docs/REFERENCE.md` § PF Scenario Suites](../../../../docs/REFERENCE.md#pf-scenario-suites).

Runnable instruments for the performance-scenario matrix. **Thresholds are deliberately absent**: §12 fixes the reuse decisions and the measurement contract only — the scenario matrix and its thresholds still need a ratification pass of their own. Run any file with:

```bash
python util/experiments/run_suite.py --suite util/experiments/suites/perf/<file>.yaml --dry-run   # inspect first
```

| ID | File | Instrument surface |
| --- | --- | --- |
| PF-1 | `pf1-cascor-spiral-repeats.yaml` | step-duration p50/p95 + wall-clock variance over 5 identical cells |
| PF-2 | `pf2-cascor-dataset-scaling.yaml` | wall-clock vs samples; RSS via the experiments dashboard Performance row |
| PF-3 | `pf3-cascor-pool-scaling.yaml` | speedup curve; oversubscription onset via the Process CPU Rate panel. **BLOCKED 2026-09-10 — do not launch as written**: its `runtime.num_processes` axis is read by nothing on either path (P2 item 2.2, owner; `notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md` §4) |
| PF-4 | — not a driver suite | cascor's in-repo perf suite; report-only timing reference cut with `--benchmark-autosave` outside every checkout (`juniper-cascor` `docs/testing/REFERENCE.md` § Micro timing reference). No `baseline_*.json` ever held timing data |
| PF-5 | `pf5-recurrence-d-scaling.yaml` | fit time vs `d`; r² vs fit time |
| PF-6 | `pf6-recurrence-nsteps-scaling.yaml` | fit time vs window count |
| PF-7 | `pf7-recurrence-readout-rungs.yaml` | fit time + r² per readout rung |
| PF-8 | — not a sequential suite | two **simultaneous** pinned runs. **Run 2026-09-10** (`util/ad-hoc/2026-09-10_pf8_two_run_parallel_suite.yaml` via `2026-09-10_pf8_pair_driver.bash`): a second concurrent pinned run costs **+11.3%/step, +8.5 to +12.7% over 3 pairs** (vs 6 controls, `step_count` 1770) — advisory. One run consumes 4.52 cores unpinned / 2.15 pinned. Not checked in (fails the R-6 gate in CI; P2 4.2, now optional). `notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md` |
