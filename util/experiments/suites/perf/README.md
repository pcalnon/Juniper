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
| PF-3 | `pf3-cascor-pool-scaling.yaml` | speedup curve; oversubscription onset via the Process CPU Rate panel |
| PF-4 | — not a driver suite | cascor's in-repo perf suite; report-only timing reference cut with `--benchmark-autosave` outside every checkout (`juniper-cascor` `docs/testing/REFERENCE.md` § Micro timing reference). No `baseline_*.json` ever held timing data |
| PF-5 | `pf5-recurrence-d-scaling.yaml` | fit time vs `d`; r² vs fit time |
| PF-6 | `pf6-recurrence-nsteps-scaling.yaml` | fit time vs window count |
| PF-7 | `pf7-recurrence-readout-rungs.yaml` | fit time + r² per readout rung |
| PF-8 | — not a sequential suite | two **simultaneous** runs with pinned equal thread budgets. `run_suite` parallel mode IS the harness (cascor ≥ 0.10.0 lifts the one-checkout refusal; dry-run verified 2026-09-08) — but a parallel cascor suite checked in here fails the R-6 drift gate in CI, which cannot read the cascor version floor without the sibling repo (P2 item 4.2) |
