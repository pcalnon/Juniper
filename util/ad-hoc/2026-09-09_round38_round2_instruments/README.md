# Round-38 round-2 instruments (2026-09-09)

The four scripts the orchestrating session ran to **re-derive validation-lane findings before acting
on them**. Kept because a measurement whose instrument is gone is a number nobody can check: round
1's own perf lane recorded exactly that failure against the round-37 chain — *"No instrument for the
§4/§0.6/§0.7 measurements is preserved in any `util/ad-hoc/`"* — and this directory is that finding
applied to this round's own work.

Consensus record: `reports/2026-09-09_round-38-consensus/`. Handoff:
`prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_defect-register-round-38-the-three-way-prompt-shipped-and-two-corrections-that-reversed-themselves.md`.

| Script | Question it settled | Result |
|---|---|---|
| `verify_adjclose_channel.py` | Is `close / adj_close` monotone, and are there 91 distinct steps, as register row `APD-DATA-041` claimed? | **Both withdrawn.** 3246 up-steps against 3349 down; non-increasing only within a 3e-6 tolerance. 91 distinct values at 6 decimals, 1997 at 8, 6123 raw. The look-ahead itself is confirmed and stated more sharply: a row 2.5 years inside the window still carries a 1.13% adjustment. Needs the cached AAPL frame. |
| `verify_b1_claims.py` | Does the deployment binder destroy the distinction a presence guard needs? How many cached share series are stale? | **Yes, and 26.** Before the bind the two states differ in `model_fields_set`; after it both carry the key, so a guard downstream is constant-true — reversing this round's own "refutation". Separately, 26 of 485 delivered series stop before 2025-06-01, median 2026-04-24 — filed as `APD-DATA-045`. Run it against a checkout containing juniper-data#388. |
| `b105_probe.py` | Does bandit B105 fire on any constant whose name contains `TOKEN`? | **No — only on a plain assignment.** `X_TOKEN = "..."` trips it; `X_TOKEN: str = "..."` does not, because the check walks `ast.Assign` only. That is why the identical constant failed canopy's hook and merged green in cascor, and it refuted a lane's inference of a latent hook failure. Four lines; run `bandit -t B105` over it. |
| `cascor_drift_changelog.py` | — (not a measurement) | The edit that appended the drift-repair entry to cascor's `CHANGELOG.md` for juniper-cascor#639, placed at the **end** of the `### Fixed` block because another session was writing to the same section. |

The register-editing scripts for the same round are `util/ad-hoc/register_round38_file.py`,
`register_round38_dates.py`, `register_round38_round2_fixes.py` and `register_close_cascor640.py`;
the two code repairs are `cascor_fix_shortfall_all.py` and `cascor_fix_spiral_inprocess_claim.py`.
