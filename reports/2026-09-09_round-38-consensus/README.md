# Round-38 handoff — independent-agent consensus record, round 2 (2026-09-09)

Validation of
`prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_defect-register-round-38-the-three-way-prompt-shipped-and-two-corrections-that-reversed-themselves.md`
and of the round-38 edit to `notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md`, under
`notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_INDEPENDENT-AGENT-CONSENSUS-PROCEDURE.md`. Round 1, on
the predecessor, is recorded separately in `reports/2026-09-08_round-37-consensus/`.

## Minimum record (procedure §7)

- **Tree.** juniper-ml worktree `pure-toasting-token` at `44de51c5` with the register edit
  uncommitted and `origin/main` at `5fbe3bc1`; juniper-cascor `main` `44dafe0` (later `3de89b11`),
  juniper-canopy `main` containing `b587e54`, juniper-data `main` `d7f4be5`. Unlike round 1 this was
  **not** a frozen tree: the three PRs under review merged during the round, and one lane recorded
  that the primary juniper-data and juniper-canopy checkouts were behind their remotes — which is
  itself a finding (§5.10 of the handoff).
- **Three lanes, one launch message, three distinct briefs:**

  | Lane | Entry point | Verdict | Report |
  |---|---|---|---|
  | A | receipts (`gh`) plus source on each repo's `main`, re-running the suites | 25 of 35 CONFIRMED, **4 REFUTED**, 4 PARTIAL; 4 new findings | `laneA-receipts-and-source.md` |
  | B1 | refute the corrections, the new register rows and the merge doctrine | **3 succeeded / 4 partial / 1 failed**; 7 new findings | `laneB1-refute.md` |
  | B2 | amputation vs the round-37 chain, executability cold, naming and receipts | NEEDS CORRECTIONS on all three lenses; 2 DROPPED-LOST recovered | `laneB2-amputation-exec.md` |

- **Iterations:** round 1 on the predecessor (six lanes), round 2 on this round's corrections and its
  register edit (three lanes). No round has yet read the finished document cold; that is §0.1 of the
  handoff.

## What round 2 changed

Four results changed shipped artifacts rather than prose:

1. **A live `main` breakage.** Lane A found `CI — juniper-cascor-model` red on juniper-cascor `main`
   since `44dafe0` (juniper-cascor#633, this arc's own PR), and the register asserting the opposite
   in that PR's verification row. Re-derived here: the two `constants_api_defaults.py` copies differ
   by a seven-line `__all__` present only in the mirror, the drift suite goes 3 passed once removed,
   and the block did not even silence the CodeQL finding it was added for, because the module already
   ends with its own complete `__all__`. Repaired in juniper-cascor#639; the register row now says
   what happened.
2. **Two false verification receipts** in the register's §5.1: "228 passed" (the nine suites total
   **220**) and "keeps the drift test green".
3. **The licensing clause withdrawn.** Lane B1's attack 7 showed the §2 note added by this round
   asserted the unparked-⇒-actionable equivalence the register's own §2 rejects, drew its authority
   from a machine-written handoff, and covered one row (`APD-DATA-039`) that round 37 never listed as
   work and the round-38 handoff's own §0.4 parks. Deleted; the note now records why.
4. **Two reversals of this round's own corrections**, both re-derived before applying: the
   `model_fields_set` presence guard is destroyed by `bind_deployment_defaults`' `model_copy(update=…)`
   (so round 37 was substantially right), and the shortfall annotation is written *before* the
   finiteness refusal, not after.

Two new register rows came out of the round: `APD-DATA-045` (26 of 485 delivered share series stop
before 2025-06-01 and are forward-filled) and `APD-CASCOR-013` (`_dataset_shortfall` is never
cleared). Two severities were regraded on lane B1's precedent argument (`APD-DATA-044` R→C,
`APD-CASCOR-010` E→C), and four rows were restated where a number or a mechanism was wrong.

## Reconciliation — lane findings the orchestrating session refuted

A lane finding is evidence, not an instruction; each was re-derived before being applied, and two
did not survive that:

- **"juniper-data#388 left a stale KO/ABT comment standing"** (lane B2). The matched text is the
  *corrective* comment that same PR added, which quotes the old wrong claim in order to retract it.
  `git log -S"wrong examples"` lands on `d7f4be5` — juniper-data#388 itself. Not applied. This is the
  grep-the-concept-then-read-the-hit failure the ecosystem memory already warns about.
- **"cascor ships a `*_TOKEN` constant inside the bandit hook's scope with no suppression, so either
  §5.7 is over-general or this is a latent hook failure"** (lane B2). Probed with bandit 1.9.4:
  `X_TOKEN = "..."` trips B105, `X_TOKEN: str = "..."` does not, because the check walks `ast.Assign`
  only. No latent failure; §5.7 was over-general, and the memory
  `reference_bandit_b105_flags_token_names` has been corrected.

Adjusted rather than refuted: lane B1 retracted its own attack-8 claim of five additional placeholder
CIKs after a delegated sub-lane showed the outlier filter removes them before delivery, and downgraded
that attack to PARTIAL. Its surviving point — that the register row asserted payload-level counts it
did not have — was applied.

## Unresolved

- Which of two measurement ranges for juniper-data#381 is right (the register carries the union of
  round 37's and round 1's, labelled as such); neither was re-run in round 2.
- Whether cascor's `Test (Python 3.12)` failure on the merged head was the drift assertion or
  coverage — no lane read the job log. It merged red either way, so it is not a required check there.

## What the evidence cannot support

SEC's live endpoint (no network in any lane); the harm magnitude of the `adj_close` dividend channel
(mechanism proven, next-day correlation 0.004); whether the six unusable share series are placeholders
at SEC or cache artifacts; whether round 37's own validation rounds ran as it describes.

## Lane instruments that could have produced a different answer

Lane A's re-run of all nine juniper-data suites (which is what produced 220 against the claimed 228)
and its `diff -r` of the two extracted constants trees; lane B1's scratch-repo `git merge-tree`
experiment, which disproved this round's merge doctrine on the session's own captured files, and its
pydantic probe through the deployment binder; the orchestrating session's own bandit polarity probe
and its re-derivation of the `close/adj_close` series, which confirmed the look-ahead while
withdrawing both numbers the register had quoted for it.
