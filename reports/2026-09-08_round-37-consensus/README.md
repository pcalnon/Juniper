# Round-37 handoff — independent-agent consensus record (2026-09-08)

Validation of
`prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-07_defect-register-round-37-the-inert-cache-and-the-partial-data-contracts-last-mile.md`
under `notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_INDEPENDENT-AGENT-CONSENSUS-PROCEDURE.md`,
run by the round-38 session before acting on the document. Consumed by
`prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-08_defect-register-round-38-the-three-way-prompt-built-and-the-annotation-made-truthful.md`
(§4 corrections, §7 record).

## Minimum record (procedure §7)

- **Frozen tree** (every lane read these SHAs in place, read-only; no branch, stash or rebase
  during the pool): juniper-data `03b7548f`, juniper-cascor `d39d537e`, juniper-canopy `eb05021d`,
  juniper-recurrence `e5679b00`, juniper-deploy `54abfb78`, juniper-ml `44de51c5`.
- **Six lanes, one launch message, four distinct entry points** (Lane A) plus two refutation
  briefs (Lane B):

  | Lane | Entry point | Verdict | Report |
  |---|---|---|---|
  | A1 | git / `gh` receipts only, no source | PASS with 4 corrections (1 false-at-archive receipt, 3 miscounts / literal-false sweeps) | `laneA1-receipts.md` |
  | A2 | the source at the frozen SHAs, executing pure functions | FAIL: 2 refuted (`model_fields_set`; `-p` advice), 2 partial, 4 stale lines; 6 new findings | `laneA2-source.md` |
  | A3 | the on-disk SEC shares cache, own scripts (graduated to `juniper-data/util/ad-hoc/2026-09-08_equities_shares_cache_census/`) | PASS 10/10 on substance; 6 new findings | `laneA3-shares-cache.md` |
  | A4 | a self-built storage benchmark + PR/CodeQL history | PASS 9/9; 8 new findings | `laneA4-perf.md` |
  | B1 | refute the load-bearing conclusions | 0 succeeded / 2 partial / 4 failed; 5 new findings | `laneB1-refute.md` |
  | B2 | amputation vs the round-36 chain, executability, past-tense receipts | NEEDS CORRECTIONS on all three lenses | `laneB2-amputation-exec.md` |

- **Iterations:** one round on the predecessor; a second round on the round-38 document's
  corrections (recorded in that handoff's §7).
- **Reconciliation:** every correction was re-derived from source by the orchestrating session
  before being applied; two lane findings were themselves adjusted (B1's "third OR consumer" is a
  third generator through an existing site; A3's 43.1% is definition-dependent at the third digit).
- **Unresolved dissent:** none on a number or a disposition. B1 and A3 split the round-37 §0.9
  "never executes the empty-units guard" sentence into a true half (the concept-loop guard) and a
  false half (the post-load guard runs on a warm hit); both agree on the consequence.
- **What the evidence cannot support:** SEC's live endpoint (no network in any lane); the harm
  magnitude of the `adj_close` dividend channel; whether round 37's own two validation rounds ran
  as described (no record exists); trading-day counts for tickers other than KO.
- **Lane instruments that could have produced a different answer:** A3's census and restatement
  scripts (re-derived KO to the digit and found the population figure); A4's benchmark (found the
  cache inert pre-fix at 0.7–1.1× and a non-overlapping 137–180× range, which is why only the
  order is quotable); B1's pydantic probe (executed both polarities of `allow_truncation`).
