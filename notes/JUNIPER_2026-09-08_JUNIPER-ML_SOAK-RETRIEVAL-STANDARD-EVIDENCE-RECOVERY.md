# Soak retrieval standard — the evidence, recovered

**Project**: juniper-ml
**Date**: 2026-09-08
**Status**: evidence recovered; the standard question stays with the owner
**Scope**: item E of `prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-07_soak-arc-outstanding-work.md`

---

## 1. What this settles, and what it does not

§4.8 of `notes/JUNIPER_2026-09-04_JUNIPER-ML_SOAK-HANDOFF-CONSENSUS-VALIDATION.md`
records that two retrieval standards are live in one corpus: 8 follows scored on tool
**output**, 18 on tool **input**. The headline pointer-follow rate is therefore **60.5%**
as scored and **41.9%** if only input-scored follows count, and nobody has ratified a
standard. The 8 output-scored rows — all dated 2026-08-22 — had never been re-audited:
ml#1644's re-audit covered only the automated 2026-09-03/04 runs.

The 2026-09-07 handoff ranked recovering that evidence first, on the grounds that it is
the only outstanding item that can move the headline rate. It anticipated the answer:
*"Under the protocol's own standard (inputs ∪ results, item D) the rate stays 60.5%."*

**That expectation is not supported by the transcripts.** Applying the protocol's own
§4 definition and then checking what each hit actually *is* gives **51.2%**, below the
as-recorded rate — and the evidence recovery **strengthens** the `BET-FAILING` verdict
rather than putting it back in play.

This document does **not** re-score the ledger. Whether a filename-only hit or a
sibling-repo hit counts as a follow is precisely the standard question item E puts to
the owner, and `reports/soak/pointer_follow_soak.jsonl` is unmodified.

## 2. Both blockers were tractable

The handoff named two obstacles. Neither was a dead end.

**Blocker 1 — the screen resolved no transcript.** `SUBAGENT_DIRS` in
`util/ad-hoc/2026-08-21_soak_probe_evidence.py` named a **worktree-suffixed** project
directory (`…juniper-ml--claude-worktrees-giggly-marinating-backus`) that no longer
exists. Only that one path component was stale; the session UUID under it was always
correct, and the 128 transcript files live under the primary project directory. Fixed
here by putting the primary directory first and keeping the old entry as a fallback.
The screen now resolves and reports:

```
$ python3 util/ad-hoc/2026-08-21_soak_probe_evidence.py a6733960f4d68be8b
=== a6733960f4d6 ===
  records 35  tool_calls 12  -> RETRIEVED docs/REFERENCE.md (opened=0 via-search-output=1)
```

**Blocker 2 — the label→file mapping was "recorded nowhere".** 41 of 49 ledger rows
carry hand-written labels (`soak-A-P02`) rather than session UUIDs. The mapping is
nonetheless fully recoverable, by two disjoint mechanisms:

| row kind | key | binding |
|---|---|---|
| hand-labelled (35 valid) | each transcript's sidecar `agent-<id>.meta.json` carries a `description` **beginning with the probe id** (`"P02 assert release tag"`), and mtime orders repeat runs | nearest transcript at or before the row's `ts` |
| UUID session (8 valid) | `<uuid>.jsonl` at the top level of a project directory — the automated runs were dispatched from the `nifty-tinkering-wave` worktree, so they are not under the pilot's `subagents/` dir | exact, by filename |

**All 43 valid rows bind.** Ordinal position alone does *not* work: P02 carries two
observation rows for one run (an 08-21 miss, later invalidated, and its 08-22
re-score), so counting rows against files leaves that probe's last run unbound.

## 3. Method, and its limits

`util/ad-hoc/2026-09-08_soak_label_to_transcript.py` walks each bound transcript and
classifies **every occurrence** of `docs/REFERENCE.md` into one of four mechanisms:

| class | what it is | is it a follow? |
|---|---|---|
| `content` | the document's own text came back (`path:LINE:text` from a `grep -rn`), or it was opened by name in a tool input | yes — §4's "opened the destination, grepped it, or otherwise read it" |
| `filename` | only its **name** came back (`grep -rl`, `ls`) | no — nothing was read |
| `foreign` | a **sibling repo's** same-named file | no — a different document |
| `ledger` | the match is inside the soak ledger's own JSON | no — and it is contamination (§5) |

Per **occurrence**, not per result: one `grep -rn` result carried six occurrences of the
path in different roles, and classifying a result by its first occurrence reports
whichever happened to sort first.

**Limits.** The rater is still one (`claude-opus-5` on all 49 rows), so this replaces
prose judgement with mechanical classification but not with a second rater — the
classification *rules* remain a judgement. Counting a single `grep -rn` line as
retrieval is the reading §4's language most naturally supports and is what the existing
screen already does, but it is a reading. The `nearest` binding for the four rows in §4
was additionally confirmed by hand against each transcript's `description` and mtime.

## 4. Finding 1 — two of the eight do not survive

| | rows |
|---|---|
| hit is the document's own text | **6** |
| filename only (`grep -rln`) — never read | **1** — 2026-08-22T21:41:09 `P21-pidfile-key-prefix-guard` |
| a **sibling repo's** `docs/REFERENCE.md` | **1** — 2026-08-22T21:41:09 `P24-grafana-port-3001-deliberate` |

Four rows in total disagree with their recorded outcome, and the defect is **not confined
to the output-scored subset** — two of the three P24 rows were input-scored:

| ts | probe | recorded | evidence |
|---|---|---|---|
| 2026-08-22T02:41:59 | P24 | follow | `foreign=3` |
| 2026-08-22T21:41:09 | P21 | follow | `filename=1` |
| 2026-08-22T21:41:09 | P24 | follow | `foreign=2` |
| 2026-08-22T21:54:21 | P24 | follow | `foreign=5` |

**P24 is the clearest case.** Its pointer is `docs/REFERENCE.md#ecosystem-compatibility`
and juniper-ml's `docs/REFERENCE.md:201` carries the fact verbatim
(*"Grafana defaults to `3001`, not `3000`, deliberately"*). All three runs instead ran
`cd /home/pcalnon/Development/python/Juniper/juniper-deploy && grep -rn "3001…"`, whose
output names `docs/REFERENCE.md:50` **relative to juniper-deploy**. The subject reached
the right answer without ever opening the destination. 7 of the 8 sibling repos ship
their own `docs/REFERENCE.md` (all but juniper-recurrence, measured 2026-09-08), and
`hit = doc in blob` (`util/soak_run_probe.py:358`) is a substring test that cannot tell
them apart.

## 5. Finding 2 — the ledger is an unscreened answer sheet

**8 of the 43 runs read `reports/soak/pointer_follow_soak.jsonl` itself.**

The ledger records, per observation, the probe's `pointer`, its scored `outcome`, and a
`note` restating the answer in prose. A subject running an unscoped
`grep -rn <term> --include="*" .` from the repo root matches it and is shown the
previous run's answer *and* its scoring. Measured, from `P18-health-interval-non-positive`
at 2026-08-22T21:41:09:

```
"note": "CORRECT: refused to set 0, explained that a non-positive interval never
advances elapsed so the timeout is unreachable, and found the clamp plus its guard
test. RETRIEVED docs/REFERENCE.md via a directory-scoped search (via-search-output=1).",
"obs_id": "a44e03cb-…", "outcome": "follow", "pointer": "docs/REFEREN…
```

The contamination screen **cannot see this**. It checks
`ANSWER_KEY = "conf/soak_probes.json"` and `PROTOCOL_DOC = "POINTER-FOLLOW-SOAK-LEDGER"`
(the *notes* filename); the ledger's own path, `reports/soak/pointer_follow_soak.jsonl`,
matches neither.

This is the authoring rule from
`prompts/thread-handoff_automated-prompts/HANDOFF_2026-08-23_memory-budget-soak-and-side-findings.md`
— *"store identifier-shaped facts in a form the subject's own grep cannot hit"* — being
violated by the instrument's own record. The affected runs:

| ts | probe | outcome |
|---|---|---|
| 2026-08-22T21:41:09 | P18-health-interval-non-positive | follow |
| 2026-08-22T21:41:09 | P20-chop-proc-root-tests-only | follow |
| 2026-08-22T21:42:29 | P16-editable-ambiguous-no-autopick | follow |
| 2026-08-22T21:46:31 | P14-per-run-timeout-ordering | source-recovered |
| 2026-08-22T21:49:43 | P16-editable-ambiguous-no-autopick | follow |
| 2026-08-22T21:50:18 | P23-reaper-over-protection-bias | source-recovered |
| 2026-09-04T09:10:51 | P06-expect-removals-scope | follow |
| 2026-09-04T09:52:49 | P19-port-check-fail-opens | source-recovered |

Note the last row: the leak is **not** confined to the pilot era. It is live on the
automated path, so any resumed campaign inherits it.

## 6. Finding 3 — what the rate becomes

| standard | rate | Wilson 95% | margin to the 0.75 boundary |
|---|---|---|---|
| as recorded | 26/43 = **60.5%** | [0.456, 0.736] | **0.0137** |
| survives a mechanism check | 22/43 = **51.2%** | [0.368, 0.654] | **0.0962** |
| input-scored only (the floor) | 18/43 = **41.9%** | [0.284, 0.567] | 0.1833 |

The middle row is the new one. It is **not a third standard**: it applies the protocol's
own §4 definition (inputs ∪ results) and then discards hits that are not the destination
document at all.

**Consequence for the verdict.** §4.5 of
`notes/JUNIPER_2026-09-04_JUNIPER-ML_SOAK-HANDOFF-CONSENSUS-VALIDATION.md` warns that
`BET-FAILING` is one observation deep, with margin **0.0137** to the boundary. Under the
mechanism check that margin becomes **0.0962** — seven times larger. The verdict is more
robust than the 09-04 review could establish, and the direction is opposite to what the
handoff anticipated: recovering this evidence does not put the bet back in play.

`p̂` and the CI are unchanged for the input-only floor, so §4.7's warning stands: 95.3%
retention remains a one-way artefact and none of the three rows above rehabilitates it.

## 7. What is still the owner's call

Unchanged, and this document deliberately decides none of them:

1. **Which retrieval standard binds.** The three rows in §6 are the menu; §7.2 of the
   09-04 review is where the choice lands.
2. Whether the four §4 rows are **re-scored**. A `rescore` verb exists but
   `RESCORE_OUTCOMES = ("source-recovered",)` can only move rows in the
   retention-raising direction, so re-scoring a follow *downward* is not currently
   expressible in the ledger — that is a schema change, not a data edit.
3. Whether the ledger leak (§5) invalidates its 8 runs, as the pilot's 8 registry-leak
   runs were once discarded. This is owner decision §7.1's neighbourhood.

## 8. Reproduce

```bash
python3 util/ad-hoc/2026-09-08_soak_label_to_transcript.py reports/soak/pointer_follow_soak.jsonl
python3 util/ad-hoc/2026-09-08_soak_label_to_transcript.py reports/soak/pointer_follow_soak.jsonl \
    --only-output-scored --evidence          # the raw hit + the tool that produced it
python3 util/ad-hoc/2026-08-21_soak_probe_evidence.py a6733960f4d68be8b   # screen now resolves
python3 -m unittest tests.test_soak_probe_evidence
```

## 9. Changed files

- `util/ad-hoc/2026-09-08_soak_label_to_transcript.py` — **new**; the resolver and
  mechanism-classifying re-audit.
- `util/ad-hoc/2026-08-21_soak_probe_evidence.py` — `SUBAGENT_DIRS` now names the
  primary project directory first; the stale worktree-suffixed entry is kept as a
  fallback. No behaviour change beyond resolving transcripts that previously resolved
  to nothing; `tests/test_soak_probe_evidence.py` passes unchanged (14 tests).
- `notes/JUNIPER_2026-09-08_JUNIPER-ML_SOAK-RETRIEVAL-STANDARD-EVIDENCE-RECOVERY.md` —
  this document.

**Not changed**: `reports/soak/pointer_follow_soak.jsonl`. No probe was run.

## 10. Follow-up not done here

The three false-positive mechanisms in §3 are defects in
`util/ad-hoc/2026-08-21_soak_probe_evidence.py`'s `scan()` and in
`util/soak_run_probe.py:358`'s `hit = doc in blob`. Hardening them changes screen
behaviour and needs its pinned tests updated, so it belongs in its own PR. The screen is
**unwired** (item F), so the blind spots cost nothing live today — but
`util/soak_run_probe.py` is not unwired, and its substring test is the one that scored
the three P24 rows.
