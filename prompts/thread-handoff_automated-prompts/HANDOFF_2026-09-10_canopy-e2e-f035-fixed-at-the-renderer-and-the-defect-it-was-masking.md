# HANDOFF — canopy E2E arc: F-CANOPY-035 read out of the renderer, FIXED and merged, and the defect that was hiding behind it

**Date**: 2026-09-10 · **Session**: <https://claude.ai/code/session_01SDwaTPuGgzypx9f1tE1ahB>
**Worktree**: `/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/glittery-wondering-cosmos`
**PRs**: juniper-canopy **#613 MERGED** (`b792256`) · juniper-ml **#1878 OPEN**

**Documents REFERENCED** (the ecosystem convention in `/home/pcalnon/Development/python/Juniper/AGENTS.md`
§ Cross-Project Conventions requires the filename on every citation, because more than one is cited):

- `notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md` — the finding ledger, the arc's
  document of record; this session added **Phase 6 — 2026-09-10** at its end
- `notes/JUNIPER_2026-08-08_JUNIPER-CANOPY_E2E-CLICK-BY-CLICK-TEST-MATRIX.md` — the row matrix
- `prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-08_canopy-e2e-arc-f035-supersession-measured-and-the-relay-drop.md`
  — the handoff this session inherited; its §3 item 1 is now closed
- `util/ad-hoc/README.md` — the instrument inventory, with a new section for this session's five tools

**Documents CHANGED by this session**: `notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md`,
`notes/JUNIPER_2026-08-08_JUNIPER-CANOPY_E2E-CLICK-BY-CLICK-TEST-MATRIX.md`, `util/ad-hoc/README.md`, and
this file. **Added**: five instruments under `util/ad-hoc/` (all `2026-09-10_*`) and 14 evidence artifacts
under `reports/e2e-canopy-2026-09-02/transcripts/`. **In juniper-canopy**: `src/canopy_constants.py`,
`src/frontend/dashboard_manager.py`, `src/tests/unit/frontend/test_poll_gating.py`,
`src/tests/unit/frontend/test_stage2_global_lane.py` (PR #613, merged as `b792256`).

---

## 0. PREFLIGHT

1. **`uptime -s` before trusting any leg.** The host has not rebooted since **2026-09-07 23:12**, so every
   leg from Phase 5 was still up when this session ran. `/tmp` is tmpfs; a reboot destroys `/tmp/juniper-e2e`
   and the fixture lives in cascor's process.
2. **There are now TWO canopy verify legs and they are different builds.** `:8052` serves `eb05021d`
   (**before** the fix) and `:8053` serves `eab7cf43` (**after**). Neither is the trio's `:8051`, whose
   browser instruments still default to it — **`JUNIPER_E2E_CANOPY_URL` must be exported for every probe**,
   the failure this arc has paid for three times now.
3. **canopy `main` already carries the fix** (`b792256`). The `:8053` leg was launched from the fix worktree
   `worktrees/juniper-canopy--fix--f035-metrics-store-running-guard--20260910-0459--8cfb29ac`, whose tree is
   clean and whose local commit `eab7cf4` is content-identical to what merged. **Do not delete that worktree
   while `:8053` runs.**
4. **The fixture is untouched.** uuid `1cd15120…`, 2/52/2/1538, `COMPLETED`, epoch 56, **66 metrics rows**
   (`output` 54 / `candidate` 12). Snapshots `snapshot_20260905T103912Z` (40), `…20260908T123427Z` (48),
   `…20260909T002658Z` (52). No growth run was done this session.

---

## 1. Goal statement

Continue the juniper-canopy E2E validation arc. Ledger
`notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md`, matrix
`notes/JUNIPER_2026-08-08_JUNIPER-CANOPY_E2E-CLICK-BY-CLICK-TEST-MATRIX.md`.

State: **63 findings — 42 fixed / 1 accepted / 2 withdrawn / 18 open (0 P0, 3 P1, 15 P2)**; it was
62 / 41 / 1 / 2 / 18 with 4 open P1. Matrix **298 rows, 296 verdicted** (the known slash-enumeration
artifact — both `M-PARAMETERS-01/02/03` rows carry PASS in the matrix itself), **21 BLOCKED**. Count
BLOCKED with a pattern that has **no trailing pipe**, or `C2.10-03`'s `BLOCKED (F-CANOPY-025)` is missed
and you get 20.

**What this session settled.** F-CANOPY-035's mechanism, read out of the shipped dash-renderer bundle
(`dash_renderer.dev.js`, unminified in `JuniperCanopy1`, dash 4.2.0) rather than inferred from behaviour:
`:2698` discards a response whose callback has left `watched`; `:3027` evicts a `watched` entry the instant
the same identity appears in `requested` (`concat(watched, requested)` grouped by `getUniqueIdentifier`,
each group sliced `[0:-1]`, `requested` concatenated LAST). `getUniqueIdentifier` hashes **one callback's
own** inputs/outputs/state, so the other nine fast-lane callbacks are different identities and **cannot**
evict it — which separates self-eviction from fast-lane promotion starvation **at the source**, and is what
the inherited handoff's §3 item 1 asked for.

**The displacing event is the TICK creating a `requested` entry**, not the next HTTP request. Every prior
measurement on this finding — and this session's first one — used the HTTP boundary, which is why the
ledger carried four numbers read as evidence *against* the mechanism. Three were boundary errors; the
fourth (`everSeen: {watched: 1}` — two concurrent entries never observed) is what the mechanism
**predicts**, because the eviction is synchronous with the insertion.

**What closed it.** A live dose-response on the unfixed leg (n=2) in which the **sign of (observed tick gap
− round trip) predicts the outcome in all four phases**; and an ~80-line clean room with **no canopy at
all** that reproduces the defect and shows `running=` fixing it.

**The fix (canopy#613, merged).** `update_metrics_store` gets its own `dcc.Interval` and
`running=[(Output(<that interval>,"disabled"), True, False)]`, which makes a re-request during flight
structurally impossible. Live: the store went from **0 across a whole 90 s window and 53 full-payload
responses** to filling **~7 s after page load** with the fast lane still ticking at 1 Hz.

**What this session did NOT get.** M-CANDIDATES-07 is still **FAIL**. With the store provably good, the
candidate loss figure renders in only **2 of 5** loads — filed as **F-CANOPY-052**, the defect the
permanently-empty store was masking.

---

## 2. State at handoff

| | |
|---|---|
| Findings ledger | **63 — 42 fixed / 1 accepted / 2 withdrawn / 18 open (0 P0, 3 P1, 15 P2)** |
| Matrix rows | 298, 296 verdicted / **21 BLOCKED**; M-CANDIDATES-07 FAIL, basis re-attributed to F-CANOPY-052 |
| cascor fixture | uuid `1cd15120…`, 2/52/2/1538, `COMPLETED`, 66 metrics rows (`output` 54 / `candidate` 12) |
| Services | `:8051` canopy (trio, `git_sha null`, v0.4.0) · `:8052` canopy `eb05021d` v0.6.0 — **the BEFORE leg** · `:8053` canopy `eab7cf43` v0.6.0 — **the AFTER leg** · `:8101` data 0.13.0 · `:8202` cascor `d39d537` (dirty; content-proven identical to merged `5eb6f144`) · `:8050`/`:8201`/`:8211` Docker deploy stack — do not touch |
| PRs | juniper-canopy **#613 MERGED** `b792256` · juniper-ml **#1878 OPEN** (signed `fcb3ad8b`, 22 files) |
| Product code changed | juniper-canopy only (#613). No cascor, no data. |

---

## 3. What is still owed (in order)

1. **F-CANOPY-052 — the wire census on a NON-rendering run.** This session captured it only on a
   *rendering* run (consumer fired exactly once, carried a one-trace figure, applied). The census on a miss
   is what separates "never fired" from "fired and was not applied" — opposite fixes, and this arc has
   twice returned a confident verdict that could not tell them apart. Instrument exists:
   `util/ad-hoc/2026-09-10_f035_downstream_consumer_probe.py` (**always `--no-force`**, see trap 3).
2. **M-METRICS-11..16/-18 re-drive on `:8053`.** All three replay callbacks compute
   `max_index = len(metrics_data) - 1 if metrics_data else 0` from the store #613 repairs
   (`metrics_panel.py:1013/1060/1088`), so the *index* rows were never testable before and are now.
   `util/ad-hoc/2026-09-08_replay_block_redrive.py` exists and has **not** been run against the fixed leg.
   The play toggle and speed buttons are data-independent and remain F-CANOPY-048's own.
3. **F-CASCOR-004 / F-CANOPY-049** — unchanged from Phase 5 and untouched here. cascor: log in `_send_json`,
   `close()` in `broadcast`'s drop path. canopy: **not** a new liveness rule — `StreamHealth` already
   degrades after 60 s and `cascor_service_adapter.py` re-arms it every 30 s off cascor's transport pings.
   `util/ad-hoc/2026-09-08_cascor_ws_drop_probe.py`'s transport-counter rule has still **never been
   exercised** — exercise it before relying on it.
4. **The fix's ~4–5 s re-enable overhead.** Measured (cadence ~7.3 s at a 1000 ms period; 250 ms gave
   ~6.1 s, so it is overhead-bound not period-bound) and **not explained**. The plausible reading is the
   `runningOff` prop update waiting on a renderer cycle contended by the fast lane — a hypothesis. This is
   the one thing that would let the poll run near 1 Hz again.
5. **Phase 5's items 3, 5, 6, 7 and 9** are untouched: M-CANDIDATES-09..11 one tab at a time,
   M-DATASET-17..26 (which arm loads a sequence dataset — owner question), the dispatch probe's control
   under a rule fixed BEFORE the run, still-owed item 7 (both legs' commits in every artifact), and
   M-TOPOLOGY-16's fade half.

---

## 4. Instruments added (all `util/ad-hoc/2026-09-10_*`)

| instrument | answers |
|---|---|
| `2026-09-10_f035_unopposed_response_test.py` | per-response bracket: did THIS response land, and could anything evict it. **Read the docstring** — its first verdict used the wrong boundary and is marked superseded |
| `2026-09-10_f035_trigger_period_sweep.py` | the dose-response that closed the mechanism |
| `2026-09-10_f035_running_guard_cleanroom.py` | reproduce with no canopy; test `running=` as the fix |
| `2026-09-10_f035_fix_wiring_check.py` | the fix's six wiring properties, off the BUILT app |
| `2026-09-10_f035_downstream_consumer_probe.py` | F-CANOPY-052: data or render, and does the consumer fire |

---

## 5. Verify the starting state

**These numbers are the state AFTER juniper-ml#1878 merges.** Until then `main` reads 62 / 41 / 1 / 2 / 18
with 4 open P1 — if you see that, the record has not landed yet, not been reverted.

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-ml   # or a fresh worktree
uptime -s                                                 # after 2026-09-07 23:12 → the legs below are gone
gh api repos/pcalnon/juniper-canopy/pulls/613 --jq '{merged,merge_commit_sha}'   # true, b792256…
gh api repos/pcalnon/juniper-ml/pulls/1878   --jq '{state,merged}'
python3 util/ad-hoc/e2e_finding_triage.py --note notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md | tail -8
python3 util/ad-hoc/e2e_row_coverage.py | head -4        # "remaining: 2" is the KNOWN artifact
grep -cE '^\| [A-Z0-9.-]+ .*\| BLOCKED' notes/JUNIPER_2026-08-08_JUNIPER-CANOPY_E2E-CLICK-BY-CLICK-TEST-MATRIX.md   # 21 — no trailing pipe
ss -ltn | grep -E ':(8051|8052|8053|8101|8202)\b'
curl -s http://127.0.0.1:8053/v1/health | grep -o '"git_sha":"[0-9a-f]*"'   # eab7cf43… — the FIXED leg
curl -s http://127.0.0.1:8052/v1/health | grep -o '"git_sha":"[0-9a-f]*"'   # eb05021d… — the BEFORE leg
curl -s http://127.0.0.1:8202/v1/network                                     # uuid 1cd15120…, hidden_units 52
grep 'WS emission summary' /tmp/juniper-e2e/logs/juniper-cascor.log | tail -1   # N active connections, N > 0
```

---

## 6. Traps (new; Phase 5's still apply)

1. **The renderer's eviction boundary is the TICK, not the HTTP request.** A call with no HTTP successor in
   flight can still have had a `requested` entry created inside it. Any classification of "opposed" built
   on request start times measures the wrong thing — that error produced a confident
   `SUPERSESSION-INSUFFICIENT` in this very session.
2. **A `dcc.Interval` does not tick at its nominal rate under load.** 54 ticks per 90 s at a 1000 ms
   period, ~0.6 Hz. Read the gap from observed `n_intervals` transitions, never from the constant.
3. **Forcing a store change can destroy the data you are testing.** Setting `window_size: 40` keeps the
   LAST 40 rows, and the 12 candidate entries sit EARLY in the history — so the force drops exactly the
   rows the candidate figure needs. `2026-09-10_f035_downstream_consumer_probe.py --no-force` exists for
   this reason.
4. **On a FIXED leg the store fills during page load**, so an observer installed after the tab settle reads
   it already full and records no transition. Use `--early-observer`.
5. **`e2e_finding_triage.py` reads only the LAST 170 characters** of a finding's bold header
   (`tail = body[-170:]`). A `FIXED` placed early in a long header is invisible and the finding still
   counts as open. Put the disposition at the END. (Also: the header must be a bold `**F-… — …**` line at
   column 0 — an `###` heading is not counted at all.)
6. **`running=` is not on the `callback_map` entry.** `dash/_callback.py:326` puts it on the callback SPEC,
   in `app._callback_list`, which is what is served as `_dash-dependencies`. A test that asserts it off
   `callback_map` silently reads `None` and passes for the wrong reason.
7. **Force-resetting a PR branch closes the PR.** Pushing the branch back to `main` to rewrite a commit
   made GitHub auto-close canopy#613 (zero commits at that instant); it reopened cleanly via
   `gh api -X PATCH …/pulls/613 -f state=open` once the new commit landed. Expect it; do not re-cut.
8. **`open_signed_pr.py --commit-body-file` takes a BODY, not a full message.** Passing the whole message
   duplicates the subject line, and juniper-canopy's `squash_merge_commit_message` is `COMMIT_MESSAGES`, so
   the duplicate would reach `main`.
9. **`2026-09-08_append_signed_commit.py` reports a network failure as "branch not found".** A TLS
   handshake timeout printed `REFUSED: branch … not found`. Re-check the ref before believing it.

---

## 7. Repository state at handoff

**juniper-ml** — everything is in ONE signed commit `fcb3ad8b` on
`docs/canopy-e2e-2026-09-10-f035-fixed-and-f052`, PR **#1878**, 22 files (3 modified, 19 added), created
through the GitHub API (a local commit hangs on a YubiKey touch that never comes in a headless session).
The worktree `glittery-wondering-cosmos` still holds those files uncommitted locally — the API commit is
the record; **do not push the local tree over it**.

**juniper-canopy** — `main` at `b792256` carries the fix. The fix worktree
`juniper-canopy--fix--f035-metrics-store-running-guard--20260910-0459--8cfb29ac` is clean, carries local
commit `eab7cf4` (content-identical to the merged squash), and **is serving `:8053`** — do not remove it
while that leg runs. Its branch `fix/f035-metrics-store-running-guard` was deleted on merge.

**juniper-cascor / juniper-data** — untouched.

---

## 8. Validation status — NOT independently validated

This document and Phase 6 of `notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md` have
**not** been through the consensus procedure in
`notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_INDEPENDENT-AGENT-CONSENSUS-PROCEDURE.md`. Phase 5 failed
round 1 with eight claims refuted or downgraded, so treat the following as the claims most worth attacking:

- **The 2-of-5 render rate for F-CANOPY-052** is n=5 on one leg on a `COMPLETED` fixture, and two of the
  five non-renders came from probe runs that differed in other ways. It is enough to refuse a PASS; it is
  not a characterised rate.
- **The ~5 s re-enable overhead** has one explanation offered and none tested.
- **The dose-response is n=2** and its two phases differ in round trip as well as period (less contention
  at 2000 ms), so the threshold's *location* is softer than the threshold's *existence*. The sign test
  holds in all four phases; the exact crossing point does not.
- **The clean room is n=1 per arm.**
- Every count, sha, port, snapshot id and file inventory in Phase 6 was read from an archived artifact.
