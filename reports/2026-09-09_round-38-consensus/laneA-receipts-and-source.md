# Round 2, Lane A — VERIFICATION by receipts and source (verbatim, 2026-09-09)

*(As returned by the lane. Four of its findings were re-derived by the orchestrating session and
acted on; see `README.md` § Reconciliation.)*

Document: `HANDOFF_2026-09-08_defect-register-round-38-...-made-truthful.md` (draft; since renamed
to `HANDOFF_2026-09-09_defect-register-round-38-the-three-way-prompt-shipped-and-two-corrections-that-reversed-themselves.md`).
Register as filed: `notes/JUNIPER_2026-08-14_JUNIPER-ECOSYSTEM_DEFECT-REGISTER.md` (uncommitted in
worktree `pure-toasting-token`). Predecessor:
`HANDOFF_2026-09-07_defect-register-round-37-...-last-mile.md`. All re-derived 2026-09-09.

**Placeholders (not errors):** `<<ML_PR>>` ×4, `<<CASCOR_633>>`, `<<CANOPY_605>>`, `<<DATA_388>>`,
`<<ML_STATE>>`, `<<ROUND2>>`, `<<X633>>`, `<<X605>>`, `<<X388>>`, `<<XREG>>` — 13 tokens, 10 distinct.

## Verdict table

| # | Claim (doc's number quoted) | Verdict | Evidence |
|---|---|---|---|
| 1 | §1 "FIXED rows **82**; `110 rows \| 82 fixed \| 28 open`; 82/82/82 AGREE" | **CONFIRMED** | From worktree root on the register AS FILED: grep → `82`; `register_open_set.py` → `110 rows \| 82 fixed \| 28 open`; `register_status_crosscheck.py` → `§4 110 rows, 82 **FIXED / §2 82 ids / §5.1 82 rows — AGREE` |
| 2 | §1 "not comparable to round 37's **79** without §4.9" | **CONFIRMED** | Same three commands on `origin/main`'s register: `79`; `96 rows \| 79 fixed \| 17 open`; `AGREE`. 110−96=14 §4.9 rows; 28−17=11 open ⇒ 3 FIXED at filing |
| 3 | §2 all three PRs merged | **CONFIRMED** | cascor#633 MERGED 2026-09-09T17:57:06Z `44dafe01`; canopy#605 19:25:55Z `b587e54e`; data#388 17:14:32Z `d7f4be5b`. All three `merge-base --is-ancestor origin/main` → true |
| 4 | cascor: `accepted_by_this_run`/`acceptance_source`; sources `request_params`/`allow_truncated_datasets`/`producer`; refusal opens `[dataset_shortfall_refused]` | **CONFIRMED** | `origin/main:src/api/lifecycle/manager.py:3706,3764,3765,3776-3778,3703`; `constants_api_defaults.py:148,157-159` |
| 5 | §2 / register §5.1 "the `juniper-cascor-model` mirror **keeps the drift test green**"; §5.3 "byte-for-byte … or `Test (Python 3.12)` fails" | **REFUTED** | `pytest juniper-cascor-model/tests/test_drift.py` on cascor `main` → **1 failed, 2 passed**. `diff -r` of the two `constants_api/` trees at `origin/main`: model copy carries an extra `__all__ = [4 SHORTFALL names]` at lines 165-171 that `src/` lacks. `gh pr checks 633` → **`Test (Python 3.12)  fail`**; `runs?head_sha=44dafe01` → **`CI — juniper-cascor-model \| failure`** on main |
| 6 | `_PROJECT_API_SHORTFALL_*` byte-identical in both dirs | **PARTIAL** | The four *definitions* are byte-identical; the containing *file* is not (row 5) |
| 7 | canopy: `dataset_shortfall` survives `normalize_status`; `PARTIAL_DATA_POLICY_FIELDS`; `· partial data`; `detail_full`; manifest rows | **CONFIRMED** | `service_backend.py:314`; `dataset_schema.py:103`,`:106`; `dashboard_manager.py:6964`, `:157-159,7748,7896`; `control_manifest.py` +30 lines = **exactly 3** `ControlContract` rows |
| 8 | §2 "**24 new tests**" / §0.3 "24 arms" | **REFUTED** | `git diff b587e54e^1 b587e54e -- src/tests/*` → **27** added `def test_`; arms = 23 funcs in `test_dataset_shortfall_prompt.py`, two parametrized = 28, plus 4 elsewhere = **32 arms** |
| 9 | data: `bind_deployment_defaults`, shared `_apply_incomplete_policy`, the new test file, census dir | **CONFIRMED** | `equities_seq/generator.py:90,119`; `equities/generator.py:469` called at `:271` and `equities_seq/generator.py:172`; census dir = **13 files** |
| 10 | register §5.1 "**228 passed**" | **REFUTED** | All nine matching test files on a `git archive origin/main` copy → **`220 passed`**. Same nine in the PR worktree → **`220 passed`**. Per-file collect: 93+50+19+19+12+9+8+6+4 = 220. 228 is unreachable by any subset |
| 11 | register §5.1 "**28 passed** in `test_allow_truncated_datasets.py`" | **CONFIRMED** | → **`28 passed`** |
| 12 | §4 ml#1813 "Merged 2026-09-07T15:15:42Z (`26e12019`), 67 min before #1818" | **CONFIRMED** | `1813` → `2026-09-07T15:15:42Z`, `26e120195c12…`; `1818` → `16:22:35Z`. Δ = 66 min 53 s |
| 13 | §4 `model_fields_set` distinguishes explicit `false` from omitted | **CONFIRMED** | Probe on the real `EquitiesParams`: omitted → not in `model_fields_set`; `allow_truncation=False` → in it. (Scoped to construction; see lane B1 attack 1 for what the binder then does) |
| 14 | §4 "canopy already polls at 1 Hz" | **CONFIRMED** | `status_cache.py:102` `REFRESH_INTERVAL_SECONDS = 1.0` |
| 15 | §4 "**Three sites**" (`or settings.*`) | **CONFIRMED** | `csv_import/generator.py:153`, `equities/generator.py:432`, `:462`. Pin at `test_csv_import_generator.py:727` |
| 16 | §4/§0.3 finiteness precondition — cascor#630 refuses *after* the shortfall is accepted | **CONFIRMED** | `_build_dataset_shortfall` at **`:4094`**, `_artifact_to_tensors` at **`:4096`** — annotation strictly first |
| 17 | §4 "juniper-ml's own handoffs mention the flag"; true for operator docs | **CONFIRMED** | cascor `main`: flag only in `CHANGELOG.md`, both constants files, `manager.py`, `settings.py`, `main.py`, one test — **not** in `AGENTS.md`, `.env.example`, `docs/` |
| 18 | §4 "Three SHAs listed; six PRs listed" | **CONFIRMED** | Round-37 §2 shows `c8f09fe`, `1ea2062`, `e5679b0` under "All four merge SHAs verified"; "round 36's five PRs" lists six |
| 19 | §4/§6 "**three** juniper-data stash entries" | **CONFIRMED** | `git stash list`: 3 |
| 20 | §4 `-p juniper_data.api.app` "stale since data#333" | **CONFIRMED** | data#333 merged 2026-09-04; my 220-test run used no `-p` and collected cleanly |
| 21 | §4 anchors `:508`, `:869/:871`, `:774`, `:3956` at the frozen SHAs | **CONFIRMED** | All four exact at data `03b7548f` / cascor `d39d537e`. **All four already stale TODAY** on `origin/main`; the shares-cache key has moved to `generator.py:925` |
| 22 | §4 "**Three** are all-zero and six carry placeholders" | **REFUTED (first half)** | My scan of 485 payloads: **2** all-zero — TAP, CVNA. The source instrument prints *"CIKs whose surviving series contains an exact 0: 3"* (DDOG contains one). Containing a zero ≠ all-zero. "Six" CONFIRMED as scoped to the **generator-surviving** series |
| 23 | §4 "43.14 raw min-`filed`, 43.22 generator-faithful" | **CONFIRMED** | `final_checks.py` re-run: 43.141% / 43.220% |
| 24 | §4/§0.4 KO "3 episodes, 103 trading days, +0.638/+0.450/+0.104%" | **CONFIRMED** | trading-day 40/44/19 = **103**; business-day 41/45/20 = 106; all OVERSTATED |
| 25 | §5.2 force-update "may fire no CI"; "7 workflow runs on the new head" | **PARTIAL** | `aecd0bb8` → **7**; `5179bca2` → **12**. The force-update fired 7, not none |
| 26 | §5.2 `push_signed_commit.py` "(new)"; §2(d) the ml PR delivers it | **REFUTED (§2(d))** | Already on juniper-ml `main`: added by **ml#1830** (`f924d22f`, 2026-09-09T00:25:17Z), extended by **ml#1853** (`469e316d`, 18:52:26Z). A second copy landed in **ml#1829**. Its documented contract is accurate |
| 27 | §5.3 mirrored dirs | **CONFIRMED** | `_EXTRACTED_DIRS = ("candidate_unit", "utils", "log_config", "cascor_constants")` |
| 28 | §5.6 `-q` in all three repos' addopts | **CONFIRMED** | data `:221-227`; cascor `:203-209`; canopy `:356-362` |
| 29 | §5.9 three recovered recipes | **CONFIRMED** | `safe_merge.py:889`; `test_equities_generator.py:976` inside `test_shares_are_not_visible_before_they_were_filed` (`:948`); the memory carries the 3-row wire-form table verbatim |
| 30 | §4.9 — 14 rows, every `Source` path/symbol exists TODAY, 3 `**FIXED` rows cite MERGED PRs | **CONFIRMED** | 8 + 6 = 14. Every anchor probed on each repo's `origin/main`, including `dataset_type` Literal `api/models/training.py:235` (excludes `csv_import` **and** `equities_seq`) |
| 31 | §0.4 "no deployment sets either env var" | **CONFIRMED** | juniper-deploy → zero hits |
| 32 | §0.5 recurrence has no `data_quality` consumer | **CONFIRMED** | zero hits |
| 33 | §0.6 `val_ratio` renders, `train_ratio`/`test_ratio` excluded | **CONFIRMED** | `dataset_schema.py:82-89` |
| 34 | §6 three worktrees still present | **CONFIRMED** | All three exist |
| 35 | §2 "two commits, `aecd0bb8` and `5179bca2`" | **PARTIAL** | `pulls/633/commits` → **9** commits; PR head at merge was `0c1a0f84`, not `5179bca2` |

## New findings not in the document

1. **cascor `main` is RED on `CI — juniper-cascor-model` today, and cascor#633 put it there.** The
   model copy gained a 7-line `__all__` block `src/` did not. `Test (Python 3.12)` was already
   **fail** on the PR when it merged. The document's own §5.3 describes exactly this failure mode,
   and this arc's PR fell into it — while §2 and the register's `APD-CASCOR-007` row both assert the
   opposite. Highest-value correction available: a live red main plus a false verification receipt
   now in the register.
2. **The register's data#381 range disagrees with round 37's.** As filed: "96.6–180.1× … 77.4–173.2×".
   Round-37 §2: "96.6–148.3× and 77.4–163.2×". Not re-derivable in this lane.
3. **The placeholder cohort is scoped to *surviving* series; the raw cache is worse.** Beyond the
   six, five more CIKs carry impossible raw counts the filter removes, plus four millions-scale unit
   errors mid-series. **Eleven** CIKs carry a value ≤ 1000. `APD-DATA-044` lacks the "surviving"
   qualifier the instrument applies.
4. **§2(d)'s deliverable list is stale.** `push_signed_commit.py` shipped in ml#1830 and was revised
   in ml#1853 before this arc's ml PR exists.

## What the evidence cannot support

- Whether cascor#633's drift failure was noticed and accepted or missed; `Test (Python 3.12)` is
  evidently not a required check, but I did not read the ruleset.
- Which of the two data#381 measurement ranges is right; neither was re-run.
- The unfilled placeholders.
- `43.14%`/`43.22%` are means over the **cached** universe with the network blocked.
- Whether `d39d537e` was cascor `main`'s tip on 2026-09-08 — only that it is an ancestor of today's.
