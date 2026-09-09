# Lane A2-source — final report (verbatim, 2026-09-08)

**FAIL** — 21 claims probed: **15 CONFIRMED, 2 REFUTED (A3, E-`-p`), 2 PARTIAL (A5, B1-§0.13 framing), 4 STALE-LINE** (505, 866/868, 3933; all stale *on the day written*). The two refutations are load-bearing: A3 underpins §0.3's "requires a schema change", and one of the "new" defects (§0.12) is under-scoped.

## Claims

| # | Claim | Verdict | Evidence | Correction |
|---|---|---|---|---|
| A1 | Three OR sites (§0.3) | CONFIRMED | `equities/generator.py:471`, `:501`, `csv_import/generator.py:153` — exactly three production sites; only other readers are tests + `api/settings.py:216/:223`. File last changed a1fc287 (09-06), so 471/501 were correct on 09-07. | — |
| A2 | Docstring / docs / CHANGELOG / test:726 | CONFIRMED | `generator.py:452-453` "cannot opt *out*"; `docs/REFERENCE.md:192,:890,:1490`; `CHANGELOG.md:405`; `test_csv_import_generator.py:726` exact name, passes alone on the frozen tree. | — |
| A3 | Plain `bool`, nothing reads `model_fields_set` ⇒ schema change needed (§0.3) | **REFUTED** | Executed: `EquitiesParams(allow_truncation=False).model_fields_set == {'allow_truncation'}`, omitted → `set()`. And the package **does** read `model_fields_set` — `csv_import/generator.py:151` for `max_bytes`. | Only the *resolution code* is blind. A one-line presence guard placed before binding would express option 3. juniper-data's documented reason for not doing so is the serialised-defaults client hazard (`csv_import/generator.py:136-140`), a policy choice, not a schema limitation. |
| A4 | `getattr` :146; binder only on `EquitiesGenerator` :505; seq calls `_resolve_symbols`; `dataset_id` collision (§0.12) | CONFIRMED / STALE-LINE | `datasets.py:146` ✓. `def bind_deployment_defaults` is **:508**. `equities_seq/generator.py:114` ✓. Bind mutates exactly `max_symbols` and `allow_truncation` via `model_copy` (`:525-526`). **Executed** env off→on: `equities-3.0.0-6dcb9091…` → `…9ffdd2cf…`; `equities_seq-3.0.0-1d6a3727b640a60e` **both times**. | Cite `:508`. Asymmetry is real. |
| A5 | Only 422 = `InputTooLargeError`/`IncompleteDataError`, solely under `not allow_truncation` | PARTIAL | Raise sites: equities `:290`, `:559`; csv `:236`, `:253` — under not-allowed. **`:301`** raises `IncompleteDataError` under `INCOMPLETE_DROP and not conditioned`, i.e. when truncation **is** allowed. Route `:187`→`:211` maps both to 422. | "solely" is wrong: one raise fires with allow=True (drop emptied the dataset). |
| A6 | `equities_seq` registered + advertised | CONFIRMED | `api/routes/generators.py:113`; `docs/REFERENCE.md` 7 mentions. | — |
| B1 | Flag from `Settings()` :3933→:3725; refuse-or-train (§0.13) | CONFIRMED / STALE-LINE | `manager.py:3956` (3933 is pre-#630). **No cascor-side refusal**: after a 200, `:3983` meta → `:3984` log → `:3998` build → `:4000` tensors. §0.13's scenario is **real**: cascor trains and annotates `false`. | See §3 for framing/remedy problems. |
| B2 | `dataset_shortfall` shape | CONFIRMED | `get_status` `:2821`; constructor `:3724-3729`; `None` at `:3706-3707`. | — |
| B3 | WS reach (§0.2) | CONFIRMED | `training_stream.py:96` sole WS caller → `initial_status`. Broadcast helpers: candidate_progress/cascade_add/event/metrics/state/topology. `create_state_message` reads `training_state.get_state()` `:1474`. | — |
| B4 | Frozenset; Literal excludes csv_import/equities_seq | CONFIRMED | `constants_api_defaults.py:139`; `models/training.py:235`; reader `:3957`. | — |
| B5 | Flag inert on `main.py` run path | CONFIRMED | `main.py:648-661` only exports the env var; sole runtime reader `manager.py:3956`; `main.py:500` builds `SpiralProblem`. | — |
| B6 | `_auto_start_training` | CONFIRMED | `app.py:482`; `:511-516` forwards nothing; `:556-557` `except Exception: logger.exception(...)`. | — |
| B7 | No shortfall on metrics | CONFIRMED | `get_metrics` keys (:2824-2882) none. | — |
| B8 | #624 presence test | CONFIRMED | `manager.py:3957`; pinned by `test_allow_truncated_datasets.py:192`. | — |
| B9 | #630 NaN guard | CONFIRMED | `:3824` def; `:3877-3882` six named arrays. | — |
| C | Canopy | CONFIRMED (+ hook exists) | 0× `allow_truncation`; `main.py:4193` → `cascor_service_adapter.py:1532` → `:1538` POST; `dashboard_manager.py:3003-3005` already populates it. **Canopy already polls** `/v1/training/status` (`adapter:1979/:2012`, `service_backend:222-224`, `state_sync:61`) and `normalize_status` carries `pending_dataset` through at `service_backend.py:308`. | "Canopy must POLL" is one carry-through line in `normalize_status`, not new plumbing. |
| D | recurrence#151 | CONFIRMED | `data.py:166-167` between X `:143` and dt `:181`. | — |
| E | `-p juniper_data.api.app` needed; env names | REFUTED (`-p`) / CONFIRMED (envs) | A pytest `-p <module>` early-import; **not needed**: data#333 (`650c91c`, 09-04) deferred `create_app`; the opt-out test passes alone, plain imports succeed. Envs: `JuniperCascor1` exists, `JuniperCascor-DEPRECATED` on disk. | Drop the `-p` advice. |

Bonus citations checked and correct: §0.6 `:785/:625/:804/:758/:79`; §0.7 `:918-923`; §0.10 `:733`; §4 `defaults.py:17`, `:751-752`; §0.3 deploy = 0; §0.5 doc-gap = 0 `.md`/`.env*`. §0.9 `:866`/`:868` → **`:869`/`:871`** (stale, pre-#369).

## New findings

1. **`equities_seq` bypasses the incomplete-data half of the contract entirely.** `_resolve_incomplete_policy` is called only from `EquitiesGenerator.generate` (`:285`); `equities_seq/generator.py` has **zero** occurrences of `unrescued`/`data_quality`/`IncompleteDataError`. An unrescued ticker in a seq request ships with NaN (or 0.0 under `fundamentals_fill="zero"`) `total_shares`/`market_cap` — no refusal, no annotation, regardless of `allow_truncation`.
2. **§0.13's remedy is not implementable as written.** The producer descriptor (`core/limits.py:261-269`) has no authority field, so cascor cannot "derive from the producer's response" *which* knob accepted. Cascor's only False-arm test (`:223`) uses empty meta — the scenario is untested.
3. **Auto-start never sets `_dataset_shortfall`.** `app.py` bypasses `_reload_dataset`, so an auto-started run on producer-forced partial data reports `dataset_shortfall: null`.
4. **Two in-tree sources of the WS misclaim**: `CHANGELOG.md:44-45` and the `manager.py:3991-3992` comment.
5. **Partial correction of stale lines**: the doc re-derived 468/498/148 but left 505, 866, 868 at pre-#369 values; 3933 is pre-#630.
6. `equities/generator.py:458-466` still describes the circular import data#333 removed.

## What the evidence cannot support

Whether any operator sets the env vars; §0.6 reachability at defaults; §4's percentages; ancestry checks (not assigned). No live truncated artifact was produced, so "trains" is a code-path reading of `:3983-4000`, not an observed run.

## Unfinished

The `equities_seq` fail-gate bypass deserves its own execution proof with a stubbed unrescued ticker (now pinned by juniper-data#388's `test_equities_seq_deployment_policy.py`).
