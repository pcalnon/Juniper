# Round-38 one-shot PR edit scripts (provenance, 2026-09-08)

The edit scripts one session ran, once each, to build the three code PRs of the round-38
defect-register arc — kept here under the script-placement rule (`AGENTS.md` § Script placement:
a script that modifies repository content may not live in `/tmp/`, which is reaped). They are
**not** re-runnable as-is: each targets, by absolute path, a PR worktree under
`/home/pcalnon/Development/python/Juniper/worktrees/` that existed on 2026-09-08, and each applies
exact-match string edits that refuse once applied. Their effects are the PR diffs; their value here
is the record of *what* was edited and *why* (the docstrings), and the edit texts themselves.

Handoff: `prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_defect-register-round-38-the-three-way-prompt-shipped-and-two-corrections-that-reversed-themselves.md`.

| Script | PR | What it did |
|---|---|---|
| `cascor_pr_a_edit.py` | juniper-cascor#633 | `src/api/lifecycle/manager.py`: `_describe_dataset_fetch_failure` with the refusal token; `_build_dataset_shortfall` with `accepted_by_this_run` / `acceptance_source`; `_as_bool_stance`; `_log_dataset_shortfall`; the `_reload_dataset` stance resolution; the WS-stream comment. Plus the four `_PROJECT_API_SHORTFALL_*` constants in `src/cascor_constants/constants_api/` (mirrored into `juniper-cascor-model/` by hand afterwards). |
| `cascor_pr_a_tests.py` | juniper-cascor#633 | The new arms of `src/tests/unit/api/test_allow_truncated_datasets.py` (28 tests after). |
| `cascor_pr_a_changelog.py` | juniper-cascor#633 | The four `## [Unreleased]` → `### Fixed` bullets (first placement, on the PR's base). Later scripts extract its `NEW` text with `ast` rather than importing it. |
| `cascor_changelog_rebuild.py` | juniper-cascor#633 | Re-placement after cascor#631 moved `CHANGELOG.md`: bullets inserted AFTER the NaN-guard entry so the 3-way merge sees a distinct region. |
| `cascor_changelog_rebuild2.py` | juniper-cascor#633 | Re-placement after the v0.11.0 release (cascor#635) moved the whole `[Unreleased]` block: bullets open a fresh `### Fixed` under the now-empty `## [Unreleased]`; the branch itself was then rebuilt from `main`'s tip through a temp ref — **which round-2 validation showed was unnecessary**: re-applying the entry on `main`'s file merges clean, and §5.2 of the handoff records the disproof. |
| `canopy_prompt_edit.py` | juniper-canopy#605 | The three-way partial-data prompt: `src/backend/protocol.py`, `src/backend/service_backend.py` (`dataset_shortfall` through `normalize_status`), `src/dataset_schema.py` (`PARTIAL_DATA_POLICY_FIELDS`), `src/frontend/dashboard_manager.py` (modal, store, callbacks, status-bar `· partial data`, `detail_full`), the manifest rows, `CHANGELOG.md`, and the tests. |
| `data_pr_finish.py` | juniper-data#388 | Finished the task-executor's PR after it was rate-limited: the shared `EquitiesGenerator._apply_incomplete_policy`, the `equities_seq` binder + policy call, comment fixes, the invented "#385" citation removed. |
| `data_fix_seq_fixture.py` | juniper-data#388 | `test_equities_seq_generator.py`: the synthetic filings moved INSIDE the mocked frame (they were unreachable, so every seq test ran on all-NaN fundamentals — handoff §5.4). |
| `data_fix_cap_fixture.py` | juniper-data#388 | `test_equities_seq_symbol_cap.py`: same fixture defect; the download-raise test given real shares. |
| `data_graduate_instruments.py` | juniper-data#388 | Moved the round-37 SEC-cache census instruments into `juniper-data/util/ad-hoc/2026-09-08_equities_shares_cache_census/` with headers and a README. |
| `data_changelog_reapply.py` | juniper-data#388 | Re-applied the PR's `CHANGELOG.md` entries on `main` after data#385 moved the file (precision withdrawal + two `### Fixed` entries). |
| `_graduate.py` | — | Copied the eleven above out of the session scratchpad and stamped the `util/ad-hoc` header into each docstring. |

The register scripts for the same round are `util/ad-hoc/register_round38_file.py` (re-runnable with
`--dry-run`), `register_round38_dates.py` and `register_round38_round2_fixes.py`; the cascor drift
repair is `util/ad-hoc/cascor_fix_shortfall_all.py`. The signed follow-up pusher used here,
`util/ad-hoc/push_signed_commit.py`, is **not** from this round — it landed in ml#1830 and was
extended by ml#1853.
