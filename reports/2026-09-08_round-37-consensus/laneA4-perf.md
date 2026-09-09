# Lane A4-perf — final report (verbatim, 2026-09-08)

**Lane verdict: PASS** — 9 claims: 6 CONFIRMED, 1 CONFIRMED-with-STALE-LINE, 1 PARTIAL, 1 ASSESSMENT. No claim refuted. Frozen trees verified: juniper-data `03b7548f` (clean), juniper-ml worktree `44de51c5` == `origin/main`.

## Verdict table

| # | Claim | Verdict | Doc | Mine |
|---|---|---|---|---|
| 1 | §2: 6 of 7 stores never called `super().__init__()`, incl. `LocalFSDatasetStore`, the store `api/app.py` wires | **CONFIRMED** | 6/7 | 7 concrete `DatasetStore` subclasses; at `c8f09fe^` six `__init__`s lack the call (`cached:40, hf_store:31, kaggle_store:35, local_fs:76, postgres_store:265, redis_store:43`), `memory.py:21` has it; `c8f09fe` adds it to exactly those six; `api/app.py:19,41-42` → `LocalFSDatasetStore`. Scoped census returns exactly 7; the only subclass outside the package is the test stub `_CountingStore` — §5.4 holds. |
| 2 | §2: order 100× at N=100/1,000; doc ranges 96.6–148.3× and 77.4–163.2× | **CONFIRMED (order); doc's ranges not reproduced** | ~100× | `filter_datasets(limit=100)` pre-fix ÷ post-fix warm: **N=100 137.3–180.1×**, **N=1000 154.3–173.2×**; 8 runs over 2 processes. Pre-fix `_list_all_metadata_cached` ÷ `list_all_metadata` = **0.7–1.1×** → cache was inert. |
| 3 | §0.4: `total` 0.0000041–0.000012% and zero in-tree consumers; `list_all_metadata()` 96.8%; ~3 ms at deployed N=21 | **PARTIAL** (conclusions hold; two numbers N-dependent with N unstated; "in-tree" should be "non-test") | as stated | Deployed N **= 21** (container `juniper-data`, `JUNIPER_DATA_STORAGE_PATH=/app/data/datasets`). Cold `filter_datasets` **2.67–2.90 ms**; warm 0.036–0.046 ms. `list_all_metadata` share **96.7–97.2%** at N=21. `total = len(filtered)` share **0.0015–0.0016%** at N=21, **0.0000037–0.0000039%** at N=10,000 — the doc's figure is the PR's N=10,000 number. Consumers: every non-test hit is the producer or docs; **27 test assertions** consume it; no other repo references the filter surface in code. |
| 4 | §0.4: `total` REQUIRED; `JUNIPER_DATA_API.md:63` / `:75` | **CONFIRMED** | — | `core/models.py:173-185` `DatasetListResponse`: **`total: int`** (`:177`, no default). `:63` "Response fields will NOT be removed within a major version"; `:75` "Removing a response field" under Breaking Changes. |
| 5 | §0.4 correction: pushdown at `postgres_store.py:519` is in `list_datasets`, not on `/filter`; Postgres `list_all_metadata` has no LIMIT/OFFSET | **CONFIRMED; STALE-LINE** | `:519` | `list_datasets` `:515`, SQL `:528`; `list_all_metadata` `:562`, `SELECT * FROM datasets ORDER BY created_at DESC` `:570`. Trace: `routes/datasets.py:387 @router.get("/filter")` → `:451 store.filter_datasets` → `base.py:462` → `:497 _list_all_metadata_cached()` → concrete `list_all_metadata()`. |
| 6 | §0.4: "A real fix pushes filter/sort/limit into each store" | **ASSESSMENT** | — | Natural only for **Postgres** (sort must be `(created_at DESC, dataset_id ASC)` to keep APD-DATA-012's total order). LocalFS: glob + per-file JSON — needs a sidecar index. Redis: `SCAN`+`GET` — needs index sets. InMemory: dict. Cached/HF/Kaggle delegate. "Each store" = 1 pushdown, 2 index builds, 4 delegations. |
| 7 | §5.2: blocked twice — `py/side-effect-in-assert`, `py/unused-import` on three `# noqa: F401` imports; fixed via `importlib.import_module` | **CONFIRMED (one rule omitted)** | 2 rules | Bot reviews: alerts **85 `py/side-effect-in-assert`** **and 86 `py/import-and-import-from`** on `085b0ae2`; **87/88/89 `py/unused-import`** on `0a30556b`. Fix `73a45ae1`; merged test uses `importlib.import_module(...)` `:357`. |
| 8 | §5.3/§5.4: suite non-vacuous; read-your-writes arms pass trivially while inert | **CONFIRMED by running** | — | Merged `test_metadata_cache.py` vs pre-fix storage: **3 failed / 24 passed** — failures exactly `test_the_cache_is_actually_live[LocalFS\|Cached\|HuggingFace]`; **all 12 read-your-writes arms PASS on the inert cache**. Post-fix: **27 passed**. |
| 9 | Register `APD-DATA-019` scoped onto `list_all_metadata()`, "do not re-file as `total`" | **CONFIRMED** | — | `:676` re-scoped; §4.1 note `:822-826`; on `origin/main` via `26e12019` (ml#1813). |

## New findings

1. **Instrument trap on N.** `grep -c meta.json` over the deployed dir reads **37**, because 16 `*.meta.json.lock` files match; the true count is 21.
2. **`total`'s share is N-dependent and neither the doc nor the register states N.**
3. **§5.2 omits a third rule**, `py/import-and-import-from` (alert 86).
4. **The register still carries "114.8×" at four significant figures** (`:676`, `:820`).
5. **§8 is stale**: ml#1813 merged the same day.
6. **Post-#381 steady state at N=21 is ~40 µs, not ~3 ms** — 2.8 ms is paid once per 5 s TTL window (`base.py:31`).
7. Census docstring names `_REQUIRES_EXTERNAL_SERVICE` while the set is `_NOT_CONSTRUCTIBLE_HERE` (`test_metadata_cache.py:348` vs `:263`). Cosmetic.
8. The PR's benchmark (`util/ad-hoc/2026-09-07_measure_filter_datasets_cost.py`, data#382) bypasses `save()` and its docstring still calls InMemory "the ONLY store whose cache is live".

## What the evidence cannot support

The doc's exact ranges and the PR's 114.8×/92.9× — machine- and instrument-specific; that the CodeQL reviews were the *merge blocker* rather than advisory; Postgres/Redis cache behaviour was not timed.
