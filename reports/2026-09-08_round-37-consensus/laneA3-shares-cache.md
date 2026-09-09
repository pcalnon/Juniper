# Lane A3-shares-cache — final report (verbatim, 2026-09-08)

**Lane verdict: PASS — 10/10 claims confirmed on substance; 0 refuted; 2 STALE-LINE, 1 PARTIAL (wording only), 1 definition-dependent at the quoted precision. 6 new findings, two of them material.**

Frozen tree `03b7548f` (clean); cache root `~/.cache/juniper_data/equities` (env var unset). Every script blocks the network (`_sec_get` patched to raise) and never writes to the real cache. Scripts graduated to `juniper-data/util/ad-hoc/2026-09-08_equities_shares_cache_census/` (juniper-data#388).

## Verdict table

| # | Claim | Verdict | Doc's value | My value (definition) |
|---|---|---|---|---|
| 1 | §4 defaults | **CONFIRMED** | start `2000-01-01` @ `defaults.py:17`; END `None`; USE_CACHE `True` | `:17`, `:18` (wall clock at `generator.py:248`), `:51` |
| 2 | §4 census | **CONFIRMED exact** | 485 / 503 / 500 / 15 named / 0 empty | 485 `.json`; CSV 503 tickers, 500 CIKs (GOOG/GOOGL, FOX/FOXA, NWS/NWSA share CIKs); missing set == doc's 15 exactly; 0 empty under the generator's own `not any(units.values())` |
| 3 | §4 first filing | **CONFIRMED** (definition-dependent) | earliest 2009-04-15, median 2009-12-18, mean 43.1% (09-07), 44.3%, 46.0%, universe 44.85% | Generator's effective date is `filed` (`:706-716`). Raw min-`filed` (n=485, calendar days): 2009-04-15 / 2009-12-18 / **43.141 / 44.272 / 46.042 / 44.847%**. Generator-faithful min-`filed` (after dedup + outlier filter): **43.220 / 44.353 / 46.126 / 44.923%** (+0.08 pp). `end`-date alternative: 42.92%. |
| 4 | §0.7 KO | **CONFIRMED exact** | 71 dei pts; +0.63822% for 40 d; 3 episodes 103 d | `0000021344.json` = dei/EntityCommonStockSharesOutstanding, 71 facts, 68 ends; 3 ends re-stated with the same value (8-K 2013-10-24, 8-K 2016-10-27, 10-Q/A 2024-05-30). On KO's cached OHLCV (6,643 sessions): 2013-02-27..04-24 **40 d, 4,485,161,506 vs 4,456,717,996 = +0.63822%**; 2016-02-25..04-27 44 d +0.45011%; 2024-05-02..05-29 19 d +0.10448%; **103 trading days, all OVERSTATED**. Business-day calendar: 41/45/20 = 106. |
| 5 | §0.6 outlier | **CONFIRMED** | `:79`; 61/485; causal 15 or 16 | `_SHARES_OUTLIER_FACTOR = 100.0`, filter `:932-934`. **61 CIKs lose ≥1 point** (92 points). Causal C1 (expanding median of survivors) **15**; C2/C3 **16**; the 16th is AIZ. |
| 6 | §0.6 adj_close | **CONFIRMED** | `:785`, `:625`, `:804` | `:785 auto_adjust=False`; `:625-626` fallback; `:804` rename. **`adj_close` IS a default feature column** — `defaults.py:138`, position 11 of 16. |
| 7 | §0.6 cost_basis | **CONFIRMED** | `:758`, inert at default, public field | `:754-758`; default `purchase_date = "2000-01-03"`, `basis_price_field = "close"`; `params.py:71-74` plain `str`, ISO-validated only; route builds it from the request verbatim. Also a default feature column (`defaults.py:137`). |
| 8 | §0.9 cache key / warm path | **CONFIRMED consequence; STALE-LINE; PARTIAL wording** | key `:866`, exists-check `:868`, OHLCV `:771` | Key now `:869`; `:871 if use_cache and cache.exists()`; OHLCV `:774`. No version/TTL/mtime anywhere. Warm hit → `:871-875` load; `:876-877` quality/origin forced; **`:878-909` skipped entirely** (concept loop incl. the `:895` guard, the `:903` ladder, the `:906` write); **`:910` empty-units guard RUNS** and returns `None` with 0 network / 0 ladder calls. |
| 9 | §0.10 comment | **CONFIRMED + strengthened** | `:733` says KO/ABT report no concept; KO has 71 | KO 71; **ABT (CIK 1800) has 68 dei points cached** |
| 10 | §0.6/§0.9 mtimes | **CONFIRMED** | June-2026 cache | All 485: **2026-06-03T01:18:28Z → 01:58:32Z**, one batch, ~9 h before #164's merge. Truthy-empty fix `04535bf` 2026-09-04 (#362); ladder `1141d0b` 2026-09-05 (#366). 473 dei + 12 us-gaap payloads. |

## New findings (not in the document)

**N1. §0.7 is population-scale and reaches the default 14-symbol prefix.** Same-value re-stated ends only, business-day calendar: **162/485 CIKs have a re-stated end (303 ends); 155 CIKs have ≥1 differing row; 17,569 rows; 7,742 OVER / 9,827 UNDER; 68 CIKs off by >1% (8,315 rows); max 66.3% (PLTR), TRGP 65%, TSCO 50%, DELL 48%, BF.B 2,603 rows.** In the default prefix `A…AEP`: **ADM 213 rows up to +11.55%, AAPL 64 rows 0.54%, ADSK 3 rows 0.81%.** Including corrections: 182 CIKs / 22,887 rows.

**N2. The same dedup DEFERS the first available count** — 9 CIKs (EXPE 729 d, NWS/NWSA 364, NXPI 349, GLW 96, TSCO 91, GOOG/GOOGL 82, SNDK 10, LITE 4, CF 2); 8 CIKs / 1,225 business-day rows ship NaN where the final figure was already public (EXPE 521). Expedia then regresses to a 2020-12-31 figure on 2022-02-11 (−8.0% for 57 days).

**N3. Six cached payloads yield silent placeholder counts on a warm cache** — TAP `0,0,0` (median 0 → the `median > 0` guard at `:933` *disables* the filter → market_cap 0 from 2009-11-05); DDOG `78M,0,0` → 0 from 2019-11-13; CVNA `0,0`; FOX/FOXA one point `val=1`; **PSKY `[1000, 1000, 1,071,666,977]` → the whole-history median (1000) drops the real count and keeps the placeholders**; BRK.B ≈1M 2009–2011. Not NaN, so `:726`, `fundamentals_fill="nan"` and `data_quality` are all silent. None in the default 14.

**N4. The cache-hit path erases provenance.** `:876-877` fix quality/origin before the branch and `:906-909` cache only `data`, so a rung-3/4 payload re-read warm is labelled `point_in_time`/`companyconcept`. `degraded` (`:281-282`) fires on the cold day and vanishes the next — and `end_date=None` mints a new `dataset_id` daily. ABNB is rung-3 and in the default 14 (instance inferred; no ABNB payload cached).

**N5.** §0.9's "0 of 485 empty … latent" is true, but three payloads (TAP, DDOG, CVNA) are all-zero — functionally worse than empty.

**N6.** Stale docstrings: `generator.py:966` "(n, 10)" and `params.py:52` "10-column" vs 16 feature columns.

## What the evidence cannot support

SEC's *live* endpoint (no network); whether the six placeholder payloads reflect June's endpoint output or dimensional-fact exclusion; trading-day counts for tickers other than KO; the ABNB day-2 instance of N4 (mechanism only).
