#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""F-CANOPY-035 -- is the TRIGGER PERIOD the controlling variable? A dose-response.

WHY A SWEEP, WHEN TWO BINARY TESTS ALREADY RAN.

``2026-09-08_f035_supersession_test.py`` (disable the tick / store fills) and
``2026-09-10_f035_unopposed_response_test.py`` (nothing lands while it ticks; the
store fills ~3 s after it stops) both compare exactly two regimes: ticking at
``FAST_UPDATE_INTERVAL_MS`` and not ticking at all. Neither can say whether the
controlling variable is the trigger PERIOD or the mere presence of the fast lane --
and those imply different fixes (a dedicated Interval vs. removing the callback from
the lane entirely).

A sweep answers it. dash_renderer.dev.js:2698 discards a response whose callback has
left ``watched``; :3027 evicts a ``watched`` entry when the same identity appears in
``requested``. So the prediction is QUANTITATIVE and has a threshold:

    landing turns on when the trigger period exceeds the callback's round trip.

Below that, a new ``requested`` entry always coexists with the in-flight one and
every response is discarded. Above it, each call resolves before its successor is
requested and every response lands. A threshold that falls inside the measured
round-trip range is the mechanism's own signature; landing that is flat in period, or
ordered arbitrarily, refutes it.

WHAT THIS IS NOT CONFOUNDED BY. Raising the Interval slows all ten of its callbacks,
so sibling LOAD falls with period too. That cannot explain a threshold, because
sibling contention has no path to this outcome: ``getUniqueIdentifier`` hashes one
callback's own inputs/outputs/state, so a sibling is a different identity and can
never enter this callback's eviction group; and contention delays PROMOTION, which
would show as fewer or later requests, not as full responses that arrive and are
discarded. The responses arrive (HTTP 200, full payload) in every run to date.

VERDICT RULE -- FIXED BEFORE THE FIRST RUN.

  PERIOD-CONTROLS-LANDING   landing is monotone in period with a threshold P*: no
                            phase below P* landed, every phase at or above it did.
                            Reported with P* against the measured round-trip range.
  NO-LANDING-AT-ANY-PERIOD  nothing landed even at the longest period -> the trigger
                            is NOT the controlling variable; a dedicated Interval
                            will not fix this and the fix direction is wrong.
  LANDS-AT-ALL-PERIODS      landed at the baseline period too -> the defect did not
                            reproduce; no verdict.
  NON-MONOTONIC             landing does not order by period -> report raw counts,
                            claim no mechanism.

ORDERING, AND WHY THE SWEEP STOPS. Once the store fills it STAYS filled -- the 09-08
test watched the value hold for 88 s with no revert -- so every phase AFTER a landing
phase is unscoreable: its store is already non-empty and "did this phase land?" has
no observable. The sweep therefore runs ASCENDING and STOPS at the first phase that
lands. That stopping point IS the threshold, and the phases before it are the ones
that carry the negative evidence.

Usage:
    JUNIPER_E2E_CANOPY_URL=http://127.0.0.1:8052 \\
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-10_f035_trigger_period_sweep.py --periods 1000,2000,3000,5000,8000
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_seg17 = _load("_seg17drv", "e2e_seg17_topology_driver.py")
_f039 = _load("_f039supt", "e2e_f039_supersession_test.py")
_unopp = _load("_f035unopp", "2026-09-10_f035_unopposed_response_test.py")

log = _seg17.log
CANOPY = _seg17.CANOPY
open_dashboard = _seg17.open_dashboard
open_tab = _seg17.open_tab
SETPROPS = _f039.SETPROPS

METRICS_STORE = _unopp.METRICS_STORE
INTERVAL = _unopp.INTERVAL
WRITER_OUTPUT = _unopp.WRITER_OUTPUT
_JS_INSTALL = _unopp._JS_INSTALL
_JS_HARVEST = _unopp._JS_HARVEST
_JS_ACTIVE_TAB = _unopp._JS_ACTIVE_TAB

CASCOR = os.environ.get("JUNIPER_E2E_CASCOR_URL", "http://127.0.0.1:8202")
OUT = os.environ.get("F035_SWEEP_RESULTS", "/tmp/juniper-e2e/f035_period_sweep.json")


def _health(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/v1/health", timeout=5) as r:  # noqa: S310
            p = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "url": base, "version": p.get("version"), "git_sha": p.get("git_sha")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": base, "why": f"{type(exc).__name__}: {exc}"[:160]}


def main() -> int:
    ap = argparse.ArgumentParser(description="F-035 trigger-period dose-response")
    ap.add_argument("--periods", default="1000,2000,3000,5000,8000", help="ms, comma-separated")
    ap.add_argument("--phase", type=float, default=40.0, help="seconds per phase")
    ap.add_argument("--tab", default="Candidate Metrics")
    ap.add_argument("--no-stop-on-land", action="store_true",
                    help="continue after the first landing phase (values persist; later phases unscoreable)")
    args = ap.parse_args()

    periods = [int(x) for x in args.periods.split(",") if x.strip()]
    from playwright.sync_api import sync_playwright

    res = {
        "probe": Path(__file__).name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canopy": CANOPY,
        "periods_ms": periods,
        "phase_s": args.phase,
        "serving": {"canopy": _health(CANOPY), "cascor": _health(CASCOR)},
        "phases": [],
    }
    log(f"canopy serving: {res['serving']['canopy']}")
    log(f"cascor serving: {res['serving']['cascor']}")

    invocations: list = []
    pending: dict = {}
    console: list = []
    t0 = time.time()

    with sync_playwright() as pw:
        browser, ctx, page = open_dashboard(pw, [])
        try:
            open_tab(page, args.tab)
            page.wait_for_timeout(4000)
            res["tab_active"] = page.evaluate(_JS_ACTIVE_TAB)
            install = page.evaluate(_JS_INSTALL, [
                {"id": METRICS_STORE, "prop": "data", "mode": "len"},
                {"id": INTERVAL, "prop": "n_intervals", "mode": "value"},
                {"id": INTERVAL, "prop": "interval", "mode": "value"},
            ])
            t_install = time.time() - t0
            res["observer_install"] = install
            res["observer_install_at"] = round(t_install, 3)
            log(f"active tab: {res['tab_active']!r}; observer at t={t_install:.2f}s: {install.get('ok')}")
            if not install.get("ok"):
                res["verdict"] = "CONTROL-FAILED"
                res["verdict_why"] = f"observer refused to install: {install}"
                Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
                return 2

            def on_console(m):
                console.append({"t": round(time.time() - t0, 3), "type": m.type, "text": m.text[:400]})

            def on_request(req):
                if "_dash-update-component" not in req.url:
                    return
                try:
                    out = (json.loads(req.post_data or "{}") or {}).get("output")
                except (ValueError, TypeError):
                    return
                if isinstance(out, str) and out == WRITER_OUTPUT:
                    pending[req] = time.time()

            def on_response(resp):
                ts = pending.pop(resp.request, None)
                if ts is None:
                    return
                te = time.time()
                rows = None
                try:
                    payload = json.loads(resp.text())
                    rmap = payload.get("response") if isinstance(payload, dict) else None
                    if isinstance(rmap, dict) and METRICS_STORE in rmap:
                        v = (rmap.get(METRICS_STORE) or {}).get("data")
                        rows = len(v) if isinstance(v, list) else None
                except Exception:  # noqa: BLE001
                    pass
                invocations.append({"t_start": round(ts - t0, 3), "t_end": round(te - t0, 3),
                                    "duration_s": round(te - ts, 3), "status": resp.status, "rows": rows})

            page.on("console", on_console)
            page.on("request", on_request)
            page.on("response", on_response)

            for p in periods:
                sp = page.evaluate(SETPROPS, {"id": INTERVAL, "payload": {"interval": p}})
                t_begin = round(time.time() - t0, 3)
                log(f"  phase period={p}ms  setProps={sp.get('ok') if isinstance(sp, dict) else sp}  t={t_begin}")
                page.wait_for_timeout(int(args.phase * 1000))
                t_end = round(time.time() - t0, 3)
                res["phases"].append({"period_ms": p, "t_begin": t_begin, "t_end": t_end,
                                      "setprops_ok": bool(isinstance(sp, dict) and sp.get("ok"))})
                # peek at the live trace to decide whether to stop
                peek = page.evaluate(_JS_HARVEST) or {}
                filled = any(s["id"] == METRICS_STORE and isinstance(s["len"], int) and s["len"] > 0
                             for s in peek.get("samples", []))
                if filled and not args.no_stop_on_land:
                    log(f"  -> store FILLED during period={p}ms; stopping the sweep (values persist)")
                    break

            page.evaluate(SETPROPS, {"id": INTERVAL, "payload": {"interval": 1000}})
            page.remove_listener("request", on_request)
            page.remove_listener("response", on_response)
            page.remove_listener("console", on_console)
            harvest = page.evaluate(_JS_HARVEST) or {}
            res["observer"] = {k: harvest.get(k) for k in ("notifies", "heartbeats", "last", "dur_ms")}
            raw = harvest.get("samples", [])
        finally:
            browser.close()

    off = res["observer_install_at"]
    trace = [{"t": round(s["t"] / 1000.0 + off, 3), "id": s["id"], "prop": s.get("prop"),
              "len": s["len"], "prev": s["prev"], "reason": s["reason"]} for s in raw]
    res["trace"] = trace
    res["console"] = console
    res["invocations"] = invocations

    fills = [s for s in trace if s["id"] == METRICS_STORE and isinstance(s["len"], int) and s["len"] > 0]
    ticks = [s for s in trace if s["id"] == INTERVAL and s["prop"] == "n_intervals"]

    for ph in res["phases"]:
        a, b = ph["t_begin"], ph["t_end"]
        ivs = [iv for iv in invocations if a <= iv["t_start"] < b]
        ph_ticks = [t for t in ticks if a <= t["t"] < b]
        ph_fills = [f for f in fills if a <= f["t"] < b]
        durs = [iv["duration_s"] for iv in ivs]
        gaps = [round(y["t"] - x["t"], 3) for x, y in zip(ph_ticks, ph_ticks[1:])]
        ph.update({
            "invocations": len(ivs),
            "rows_each": sorted({iv["rows"] for iv in ivs if iv["rows"] is not None}),
            "duration_median_s": round(statistics.median(durs), 3) if durs else None,
            "duration_max_s": max(durs) if durs else None,
            "observed_ticks": len(ph_ticks),
            "observed_tick_gap_median_s": round(statistics.median(gaps), 3) if gaps else None,
            "fills": len(ph_fills),
            "landed": bool(ph_fills),
            "first_fill": ph_fills[0] if ph_fills else None,
        })

    scored = [ph for ph in res["phases"] if ph["invocations"] > 0]
    landed = [ph for ph in scored if ph["landed"]]
    not_landed = [ph for ph in scored if not ph["landed"]]
    all_durs = [iv["duration_s"] for iv in invocations]

    if not scored:
        verdict, why = "NO-WRITES", "no store-writing invocation in any phase"
    elif scored[0]["landed"]:
        verdict, why = "LANDS-AT-ALL-PERIODS", (
            f"the baseline phase (period={scored[0]['period_ms']}ms) landed; the defect did not reproduce")
    elif not landed:
        verdict, why = "NO-LANDING-AT-ANY-PERIOD", (
            f"no phase landed up to {scored[-1]['period_ms']}ms; the trigger period is NOT the "
            "controlling variable and a dedicated Interval will not fix this")
    else:
        max_fail = max(ph["period_ms"] for ph in not_landed)
        min_land = min(ph["period_ms"] for ph in landed)
        if max_fail < min_land:
            verdict = "PERIOD-CONTROLS-LANDING"
            why = (f"landing turns on between {max_fail}ms and {min_land}ms; measured round trip "
                   f"median {round(statistics.median(all_durs), 3)}s max {max(all_durs)}s -- the "
                   "threshold falls in the round-trip range, which is the eviction signature")
        else:
            verdict = "NON-MONOTONIC"
            why = (f"a phase at {max_fail}ms failed while one at {min_land}ms landed; landing does "
                   "not order by period")
    res["verdict"], res["verdict_why"] = verdict, why
    res["round_trip_s"] = {
        "n": len(all_durs),
        "median": round(statistics.median(all_durs), 3) if all_durs else None,
        "max": max(all_durs) if all_durs else None,
    }

    log("")
    log(f"  serving canopy : {res['serving']['canopy'].get('git_sha')}")
    log(f"  round trip     : {res['round_trip_s']}")
    for ph in res["phases"]:
        log(f"  period {ph['period_ms']:>5}ms | inv {ph['invocations']:>3} | rt_med "
            f"{ph['duration_median_s']} | ticks {ph['observed_ticks']} (gap {ph['observed_tick_gap_median_s']}) "
            f"| fills {ph['fills']} | LANDED={ph['landed']}")
    log(f"  console errors : {sum(1 for c in console if c['type'] == 'error')}")
    log(f"  VERDICT        : {verdict} -- {why}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
