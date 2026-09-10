#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""F-CANOPY-035 -- the discriminating test: remove the competing cadence, trigger ONCE.

WHERE THIS PICKS UP. Over 2026-09-05/07 the arc narrowed F-CANOPY-035 to: the
callback that writes ``metrics-panel-metrics-store`` (``update_metrics_store``,
``dashboard_manager.py:4104``) is scheduled ~24 times per 90 s and its lifecycle
never completes -- it reaches a terminal ``state.callbacks`` list zero times while
the store's value never advances (``2026-09-07_f035_callback_lifecycle_probe.py``).
WHY it never completes was left open, with two live candidates:

  SUPERSESSION   each in-flight call is retired by the next 1 Hz tick before its
                 1.8 s round trip lands (``FAST_UPDATE_INTERVAL_MS = 1000`` vs a
                 measured 1.827 s median). Fix: suppress the TRIGGER.
  FAILED RESPONSE
                 the renderer treats the response as failed/aborted
                 (``Callback failed: the server did not respond`` appeared in 2 of
                 the 4 failing runs and none of the successful one). Fix: something
                 else entirely -- timeout, transport, payload.

A Lane B review found four numbers that fit supersession badly (a 0.11 s measured
margin, 9 of 29 calls unopposed and still lost, a scheduling rate 3.5x SLOWER than
the trigger, and two concurrent entries never once observed), so neither may be
assumed. This is the test that separates them, on the model of
``e2e_f039_supersession_test.py`` (2026-09-02): if supersession is the mechanism,
then removing the competing cadence and triggering exactly once must let the
store fill; if the store stays empty with nothing to supersede it, supersession
is falsified cheaply, and the failed-response candidate takes the lead.

METHOD, no code change and no restart:

  1. open the Candidate Metrics tab, read the store (expect ``len 0``) and count
     what the wire does for ``--baseline`` seconds -- the contended regime;
  2. ``setProps({disabled: true})`` on ``fast-update-interval`` through the
     component's own React fiber (the §12.1 idiom). That silences the writer's
     only periodic trigger AND every other fast-lane callback, so nothing is left
     to supersede anything;
  3. let in-flight work drain for ``--settle`` seconds, reading the store again --
     a fill HERE means the last in-flight call landed once nothing displaced it,
     which is already the supersession signature;
  4. trigger the writer exactly ONCE by ``setProps({n_intervals: n+1})`` on the
     same Interval -- the very prop change a tick produces, so the callback runs
     with its real Input and its real handler, with no second tick behind it;
  5. watch the store for ``--watch`` seconds, recording every
     ``/_dash-update-component`` response that names the store (with row count),
     every console error, and the store length every second.

READING RULE, fixed before the run:

  APPLIED-UNCONTENDED     the store advanced after (3) or (4) -> the write path
                          works when nothing supersedes it. SUPERSESSION is the
                          mechanism; the fix belongs at the trigger.
  EMPTY-DESPITE-RESPONSE  exactly one response carrying rows was observed and the
                          store still reads 0 -> the discard is independent of
                          contention. Supersession is FALSIFIED; look at how the
                          renderer treats that response.
  NO-RESPONSE             the single trigger produced no response naming the store
                          (or a console ``Callback failed``) -> the failed-response
                          candidate, directly observed.
  NOT-MEASURED            a setProps target was unreachable (exit 2). Says nothing.

The baseline in (1) is what makes (5) a comparison rather than an anecdote: the
same page, the same leg, the same minute, with and without the cadence.

Usage:
    JUNIPER_E2E_CANOPY_URL=http://127.0.0.1:8052 \\
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-08_f035_supersession_test.py --baseline 30 --settle 20 --watch 60

Exit codes: 0 the test ran (read the verdict), 2 a setProps target was unreachable.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_w3 = _load("_w3drv", "e2e_w3_params_driver.py")
_f027 = _load("_f027drv", "e2e_f027_redrive.py")
_seg17 = _load("_seg17drv", "e2e_seg17_topology_driver.py")
_f039 = _load("_f039sup", "e2e_f039_supersession_test.py")

log = _w3.log
CANOPY = _w3.CANOPY
serving_commit = _w3.serving_commit
open_dashboard = _seg17.open_dashboard
_store = _seg17._store
open_tab = _f027.open_tab
ensure_no_modal = _f027.ensure_no_modal
fig_info = _f027.fig_info
SETPROPS = _f039.SETPROPS  # the same fiber walk the F-039 test used; one idiom, not two

STORE = "metrics-panel-metrics-store"
INTERVAL = "fast-update-interval"
LOSS_FIG = "candidate-metrics-panel-loss-plot"
OUT = os.environ.get("F035_SUPERSESSION_RESULTS", "/tmp/juniper-e2e/f035_supersession.json")


def _prop(page, comp_id: str, prop: str):
    """Read one prop of a component off dash-renderer's layout via ``paths.strs``."""
    return page.evaluate(
        """(a) => { const [id, prop] = a;
             const st = window.store && window.store.getState ? window.store.getState() : null;
             if (!st || !st.layout) return {ok:false, via:'<no redux>'};
             const strs = st.paths && st.paths.strs ? st.paths.strs : null;
             if (!strs || !strs[id]) return {ok:false, via:'<no path>'};
             let node = st.layout;
             for (const key of strs[id]) { if (node == null) break; node = node[key]; }
             if (!node || !node.props) return {ok:false, via:'<no props>'};
             return {ok:true, value: node.props[prop] === undefined ? null : node.props[prop], via:'paths.strs'}; }""",
        [comp_id, prop],
    )


def _store_len(page):
    rd = _store(page, STORE) or {}
    v = rd.get("value")
    return len(v) if isinstance(v, list) else (None if v is None else f"<{type(v).__name__}>")


class _Wire:
    """Responses naming the store, with the row count each carried (or ``omitted``)."""

    def __init__(self, page):
        self.page = page
        self.events: list = []
        self.console: list = []
        self._t0 = time.time()
        page.on("response", self._on_response)
        page.on("console", self._on_console)

    def _on_console(self, m):
        if m.type in ("error", "warning") and ("Callback" in m.text or "callback" in m.text or "server" in m.text):
            self.console.append({"t": round(time.time() - self._t0, 2), "type": m.type, "text": m.text[:240]})

    def _on_response(self, resp):
        if "_dash-update-component" not in resp.url:
            return
        t = round(time.time() - self._t0, 2)
        try:
            body = resp.text()
        except Exception:  # noqa: BLE001
            self.events.append({"t": t, "status": resp.status, "kind": "unreadable"})
            return
        if STORE not in body:
            return
        try:
            payload = json.loads(body)
        except ValueError:
            self.events.append({"t": t, "status": resp.status, "kind": "unparsed"})
            return
        rm = payload.get("response") if isinstance(payload, dict) else None
        if isinstance(rm, dict) and STORE in rm:
            val = (rm.get(STORE) or {}).get("data")
            self.events.append({"t": t, "status": resp.status, "kind": "wrote", "rows": len(val) if isinstance(val, list) else f"<{type(val).__name__}>"})
        else:
            self.events.append({"t": t, "status": resp.status, "kind": "omitted"})

    def window(self, since: float) -> list:
        return [e for e in self.events if e["t"] >= since]

    def now(self) -> float:
        return round(time.time() - self._t0, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=float, default=30.0, help="seconds to observe the contended regime first")
    ap.add_argument("--settle", type=float, default=20.0, help="seconds to let in-flight work drain after disabling the tick")
    ap.add_argument("--watch", type=float, default=60.0, help="seconds to watch after the single trigger")
    ap.add_argument("--tab", default="Candidate Metrics")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    res: dict = {"canopy": CANOPY, "serving": serving_commit(), "store": STORE, "interval": INTERVAL, "args": vars(args), "phases": {}}
    log(f"serving: {json.dumps(res['serving'])}")

    with sync_playwright() as pw:
        browser, _ctx, page = open_dashboard(pw, [])
        try:
            ensure_no_modal(page)
            open_tab(page, args.tab)
            page.wait_for_timeout(4000)
            wire = _Wire(page)

            # (1) baseline: the contended regime
            t_b = wire.now()
            len0 = _store_len(page)
            seq = [(0.0, len0)]
            for _i in range(int(args.baseline)):
                page.wait_for_timeout(1000)
                L = _store_len(page)
                if L != seq[-1][1]:
                    seq.append((wire.now() - t_b, L))
            base_events = wire.window(t_b)
            res["phases"]["baseline"] = {
                "seconds": args.baseline, "store_before": len0, "store_after": _store_len(page), "store_seq": seq,
                "responses_naming_store": len(base_events), "wrote": [e["rows"] for e in base_events if e["kind"] == "wrote"],
                "omitted": sum(1 for e in base_events if e["kind"] == "omitted"), "console": list(wire.console),
                "n_intervals": (_prop(page, INTERVAL, "n_intervals") or {}).get("value"),
                "loss_fig_traces": len((fig_info(page, LOSS_FIG) or {}).get("traces") or []),
            }
            b = res["phases"]["baseline"]
            log(f"  BASELINE {args.baseline}s: store {b['store_before']} -> {b['store_after']}; responses naming store={b['responses_naming_store']} wrote={b['wrote'][:12]} omitted={b['omitted']} n_intervals={b['n_intervals']}")

            # (2) disable the competing cadence -- the writer's own trigger included
            r = page.evaluate(SETPROPS, {"id": INTERVAL, "payload": {"disabled": True}})
            log(f"  disable {INTERVAL}: {json.dumps(r)}")
            res["phases"]["disable"] = r
            if not r.get("ok"):
                log("  !! could not disable the tick -- this test measured NOTHING")
                res["result"] = {"verdict": "NOT-MEASURED", "why": "setProps on the interval was unreachable"}
                Path(OUT).parent.mkdir(parents=True, exist_ok=True)
                Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
                return 2

            # (3) settle
            t_s = wire.now()
            n_before = (_prop(page, INTERVAL, "n_intervals") or {}).get("value")
            sseq = [(0.0, _store_len(page))]
            for _ in range(int(args.settle)):
                page.wait_for_timeout(1000)
                L = _store_len(page)
                if L != sseq[-1][1]:
                    sseq.append((round(wire.now() - t_s, 1), L))
            n_after_settle = (_prop(page, INTERVAL, "n_intervals") or {}).get("value")
            settle_events = wire.window(t_s)
            res["phases"]["settle"] = {
                "seconds": args.settle, "store_seq": sseq, "n_intervals_start": n_before, "n_intervals_end": n_after_settle,
                "tick_stopped": n_before == n_after_settle,
                "responses_naming_store": len(settle_events), "wrote": [e["rows"] for e in settle_events if e["kind"] == "wrote"],
            }
            s = res["phases"]["settle"]
            log(f"  SETTLE {args.settle}s: tick stopped={s['tick_stopped']} (n_intervals {n_before} -> {n_after_settle}); store seq {sseq}; responses naming store={s['responses_naming_store']} wrote={s['wrote']}")
            if not s["tick_stopped"]:
                log("  !! the interval kept ticking after disabled=true -- the cadence was NOT removed; treat everything below as contended")

            # (4) one trigger, through the Interval's own prop
            t_t = wire.now()
            target_n = (n_after_settle or 0) + 1
            r2 = page.evaluate(SETPROPS, {"id": INTERVAL, "payload": {"n_intervals": target_n}})
            log(f"  single trigger ({INTERVAL}.n_intervals -> {target_n}): {json.dumps(r2)}")
            res["phases"]["trigger"] = {"setprops": r2, "n_intervals": target_n}
            if not r2.get("ok"):
                log("  !! could not drive the trigger -- this test measured NOTHING")
                res["result"] = {"verdict": "NOT-MEASURED", "why": "setProps on n_intervals was unreachable"}
                Path(OUT).parent.mkdir(parents=True, exist_ok=True)
                Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
                return 2

            # (5) watch
            wseq = [(0.0, _store_len(page))]
            applied_at = None
            for _i in range(int(args.watch)):
                page.wait_for_timeout(1000)
                L = _store_len(page)
                if L != wseq[-1][1]:
                    wseq.append((round(wire.now() - t_t, 1), L))
                    if isinstance(L, int) and L > 0 and applied_at is None:
                        applied_at = round(wire.now() - t_t, 1)
            watch_events = wire.window(t_t)
            res["phases"]["watch"] = {
                "seconds": args.watch, "store_seq": wseq, "applied_at_s": applied_at,
                "responses_naming_store": len(watch_events), "events": watch_events[:20],
                "wrote": [e["rows"] for e in watch_events if e["kind"] == "wrote"],
                "console_after_trigger": [c for c in wire.console if c["t"] >= t_t],
                "n_intervals_end": (_prop(page, INTERVAL, "n_intervals") or {}).get("value"),
                "loss_fig_traces": len((fig_info(page, LOSS_FIG) or {}).get("traces") or []),
                "independent_read": _store_len(page),
            }
            w = res["phases"]["watch"]
            log(f"  WATCH {args.watch}s: store seq {wseq} applied_at={applied_at}; responses naming store={w['responses_naming_store']} wrote={w['wrote']} console={w['console_after_trigger'][:3]}")

            # courtesy: re-enable the tick before leaving (the page is closed anyway)
            page.evaluate(SETPROPS, {"id": INTERVAL, "payload": {"disabled": False}})
        finally:
            browser.close()

    # --- verdict, from the rule in the docstring ---
    st = res["phases"]["settle"]
    wt = res["phases"]["watch"]
    settled_fill = any(isinstance(L, int) and L > 0 for _, L in st["store_seq"])
    watched_fill = wt["applied_at_s"] is not None
    wrote_rows = [r for r in wt["wrote"] if isinstance(r, int) and r > 0]
    console_fail = [c for c in wt["console_after_trigger"] if "fail" in c["text"].lower()]
    if settled_fill or watched_fill:
        where = "during settle, before any new trigger" if settled_fill else f"{wt['applied_at_s']}s after the single trigger"
        res["result"] = {
            "verdict": "APPLIED-UNCONTENDED",
            "why": (f"the store advanced to a non-empty length {where}, with the fast lane disabled and at most one "
                    "call in flight. The write path works when nothing supersedes it: SUPERSESSION is the mechanism, "
                    "and the fix belongs at the trigger (its own Interval, or clientside gating), not in the handler."),
        }
    elif wrote_rows and not console_fail:
        res["result"] = {
            "verdict": "EMPTY-DESPITE-RESPONSE",
            "why": (f"{len(wrote_rows)} response(s) carrying {wrote_rows} rows landed after the single trigger with nothing "
                    "to supersede them, and the store still reads 0. The discard is independent of contention: "
                    "supersession is FALSIFIED. Look at how dash-renderer treats THAT response."),
        }
    elif not wt["wrote"] or console_fail:
        res["result"] = {
            "verdict": "NO-RESPONSE",
            "why": (f"the single trigger produced {len(wt['wrote'])} response(s) carrying rows and {len(console_fail)} console "
                    "failure(s). The failed-response candidate, observed directly with no competing cadence to blame."),
        }
    else:
        res["result"] = {"verdict": "INDETERMINATE", "why": "read the phases; none of the pre-registered branches matched cleanly"}
    if not st["tick_stopped"]:
        res["result"]["caveat"] = "the interval kept ticking after disabled=true; the uncontended regime was NOT established"

    log("")
    log(f"  => VERDICT: {res['result']['verdict']}")
    log(f"     {res['result']['why']}")
    if res["result"].get("caveat"):
        log(f"     CAVEAT: {res['result']['caveat']}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
