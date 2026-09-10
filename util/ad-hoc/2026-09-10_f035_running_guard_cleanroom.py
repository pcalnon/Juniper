#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""F-CANOPY-035 -- reproduce the defect in ~80 lines, and test the proposed fix.

WHY A CLEAN ROOM. The measurements on the live canopy leg establish that the store
never fills while its trigger period is under the callback's round trip, and fills as
soon as the period exceeds it (``2026-09-10_f035_trigger_period_sweep.py``:
PERIOD-CONTROLS-LANDING). The fix that follows from dash_renderer.dev.js:2698/:3027
is to make a re-request DURING flight structurally impossible rather than to tune a
constant -- Dash's ``running=`` argument sets a prop while a callback is in flight and
restores it after, so ``running=[(Output(tick, "disabled"), True, False)]`` stops the
clock for exactly the duration of the call.

``running=`` is documented next to three parameters that say "only applies to
background callbacks"; it does NOT carry that sentence, and dash/_callback.py passes
it to ``insert_callback`` with no ``background`` gate, and the renderer dispatches
``sideUpdate(running.running)`` on the ordinary fetch path (:818) and
``runningOff`` on response (:1038). That is a source reading. This module is the
EXPERIMENT -- the repo's own technique from ``e2e_f027_cleanroom.py``, which settled
the 12-slot cap the same way.

Same dash, same env, no canopy: an Interval, a deliberately slow callback, and a
store. Two arms, one variable.

  --mode plain     the F-035 shape: Interval(period) drives a callback that takes
                   ``delay`` > period. PREDICTION: the store never fills.
  --mode running   identical, plus running=[(Output(tick,'disabled'), True, False)].
                   PREDICTION: the store fills on the first completed call.

VERDICT RULE -- FIXED BEFORE THE FIRST RUN.

  REPRODUCED-AND-FIXED   plain never filled AND running filled -> the defect is the
                         renderer's, reproduced outside canopy, and ``running=`` is a
                         sufficient fix for it.
  NOT-REPRODUCED         plain filled -> the clean room does not carry the defect;
                         it says nothing about canopy and the fix is untested.
  FIX-INEFFECTIVE        plain never filled AND running never filled -> ``running=``
                         does not apply to ordinary callbacks on this dash; the fix
                         direction is wrong and a dedicated period is the fallback.
  BOTH-FILLED            both filled -> `delay` was not above `period`; re-run with a
                         larger --delay.

Usage:
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-10_f035_running_guard_cleanroom.py --both
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

OUT = os.environ.get("F035_CLEANROOM_RESULTS", "/tmp/juniper-e2e/f035_cleanroom.json")
ROWS = 66  # the fixture's own metrics-row count, so the payload shape matches


def _serve(mode: str, port: int, period_ms: int, delay_s: float) -> None:
    """Run the app. Child process -- never returns."""
    import dash
    from dash import Input, Output, State, dcc, html

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Interval(id="tick", interval=period_ms, n_intervals=0),
            dcc.Store(id="store", data=[]),
            html.Div(id="readout", children="len=0"),
        ]
    )

    kwargs = {}
    if mode == "running":
        kwargs["running"] = [(Output("tick", "disabled"), True, False)]

    @app.callback(
        Output("store", "data"),
        Input("tick", "n_intervals"),
        State("store", "data"),
        prevent_initial_call=False,
        **kwargs,
    )
    def fill(n, current):  # noqa: ARG001
        time.sleep(delay_s)  # the round trip canopy's handler really takes
        return [{"epoch": i, "metrics": {"loss": 1.0 / (i + 1)}} for i in range(ROWS)]

    @app.callback(Output("readout", "children"), Input("store", "data"))
    def show(data):
        return f"len={len(data) if isinstance(data, list) else 'NA'}"

    app.run(host="127.0.0.1", port=port, debug=False)


def _drive(port: int, watch_s: float) -> dict:
    """Watch the store's length for `watch_s`, reading through paths.strs."""
    from playwright.sync_api import sync_playwright

    js = """
    () => {
      const s = (window.store && window.store.getState) ? window.store.getState() : null;
      if (!s || !s.layout) return null;
      const strs = (s.paths && s.paths.strs) ? s.paths.strs : null;
      if (!strs || !strs['store']) return null;
      let node = s.layout;
      for (const k of strs['store']) { if (node == null) break; node = node[k]; }
      const v = (node && node.props) ? node.props.data : undefined;
      return Array.isArray(v) ? v.length : -1;
    }
    """
    samples = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        page = b.new_context().new_page()
        # NOT networkidle: a 1 Hz poller means the network is never idle, so the wait
        # would time out on the very app shape this probe exists to build.
        page.goto(f"http://127.0.0.1:{port}/", wait_until="load", timeout=30000)
        page.wait_for_selector("#readout", timeout=15000)
        t0 = time.time()
        while time.time() - t0 < watch_s:
            try:
                samples.append({"t": round(time.time() - t0, 2), "len": page.evaluate(js)})
            except Exception as exc:  # noqa: BLE001
                samples.append({"t": round(time.time() - t0, 2), "len": f"<err {exc}>"[:80]})
            page.wait_for_timeout(500)
        readout = page.text_content("#readout")
        b.close()
    lens = [s["len"] for s in samples if isinstance(s["len"], int) and s["len"] > 0]
    return {"samples": samples[:80], "n_samples": len(samples), "readout": readout,
            "max_len": max(lens) if lens else 0, "filled": bool(lens)}


def _arm(mode: str, port: int, period_ms: int, delay_s: float, watch_s: float) -> dict:
    proc = mp.Process(target=_serve, args=(mode, port, period_ms, delay_s), daemon=True)
    proc.start()
    time.sleep(4.0)  # let the server bind
    try:
        res = _drive(port, watch_s)
    finally:
        proc.terminate()
        proc.join(timeout=5)
    res.update({"mode": mode, "port": port, "period_ms": period_ms, "delay_s": delay_s})
    print(f"  [{mode}] filled={res['filled']} max_len={res['max_len']} readout={res['readout']!r}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", type=int, default=1000, help="Interval period ms")
    ap.add_argument("--delay", type=float, default=1.7, help="callback duration s (canopy's measured round trip)")
    ap.add_argument("--watch", type=float, default=30.0, help="seconds to watch each arm")
    ap.add_argument("--port", type=int, default=8061)
    ap.add_argument("--both", action="store_true", help="run both arms (default)")
    ap.add_argument("--mode", choices=["plain", "running"], help="run one arm only")
    args = ap.parse_args()

    res = {"probe": Path(__file__).name, "period_ms": args.period, "delay_s": args.delay,
           "watch_s": args.watch, "arms": {}}
    import dash
    res["dash_version"] = dash.__version__
    print(f"dash {dash.__version__}; period={args.period}ms delay={args.delay}s "
          f"(delay {'>' if args.delay > args.period / 1000 else '<='} period)")

    modes = [args.mode] if args.mode else ["plain", "running"]
    for i, m in enumerate(modes):
        res["arms"][m] = _arm(m, args.port + i, args.period, args.delay, args.watch)

    plain = res["arms"].get("plain")
    running = res["arms"].get("running")
    if plain and running:
        if plain["filled"] and running["filled"]:
            v, why = "BOTH-FILLED", "delay was not effectively above period; re-run with a larger --delay"
        elif plain["filled"]:
            v, why = "NOT-REPRODUCED", "the plain arm filled; the clean room does not carry the defect"
        elif running["filled"]:
            v, why = ("REPRODUCED-AND-FIXED",
                      f"plain never filled (max_len {plain['max_len']}) and running reached "
                      f"{running['max_len']} rows; running= is sufficient on dash {res['dash_version']}")
        else:
            v, why = ("FIX-INEFFECTIVE",
                      "neither arm filled; running= does not apply to ordinary callbacks here")
    else:
        v, why = "SINGLE-ARM", "only one arm was run; no comparison"
    res["verdict"], res["verdict_why"] = v, why
    print(f"\n  VERDICT: {v} -- {why}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
