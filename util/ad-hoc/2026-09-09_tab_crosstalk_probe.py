#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""Do two canopy pages in ONE browser context stay on the tabs they selected?

WHY THIS EXISTS. The 2026-09-08 live-run probe watches a growth run from three
pages at once -- Candidate Metrics, Training Metrics, Network Topology -- opened
with ``ctx.new_page()`` on a single Playwright context, and scores
M-CANDIDATES-09/-10/-11, F-CANOPY-026 and M-TOPOLOGY-16 from what each page shows.
Its four end-of-run screenshots came back as only TWO distinct images, and BOTH
show the **Network Topology** tab -- including the files named
``candidates_end`` and ``metrics_end``.

THE CANDIDATE MECHANISM, from source. canopy persists the active tab in
``layout-state-store``, a ``dcc.Store(storage_type="local")``
(``dashboard_manager.py:1884``, the persistence note at ``:3591``), and a
clientside callback takes ``Input("layout-state-store", "data")`` to
``Output("visualization-tabs", "active_tab")`` with an equality guard
(``:3832-3849``). ``storage_type="local"`` is browser localStorage, which is shared
by every page of the same origin in one browser context -- so if dcc.Store
propagates a cross-page ``storage`` event, page B selecting a tab rewrites the
store in page A, whose restore callback then switches A's tab to match. Every
per-tab poll lane in canopy gates on ``active_tab``, so a page that silently
drifts stops driving the lane the probe believes it is driving.

THE TEST. One context, two pages. A selects Candidate Metrics, B then selects
Network Topology, both settle, and each page reports its OWN ``active_tab`` (read
from the tabs component through ``state.paths.strs``, not from a screenshot) plus
the localStorage value both share.

  CROSSTALK      A's active_tab followed B's selection: the three-tab design is
                 unsound and every row scored from a non-frontmost page needs
                 re-driving one tab at a time.
  INDEPENDENT    A stayed on its own tab: the pages are independent and the
                 duplicated screenshots are a capture artefact, not a tab drift.

Usage:
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-09_tab_crosstalk_probe.py [--settle 12]
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

log = _w3.log
CANOPY = _w3.CANOPY
dismiss_welcome = _w3.dismiss_welcome
open_dashboard = _seg17.open_dashboard
open_tab = _f027.open_tab
ensure_no_modal = _f027.ensure_no_modal

TABS = "visualization-tabs"
OUT = os.environ.get("TAB_CROSSTALK_RESULTS", "/tmp/juniper-e2e/tab_crosstalk.json")


def _active_tab(page):
    """The page's OWN active_tab, read out of the renderer's prop tree."""
    return page.evaluate(
        """(tid) => {
             const st = window.store && window.store.getState ? window.store.getState() : null;
             if (!st) return {error: 'no store'};
             const path = st.paths && st.paths.strs ? st.paths.strs[tid] : null;
             if (!path) return {error: 'id not in paths.strs'};
             let node = st.layout;
             for (const key of path) { node = node[key]; }
             let ls = null;
             try { ls = window.localStorage.getItem('layout-state-store'); } catch (e) { ls = 'unreadable'; }
             return {active_tab: node && node.props ? node.props.active_tab : null, localStorage: ls};
           }""",
        TABS,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settle", type=float, default=12.0, help="seconds to wait after B's selection before reading both pages")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    res: dict = {"canopy": CANOPY, "serving": _w3.serving_commit(), "settle_s": args.settle,
                 "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log(f"serving: {json.dumps(res['serving'])}")
    capture: list = []

    with sync_playwright() as pw:
        browser, ctx, page_a = open_dashboard(pw, capture)
        try:
            ok_a = open_tab(page_a, "Candidate Metrics")
            page_a.wait_for_timeout(2500)
            res["A_after_own_selection"] = _active_tab(page_a)
            log(f"A selected Candidate Metrics (ok={ok_a}) -> {json.dumps(res['A_after_own_selection'])}")

            page_b = ctx.new_page()
            page_b.goto(CANOPY, wait_until="domcontentloaded", timeout=60_000)
            page_b.wait_for_timeout(3000)
            dismiss_welcome(page_b)
            page_b.wait_for_timeout(1500)
            ensure_no_modal(page_b)
            ok_b = open_tab(page_b, "Network Topology")
            log(f"B selected Network Topology (ok={ok_b})")

            page_b.wait_for_timeout(int(args.settle * 1000))
            res["A_after_B_selection"] = _active_tab(page_a)
            res["B_after_B_selection"] = _active_tab(page_b)
            log(f"A now -> {json.dumps(res['A_after_B_selection'])}")
            log(f"B now -> {json.dumps(res['B_after_B_selection'])}")
        finally:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass

    a0 = (res.get("A_after_own_selection") or {}).get("active_tab")
    a1 = (res.get("A_after_B_selection") or {}).get("active_tab")
    b1 = (res.get("B_after_B_selection") or {}).get("active_tab")
    if a1 is not None and b1 is not None and a1 == b1 and a1 != a0:
        res["verdict"] = "CROSSTALK"
        res["why"] = (f"page A selected {a0!r}, page B then selected {b1!r}, and A now reports {a1!r}: "
                      "one page's tab selection moves every other page in the same browser context. "
                      "Any row scored from a page that was not the last to select a tab is unsound.")
    elif a1 == a0 and b1 != a1:
        res["verdict"] = "INDEPENDENT"
        res["why"] = f"A held {a1!r} while B moved to {b1!r}: the pages are independent."
    else:
        res["verdict"] = "INDETERMINATE"
        res["why"] = f"A: {a0!r} -> {a1!r}; B: {b1!r}. Read the localStorage values in the artifact."

    log("")
    log(f"  => VERDICT: {res['verdict']}")
    log(f"     {res['why']}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
