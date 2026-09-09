#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""Dump what the browser's ``network-visualizer-topology-store`` actually holds.

Two instruments disagreed on 2026-09-08 about a store whose graph paints 48 hidden
units: the dispatch probe saw three ``Callbacks.Aggregate`` actions carrying a dict for
it, and the lifecycle probe's independent read at the end of its window said
``hidden_units: 0``. Both read through ``paths.strs``. This dumps the value's SHAPE
(keys, the ``hidden_units`` field, the length of ``nodes`` / ``connections``, the node
type census) beside the graph's own node count and canopy's ``/api/topology`` count,
so "the store holds the mount default" and "the store holds a payload whose count
field is 0" stop being the same sentence.

Usage:
    JUNIPER_E2E_CANOPY_URL=http://127.0.0.1:8052 \\
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-08_topology_store_dump.py --settle 30
"""

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
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
http_get = _w3.http_get
serving_commit = _w3.serving_commit
open_dashboard = _seg17.open_dashboard
_store = _seg17._store
counts = _seg17.counts
open_tab = _f027.open_tab
ensure_no_modal = _f027.ensure_no_modal

STORE = "network-visualizer-topology-store"
GRAPH = "network-visualizer-graph"
OUT = os.environ.get("TOPO_DUMP_RESULTS", "/tmp/juniper-e2e/topology_store_dump.json")


def _shape(v) -> dict:
    if not isinstance(v, dict):
        return {"type": type(v).__name__, "value": v if v is None or isinstance(v, (int, float, str)) else str(v)[:80]}
    out = {"type": "dict", "keys": sorted(v.keys()), "hidden_units_field": v.get("hidden_units")}
    nodes = v.get("nodes")
    if isinstance(nodes, list):
        out["n_nodes"] = len(nodes)
        out["node_types"] = dict(Counter(n.get("type") for n in nodes if isinstance(n, dict)))
    conns = v.get("connections")
    if isinstance(conns, list):
        out["n_connections"] = len(conns)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settle", type=float, default=30.0, help="seconds on the topology tab before the final read")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    res: dict = {"serving": serving_commit(), "reads": []}
    try:
        api = http_get("/api/topology", timeout=30)[1]
        res["api_topology"] = _shape(api.get("data") if isinstance(api, dict) and "data" in api else api)
    except Exception as exc:  # noqa: BLE001
        res["api_topology"] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
    log(f"  /api/topology: {json.dumps(res['api_topology'])[:300]}")

    with sync_playwright() as pw:
        browser, _ctx, page = open_dashboard(pw, [])
        try:
            ensure_no_modal(page)
            # Read BEFORE the tab is opened: the mount value.
            rd = _store(page, STORE) or {}
            res["reads"].append({"when": "before topology tab", "via": rd.get("via"), "shape": _shape(rd.get("value"))})
            log(f"  store before tab: {json.dumps(res['reads'][-1])[:300]}")
            open_tab(page, "Network Topology")
            for t in (5, 15, int(args.settle)):
                page.wait_for_timeout(int((t - (res['reads'][-1].get('t') or 0)) * 1000) if len(res["reads"]) > 1 else t * 1000)
                rd = _store(page, STORE) or {}
                graph_nodes = page.evaluate(
                    """(id) => { const root = document.getElementById(id); if (!root) return null;
                         const gd = root.classList.contains('js-plotly-plot') ? root : root.querySelector('.js-plotly-plot');
                         if (!gd || !gd.data) return null;
                         let n = 0; for (const tr of gd.data) { if ((tr.name||'').toLowerCase().indexOf('hidden') !== -1 && tr.x) n += tr.x.length; }
                         return {traces: gd.data.length, hidden_points: n}; }""",
                    GRAPH,
                )
                rec = {"when": f"topology tab +{t}s", "t": t, "via": rd.get("via"), "shape": _shape(rd.get("value")), "stats_bar": counts(page), "graph": graph_nodes}
                res["reads"].append(rec)
                log(f"  {rec['when']}: store={json.dumps(rec['shape'])[:200]} stats_bar={rec['stats_bar']} graph={graph_nodes}")
        finally:
            browser.close()

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
