#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""The §3.1 replay block, M-METRICS-11..16 and -18, re-driven on a COMPLETED fixture.

WHY. Segment 15 (2026-08-20) scored the seven replay-transport rows BLOCKED: the
controls revealed at COMPLETED but ``metrics-panel-replay-position`` stayed ``0 / 0``,
so ``max_index = 0`` clamped every index transition and none had an observable.
M-METRICS-13 -- the play toggle -- was named the discriminator because its
observable is DATA-INDEPENDENT (the icon flips to ⏸ and ``replay-interval`` is
enabled whatever the store holds), and it too failed then, with zero wire output
across 196 responses. Whether that was a third face of F-CANOPY-027 (the pre-Stage-2
poller starvation, since fixed) or its own defect was left open.

WHAT THE SOURCE SAYS, read 2026-09-08 (``metrics_panel.py:982-1099``): all three
replay callbacks -- ``handle_replay_controls``, ``replay_tick``, ``update_replay_ui``
-- compute ``max_index = len(metrics_data) - 1 if metrics_data else 0`` from
``State(metrics-panel-metrics-store)``. So the index rows are downstream of the
F-CANOPY-035 store (empty -> every transition clamps to 0), while the play toggle
(``mode`` flip -> ``update_play_button`` -> ⏸, and ``replay-interval.disabled``)
and the speed buttons (``interval = 1000/speed``) are not.

WHAT THIS DRIVES, on the Training Metrics tab with training COMPLETED:

  M-METRICS-13  click ▶: play button text ▶ -> ⏸, ``replay-interval.disabled`` -> false,
                ``replay-state.mode`` -> playing; click again -> back. Wire: a
                ``/_dash-update-component`` response naming ``replay-state``.
  M-METRICS-16  click 2x / 4x / 1x: ``replay-interval.interval`` -> 500 / 250 / 1000.
  M-METRICS-11/-12/-14/-15/-18
                ⏮ / ◀ / step-forward / ⏭ / slider: ``current_index`` and the position
                text. With ``max_index`` 0 these are expected to CLAMP; the row is
                then scored BLOCKED with the blocker named (F-035), not FAIL -- the
                callback ran and did what its code says for an empty store.

Every prop is read off dash-renderer's layout through ``paths.strs`` (the arc's
store reader), never guessed from the DOM.

Usage:
    JUNIPER_E2E_CANOPY_URL=http://127.0.0.1:8052 \\
    LIBTORCH= LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-08_replay_block_redrive.py
"""

import argparse
import importlib.util
import json
import os
import sys
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
serving_commit = _w3.serving_commit
open_dashboard = _seg17.open_dashboard
_store = _seg17._store
open_tab = _f027.open_tab
ensure_no_modal = _f027.ensure_no_modal
vis = _f027.vis
shot = _f027.shot

MP = "metrics-panel"
OUT = os.environ.get("REPLAY_RESULTS", "/tmp/juniper-e2e/replay_block.json")


def _prop(page, comp_id: str, prop: str):
    return page.evaluate(
        """(a) => { const [id, prop] = a;
             const st = window.store && window.store.getState ? window.store.getState() : null;
             if (!st || !st.layout) return null;
             const strs = st.paths && st.paths.strs ? st.paths.strs : null;
             if (!strs || !strs[id]) return null;
             let node = st.layout;
             for (const key of strs[id]) { if (node == null) break; node = node[key]; }
             if (!node || !node.props) return null;
             return node.props[prop] === undefined ? null : node.props[prop]; }""",
        [comp_id, prop],
    )


def _snap(page) -> dict:
    ms = (_store(page, f"{MP}-metrics-store") or {}).get("value")
    rs = (_store(page, f"{MP}-replay-state") or {}).get("value")
    return {
        "metrics_store_len": len(ms) if isinstance(ms, list) else None,
        "replay_state": rs if isinstance(rs, dict) else rs,
        "position": (vis(page, f"{MP}-replay-position") or {}).get("text"),
        "play_text": (vis(page, f"{MP}-replay-play") or {}).get("text"),
        "interval_disabled": _prop(page, f"{MP}-replay-interval", "disabled"),
        "interval_ms": _prop(page, f"{MP}-replay-interval", "interval"),
        "slider_value": _prop(page, f"{MP}-replay-slider", "value"),
        "controls": {k: v for k, v in (vis(page, f"{MP}-replay-controls") or {}).items() if k in ("present", "display", "h")},
    }


def _click(page, comp_id: str) -> bool:
    return page.evaluate(
        """(id) => { const el = document.getElementById(id); if (!el) return false; el.click(); return true; }""",
        comp_id,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to wait after each click")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    capture: list = []
    res: dict = {"canopy": CANOPY, "serving": serving_commit(), "steps": []}
    log(f"serving: {json.dumps(res['serving'])}")

    def step(page, label: str, comp: str | None, expect: str):
        before = _snap(page)
        n0 = len(capture)
        clicked = _click(page, f"{MP}-{comp}") if comp else None
        page.wait_for_timeout(int(args.settle * 1000))
        after = _snap(page)
        reqs = [c for c in capture[n0:] if "_dash-update-component" in (c.get("url") or "")]
        naming = [c for c in reqs if "replay" in (c.get("body") or "")]
        rec = {"label": label, "component": comp, "clicked": clicked, "expect": expect, "before": before, "after": after, "requests": len(reqs), "requests_naming_replay": len(naming)}
        res["steps"].append(rec)
        log(f"  {label}: clicked={clicked} play={before['play_text']!r}->{after['play_text']!r} pos={before['position']!r}->{after['position']!r} "
            f"disabled={before['interval_disabled']}->{after['interval_disabled']} interval={before['interval_ms']}->{after['interval_ms']} "
            f"mode={(before['replay_state'] or {}).get('mode') if isinstance(before['replay_state'], dict) else before['replay_state']}->"
            f"{(after['replay_state'] or {}).get('mode') if isinstance(after['replay_state'], dict) else after['replay_state']} "
            f"idx={(after['replay_state'] or {}).get('current_index') if isinstance(after['replay_state'], dict) else None} reqs={len(reqs)}/{len(naming)}")
        return rec

    with sync_playwright() as pw:
        browser, _ctx, page = open_dashboard(pw, capture)
        try:
            ensure_no_modal(page)
            open_tab(page, "Training Metrics")
            page.wait_for_timeout(6000)
            res["initial"] = _snap(page)
            log(f"  initial: {json.dumps(res['initial'])[:400]}")
            shot(page, "REPLAY__initial.png")

            s13a = step(page, "M-METRICS-13 play", "replay-play", "play text -> ⏸, interval enabled, mode playing")
            s13b = step(page, "M-METRICS-13 pause", "replay-play", "play text -> ▶, interval disabled, mode paused")
            s16_2 = step(page, "M-METRICS-16 2x", "speed-2x", "interval -> 500")
            s16_4 = step(page, "M-METRICS-16 4x", "speed-4x", "interval -> 250")
            s16_1 = step(page, "M-METRICS-16 1x", "speed-1x", "interval -> 1000")
            s14 = step(page, "M-METRICS-14 step-forward", "replay-step-forward", "index +1 capped at max_index")
            s12 = step(page, "M-METRICS-12 step-back", "replay-step-back", "index -1 floored at 0")
            s15 = step(page, "M-METRICS-15 end", "replay-end", "index -> end_index")
            s11 = step(page, "M-METRICS-11 start", "replay-start", "index -> start_index (0)")
            # M-METRICS-18: the slider. Drive its value through the component (Radix-free:
            # dcc.Slider) via setProps on the rendered handle is unreliable; use the
            # keyboard on the focused handle, which plotly-dash's rc-slider honours.
            before18 = _snap(page)
            n0 = len(capture)
            moved = page.evaluate(
                f"""() => {{ const el = document.getElementById('{MP}-replay-slider');
                       if (!el) return false;
                       const h = el.querySelector('[role=slider]') || el.querySelector('.rc-slider-handle');
                       if (!h) return false; h.focus();
                       return true; }}"""
            )
            if moved:
                for _ in range(10):
                    page.keyboard.press("ArrowRight")
                    page.wait_for_timeout(120)
            page.wait_for_timeout(int(args.settle * 1000))
            after18 = _snap(page)
            reqs18 = [c for c in capture[n0:] if "_dash-update-component" in (c.get("url") or "") and "replay" in (c.get("body") or "")]
            res["steps"].append({"label": "M-METRICS-18 slider", "component": "replay-slider", "clicked": moved, "expect": "value maps to index, mode paused", "before": before18, "after": after18, "requests_naming_replay": len(reqs18)})
            log(f"  M-METRICS-18 slider: handle focused={moved} value={before18['slider_value']}->{after18['slider_value']} pos={before18['position']!r}->{after18['position']!r} reqs naming replay={len(reqs18)}")
            shot(page, "REPLAY__after_drive.png")
            res["final"] = _snap(page)
        finally:
            browser.close()

    # --- verdicts, from the rules in the docstring ---
    def mode(s):
        rs = s.get("replay_state")
        return rs.get("mode") if isinstance(rs, dict) else None

    def idx(s):
        rs = s.get("replay_state")
        return rs.get("current_index") if isinstance(rs, dict) else None

    v: dict = {}
    max_index = max(0, (res["initial"].get("metrics_store_len") or 0) - 1)
    toggled_on = s13a["after"]["play_text"] == "⏸" and s13a["after"]["interval_disabled"] is False and mode(s13a["after"]) == "playing"
    toggled_off = s13b["after"]["play_text"] == "▶" and s13b["after"]["interval_disabled"] is True and mode(s13b["after"]) == "paused"
    v["M-METRICS-13"] = "PASS" if (toggled_on and toggled_off) else ("FAIL" if s13a["clicked"] else "BLOCKED (button absent)")
    v["M-METRICS-16"] = "PASS" if (s16_2["after"]["interval_ms"] == 500 and s16_4["after"]["interval_ms"] == 250 and s16_1["after"]["interval_ms"] == 1000) else "FAIL"
    blocked = f"BLOCKED-BY-F-035 (metrics store len {res['initial'].get('metrics_store_len')} -> max_index {max_index}; every index transition clamps to 0 by metrics_panel.py:1013/1060/1088)"
    if max_index > 0:
        v["M-METRICS-14"] = "PASS" if idx(s14["after"]) == 1 else "FAIL"
        v["M-METRICS-12"] = "PASS" if idx(s12["after"]) == 0 else "FAIL"
        v["M-METRICS-15"] = "PASS" if idx(s15["after"]) == max_index else "FAIL"
        v["M-METRICS-11"] = "PASS" if idx(s11["after"]) == 0 else "FAIL"
        v["M-METRICS-18"] = "PASS" if (res["steps"][-1]["after"]["slider_value"] or 0) > 0 else "FAIL"
    else:
        for row, s in (("M-METRICS-14", s14), ("M-METRICS-12", s12), ("M-METRICS-15", s15), ("M-METRICS-11", s11)):
            ran = s["requests_naming_replay"] > 0 and mode(s["after"]) == "paused"
            v[row] = blocked if ran else "FAIL (the click produced no replay-state write)"
        v["M-METRICS-18"] = blocked if res["steps"][-1]["requests_naming_replay"] > 0 or res["steps"][-1]["clicked"] else "BLOCKED (slider handle not focusable)"
    res["verdicts"] = v
    res["max_index"] = max_index
    log("")
    for k, val in v.items():
        log(f"  => {k}: {val}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
