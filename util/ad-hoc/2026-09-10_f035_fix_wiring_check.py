#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""F-CANOPY-035 fix -- static wiring check against the BUILT canopy app.

Builds ``DashboardManager({})`` and asserts the six properties the fix depends on,
from ``app.callback_map`` and the real layout rather than from the source text. An
AST/grep pass over canopy's frontend resolves only 151 of 182 callbacks and has
already missed two real pollers (juniper-ml memory: canopy poller census gotcha), so
anything load-bearing reads the built app.

Run with canopy's src on the path:
    LIBTORCH= LD_LIBRARY_PATH= PYTHONPATH=<canopy-worktree>/src \\
        /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-10_f035_fix_wiring_check.py
"""

import sys


def main() -> int:
    from frontend.dashboard_manager import _METRICS_STORE_INTERVAL, DashboardManager

    dm = DashboardManager({})
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")

    def walk(c):
        yield c
        ch = getattr(c, "children", None)
        if ch is None:
            return
        if not isinstance(ch, (list, tuple)):
            ch = [ch]
        for x in ch:
            if hasattr(x, "children") or hasattr(x, "id"):
                yield from walk(x)

    ids = {getattr(c, "id", None) for c in walk(dm.app.layout)}
    check("the dedicated Interval exists in the layout", _METRICS_STORE_INTERVAL in ids, _METRICS_STORE_INTERVAL)

    found = None
    for key, entry in dm.app.callback_map.items():
        cb = entry.get("callback")
        raw = getattr(cb, "__wrapped__", cb)
        if getattr(raw, "__name__", None) == "update_metrics_store":
            found = (key, entry)
    check("update_metrics_store is registered", found is not None)
    if not found:
        return 1

    key, entry = found
    inputs = [f"{i.get('id')}.{i.get('property')}" for i in entry.get("inputs", [])]
    states = [f"{i.get('id')}.{i.get('property')}" for i in entry.get("state", [])]
    print(f"       output={key}")
    print(f"       inputs={inputs}")
    print(f"       state={states}")

    check("it is triggered by the DEDICATED interval",
          f"{_METRICS_STORE_INTERVAL}.n_intervals" in inputs)
    check("it no longer rides the fast lane",
          not any(i.startswith("fast-update-interval") for i in inputs))

    # ``running`` is NOT kept on the callback_map entry: dash/_callback.py:326 puts it
    # on the callback SPEC, which lives in ``app._callback_list`` and is what gets
    # served as ``_dash-dependencies`` and read by the renderer. Look there.
    running = entry.get("running")
    if not running:
        for spec in getattr(dm.app, "_callback_list", []):
            if str(spec.get("output")) == str(key):
                running = spec.get("running")
                break
    print(f"       running={running}")
    check("it declares a running= guard", bool(running))
    if running:
        on = running.get("running", {})
        off = running.get("runningOff", {})
        prop = f"{_METRICS_STORE_INTERVAL}.disabled"
        check("the guard disables ITS OWN interval while in flight",
              any(prop in str(k) for k in on) and all(v is True for v in on.values()),
              f"on={on}")
        check("the guard re-enables it on completion",
              any(prop in str(k) for k in off) and all(v is False for v in off.values()),
              f"off={off}")

    writers = [k for k in dm.app.callback_map if f"{_METRICS_STORE_INTERVAL}.disabled" in k]
    check("exactly one REGISTERED writer of its disabled prop (the CAN-000 gate)",
          len(writers) == 1, f"{writers}")

    fast = []
    for e in dm.app.callback_map.values():
        cb = e.get("callback")
        raw = getattr(cb, "__wrapped__", cb)
        iid = {i.get("id") for i in e.get("inputs", []) if isinstance(i, dict)}
        if "fast-update-interval" in iid:
            n = getattr(raw, "__name__", None)
            if n:
                fast.append(n)
    print(f"       fast-lane server-side riders now: {sorted(fast)}")
    check("the fast lane keeps its other two server-side riders",
          sorted(fast) == sorted(["update_unified_status_bar", "handle_button_timeout_and_acks"]),
          f"{sorted(fast)}")

    print(f"\n  {'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
