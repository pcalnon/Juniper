#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# ---------------------------------------------------------------------------
"""Does cascor drop its training-stream subscribers when it broadcasts a state frame?

THE OBSERVATION. 2026-09-08, both growth runs: cascor's own log shows the two canopy
relays "promoted to active" and then, at 07:26:41.941 / .943 -- one to three
milliseconds after "State transition: ResumeReady -> Started" -- both "disconnected".
Its per-minute summaries read "0 active connections" for the whole of both runs,
while both canopy legs kept logging their relay as connected and ``status=healthy``
with 0 frames and a frozen last-frame age. Every field the candidate panel, the
phase-duration text, the WS metrics path and the cascade-add trigger depend on
arrives only over that stream.

THE CANDIDATE MECHANISM, from source: ``manager._send_json`` swallows any exception
(``except Exception: return False``, no log), ``broadcast`` then calls
``disconnect()`` on the failed client -- which FORGETS the socket without closing it
-- and Starlette's ``send_json`` is stdlib ``json.dumps`` with no ``default``. And
``snapshot_serializer._load_config_to_network`` restores every runtime tunable
(``learning_rate``, ``epochs_max``, ``patience``, ...) straight from HDF5 attrs, i.e.
as NumPy scalars, which ``json.dumps`` cannot serialise. So the first state frame
after a restore should raise inside ``send_json``, and every subscriber is dropped
silently and left half-open.

THE TEST. Connect a raw WebSocket client to ``/ws/training``, wait past the 5 s
resume handshake so it is promoted, then trigger a transition that broadcasts a
state frame WITHOUT training anything: ``POST /v1/snapshots/<id>/resume`` calls
``_broadcast_training_state(force=True)`` after reloading the same network. Then
watch: frames received, and whether the socket is closed (with what code) or left
open and silent.

  DROPPED-SILENT   no frame after the trigger and no close frame either: the server
                   forgot us without closing -- the half-open state canopy's relay is
                   sitting in.
  DROPPED-CLOSED   a close frame arrived: the server closed properly (a client would
                   reconnect).
  STATE-RECEIVED   a ``state`` frame arrived after the trigger: no drop on this
                   transition -- the mechanism above is NOT what happened.

Run it once against a cascor whose loader leaves NumPy scalars in place and once
against the fixed loader (juniper-cascor#632 extended): the verdict should flip.

Usage:
    LD_LIBRARY_PATH= /opt/miniforge3/envs/JuniperCanopy1/bin/python \\
        util/ad-hoc/2026-09-08_cascor_ws_drop_probe.py --snapshot snapshot_20260908T123427Z
"""

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

CASCOR = os.environ.get("JUNIPER_E2E_CASCOR_URL", "http://127.0.0.1:8202")
OUT = os.environ.get("WS_DROP_RESULTS", "/tmp/juniper-e2e/cascor_ws_drop.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _post(path: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(CASCOR + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return {"http": r.status, "body": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        return {"http": e.code, "body": e.read().decode()[:300]}


def _get(path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(CASCOR + path, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode())


async def main_async(args) -> dict:
    import websockets

    ws_url = CASCOR.replace("http://", "ws://").replace("https://", "wss://") + "/ws/training"
    res: dict = {"cascor": CASCOR, "ws_url": ws_url, "snapshot": args.snapshot, "frames_before": [], "frames_after": [], "close": None}
    try:
        h = _get("/v1/health")
        res["cascor_git_sha"] = h.get("git_sha")
    except Exception as exc:  # noqa: BLE001
        res["cascor_git_sha"] = f"<unreadable: {exc}>"

    # THE SERVER'S OWN COUNTERS ARE THE VERDICT, NOT THE PINGS. cascor's ping is sent by
    # each handler's per-connection task, which keeps running after ``broadcast`` has
    # forgotten the socket -- so run A (2026-09-08, the leg with NumPy-typed tunables)
    # received a ping 26 s after the trigger while cascor's own summary six seconds after
    # the resume read "0 active connections". ``/v1/metrics/transport`` exposes
    # ``active_connections`` and ``send_failures``; their deltas across the trigger are
    # what distinguish "forgotten" from "served".
    def _transport() -> dict:
        try:
            d = _get("/v1/metrics/transport").get("data") or {}
            return {k: d.get(k) for k in ("active_connections", "pending_connections", "send_failures", "messages_sent_by_type")}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"[:100]}

    async with websockets.connect(ws_url, max_size=None) as ws:
        log(f"connected to {ws_url}")
        t0 = time.monotonic()
        # Pre-trigger: drain the initial burst and sit through the 5 s resume handshake.
        while time.monotonic() - t0 < args.pre:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                m = json.loads(raw)
                res["frames_before"].append({"t": round(time.monotonic() - t0, 2), "type": m.get("type")})
                if m.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
            except ValueError:
                res["frames_before"].append({"t": round(time.monotonic() - t0, 2), "type": "<non-json>"})
        log(f"pre-trigger frames: {[f['type'] for f in res['frames_before']]}")

        res["transport_before"] = _transport()
        log(f"transport before: {json.dumps(res['transport_before'])[:200]}")

        # The trigger: a resume of the SAME network broadcasts a state frame, trains nothing.
        t_trig = time.monotonic()
        res["trigger"] = _post(f"/v1/snapshots/{args.snapshot}/resume")
        log(f"trigger: POST /v1/snapshots/{args.snapshot}/resume -> http {res['trigger'].get('http')}")

        closed = None
        while time.monotonic() - t_trig < args.post:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed as cc:
                closed = {"t": round(time.monotonic() - t_trig, 2), "code": cc.code, "reason": str(cc.reason)[:120]}
                log(f"CLOSED by server: code={cc.code} reason={cc.reason!r} at +{closed['t']}s")
                break
            try:
                m = json.loads(raw)
                res["frames_after"].append({"t": round(time.monotonic() - t_trig, 2), "type": m.get("type")})
                if m.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                log(f"  +{round(time.monotonic() - t_trig, 2)}s frame type={m.get('type')}")
            except ValueError:
                res["frames_after"].append({"t": round(time.monotonic() - t_trig, 2), "type": "<non-json>"})
        res["close"] = closed
        # A ping from the server after the trigger proves the socket is still SERVED;
        # a silence proves it is not (pings are periodic while a socket is active).
        res["socket_open_at_end"] = closed is None
        try:
            # A liveness probe of our own: sending must not raise on an open socket.
            await ws.send(json.dumps({"type": "pong"}))
            res["send_after_trigger_ok"] = True
        except Exception as exc:  # noqa: BLE001
            res["send_after_trigger_ok"] = f"{type(exc).__name__}: {exc}"[:120]

    res["transport_after"] = _transport()
    log(f"transport after : {json.dumps(res['transport_after'])[:200]}")

    types_after = [f["type"] for f in res["frames_after"]]
    tb, ta = res.get("transport_before") or {}, res.get("transport_after") or {}
    d_fail = (ta.get("send_failures") or 0) - (tb.get("send_failures") or 0) if "send_failures" in ta and "send_failures" in tb else None
    active_after = ta.get("active_connections")
    if "state" in types_after or "initial_status" in types_after:
        res["verdict"] = "STATE-RECEIVED"
        res["why"] = (f"frames after the trigger: {types_after}; send_failures delta {d_fail}, active_connections after {active_after}. "
                      "The broadcast reached this subscriber: no drop on this transition.")
    elif closed is not None:
        res["verdict"] = "DROPPED-CLOSED"
        res["why"] = f"the server closed the socket {closed['t']}s after the trigger (code {closed['code']})."
    elif d_fail is not None and d_fail > 0:
        res["verdict"] = "DROPPED-SILENT"
        res["why"] = (f"send_failures rose by {d_fail} across the trigger and active_connections is now {active_after}, while this socket "
                      f"stayed open (frames after: {types_after} -- pings come from the handler's own task, not from the manager). "
                      "The server forgot this subscriber without closing it: the half-open state canopy's relay reports as healthy.")
    elif not types_after and active_after == 0:
        res["verdict"] = "DROPPED-SILENT"
        res["why"] = "no frame after the trigger and the server counts 0 active connections: forgotten, not closed."
    else:
        res["verdict"] = "INDETERMINATE"
        res["why"] = (f"frames after the trigger were {types_after}; send_failures delta {d_fail}; active_connections after {active_after}. "
                      "No state frame and no failure counted -- read the cascor log for this window.")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="snapshot id to /resume (the SAME network the leg already holds)")
    ap.add_argument("--pre", type=float, default=9.0, help="seconds to sit connected before the trigger (past the 5 s resume handshake)")
    ap.add_argument("--post", type=float, default=45.0, help="seconds to watch after the trigger (server pings are periodic)")
    args = ap.parse_args()
    res = asyncio.run(main_async(args))
    log("")
    log(f"  frames before: {[f['type'] for f in res['frames_before']]}")
    log(f"  frames after : {[f['type'] for f in res['frames_after']]}")
    log(f"  close        : {res['close']}")
    log(f"  => VERDICT: {res['verdict']}")
    log(f"     {res['why']}")
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"results -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
