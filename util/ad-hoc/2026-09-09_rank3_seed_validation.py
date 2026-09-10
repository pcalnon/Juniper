#!/usr/bin/env python3
"""§12.4 validation for the five rank-3 synthetic generators canopy has never seeded.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

§12.4 of notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md requires,
per newly-seeded generator, that generate -> train be OBSERVED once -- "a generator that
cannot complete that sequence is seeded disabled with a reason, not seeded silently broken".
The equities_seq repair (canopy#610) is the cautionary case: it passed a count check for
weeks and could neither generate nor fit.

For each rank-3 generator this measures, at the generator's OWN defaults (i.e. what canopy
would send with empty ``default_params``):

  * generate wall-clock, X_train shape, and non-finite count;
  * dt presence and whether it is uniform EXCLUDING column 0 (that column is a
    no-previous-step 0.0 sentinel, and including it makes every generator look irregular);
  * LMU fit wall-clock at the recurrence service's effective defaults, plus final metrics.

Total is reported against the 300 s train timeout, and r2 is reported so a generator that
fits but learns nothing is visible as such.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_rank3_seed_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_SRC = Path("/home/pcalnon/Development/python/Juniper/juniper-recurrence/juniper-recurrence-model")
if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))

from juniper_data.api.routes.generators import GENERATOR_REGISTRY  # noqa: E402
from juniper_recurrence_model.model import LMURegressor  # noqa: E402

TRAIN_TIMEOUT_S = 300.0
RANK3 = ("multi_sine", "mackey_glass", "ar_p", "irregular_sine", "delay_product")


def _validate(name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name}
    info = GENERATOR_REGISTRY[name]
    try:
        params = info["params_class"]()  # generator's OWN defaults -- canopy sends nothing
        t0 = time.monotonic()
        result = info["generator"].generate(params)
        row["generate_s"] = round(time.monotonic() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        row["verdict"] = f"GENERATE FAILED: {type(exc).__name__}: {exc}"
        return row

    data = result[0] if isinstance(result, tuple) else result
    arrays: dict[str, Any] = data if isinstance(data, dict) else getattr(data, "__dict__", {})
    x_tr = np.asarray(arrays["X_train"])
    y_tr = np.asarray(arrays["y_train"])
    dt_tr = arrays.get("dt_train")

    row["X_train"] = tuple(int(d) for d in x_tr.shape)
    row["y_train"] = tuple(int(d) for d in y_tr.shape)
    row["nonfinite_X_train"] = int((~np.isfinite(np.asarray(x_tr, dtype=np.float64))).sum())
    row["nonfinite_X_val"] = int((~np.isfinite(np.asarray(arrays["X_val"], dtype=np.float64))).sum()) if arrays.get("X_val") is not None else None

    if dt_tr is None:
        row["dt"] = "ABSENT"
    else:
        dt = np.asarray(dt_tr, dtype=np.float64)
        tail = np.unique(np.round(dt[:, 1:], 9)) if dt.ndim == 2 and dt.shape[1] > 1 else np.unique(np.round(dt, 9))
        row["dt"] = f"regular({tail[0]:g})" if tail.size == 1 else f"irregular({tail.size} values)"

    if row["nonfinite_X_train"]:
        row["verdict"] = "NON-FINITE X_train -- the LMU will refuse it"
        return row

    try:
        model = LMURegressor()
        fit_kw: dict[str, Any] = {"dt": np.asarray(dt_tr, dtype=float)} if dt_tr is not None else {}
        t1 = time.monotonic()
        train_result = model.fit(x_tr, y_tr, **fit_kw)
        row["fit_s"] = round(time.monotonic() - t1, 2)
    except Exception as exc:  # noqa: BLE001
        row["verdict"] = f"FIT FAILED: {type(exc).__name__}: {exc}"
        return row

    total = row["generate_s"] + row["fit_s"]
    row["total_s"] = round(total, 2)
    metrics = getattr(train_result, "final_metrics", None) or getattr(train_result, "metrics", None) or {}
    row["r2"] = round(float(metrics.get("r2")), 4) if metrics.get("r2") is not None else None
    row["mse"] = round(float(metrics.get("mse")), 6) if metrics.get("mse") is not None else None
    row["verdict"] = "OK" if total < TRAIN_TIMEOUT_S else f"OVER BUDGET ({total:.0f}s)"
    return row


def main() -> int:
    rows = [_validate(name) for name in RANK3]
    width = max(len(r["name"]) for r in rows)
    print(f"\n{'generator':<{width}}  {'X_train':>18}  {'dt':<22}  {'gen':>6}  {'fit':>7}  {'r2':>8}  verdict")
    print("-" * (width + 84))
    for r in rows:
        print(
            f"{r['name']:<{width}}  {str(r.get('X_train', '-')):>18}  {str(r.get('dt', '-')):<22}  "
            f"{r.get('generate_s', '-'):>6}  {r.get('fit_s', '-'):>7}  {str(r.get('r2', '-')):>8}  {r['verdict']}"
        )
    print("\nfull rows:")
    for r in rows:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
