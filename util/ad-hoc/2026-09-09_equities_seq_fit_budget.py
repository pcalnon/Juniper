#!/usr/bin/env python3
"""Measure the LMU fit budget for a candidate canopy equities_seq seed.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

canopy's model_registry comment says equities_seq's default_params were chosen to keep
the one-shot fit "inside the 300 s train timeout". 2026-09-09_equities_seq_remedies.py
showed the SHIPPED params generate nothing at all (InputTooLargeError), so that budget
was never actually exercised. Changing the params without re-measuring would repeat the
error §12.4 of the selection-reachability design calls out -- grading a count instead of
a measured capability.

Measures, for each candidate symbol list: generation wall-clock, resulting shapes, and
LMU fit wall-clock at the service's effective defaults (d=16, theta data-driven,
ridge=0.0 linear readout), then reports the total against the 300 s budget.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_equities_seq_fit_budget.py
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

CANDIDATES: dict[str, list[str]] = {
    "5 mega-caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
}


def _measure(label: str, symbols: list[str]) -> None:
    print(f"\n{'=' * 74}\n{label}: {symbols}\n{'=' * 74}")
    info = GENERATOR_REGISTRY["equities_seq"]
    params = info["params_class"](symbols=symbols, regression_target="return")

    t0 = time.monotonic()
    result = info["generator"].generate(params)
    gen_s = time.monotonic() - t0
    data = result[0] if isinstance(result, tuple) else result
    arrays: dict[str, Any] = data if isinstance(data, dict) else getattr(data, "__dict__", {})

    x_tr = np.asarray(arrays["X_train"])
    y_tr = np.asarray(arrays["y_train"])
    dt_tr = arrays.get("dt_train")
    print(f"  generate      : {gen_s:7.1f}s")
    print(f"  X_train       : {x_tr.shape}  y_train: {y_tr.shape}")
    for key in ("X_val", "X_test"):
        if arrays.get(key) is not None:
            print(f"  {key:14s}: {np.asarray(arrays[key]).shape}")
    print(f"  dt_train      : {'present ' + str(np.asarray(dt_tr).shape) if dt_tr is not None else 'absent'}")

    model = LMURegressor()  # service effective defaults: d=16, theta data-driven, ridge=0.0
    fit_kw: dict[str, Any] = {}
    if dt_tr is not None:
        fit_kw["dt"] = np.asarray(dt_tr, dtype=float)
    t1 = time.monotonic()
    train_result = model.fit(x_tr, y_tr, **fit_kw)
    fit_s = time.monotonic() - t1

    total = gen_s + fit_s
    print(f"  fit           : {fit_s:7.1f}s  (d={model.d}, theta={model.theta:.3f}, ridge={model.ridge})")
    print(f"  TOTAL         : {total:7.1f}s  vs {TRAIN_TIMEOUT_S:.0f}s budget -> {'OK' if total < TRAIN_TIMEOUT_S else 'OVER BUDGET'}")
    metrics = getattr(train_result, "final_metrics", None) or getattr(train_result, "metrics", None)
    print(f"  final_metrics : {metrics}")


def main() -> int:
    for label, symbols in CANDIDATES.items():
        try:
            _measure(label, symbols)
        except Exception as exc:  # noqa: BLE001 — a failure to fit IS the measurement
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
