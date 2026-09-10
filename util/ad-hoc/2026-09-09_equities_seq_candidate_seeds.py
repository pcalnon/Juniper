#!/usr/bin/env python3
"""Choose canopy's equities_seq seed on measured generate+fit evidence, not on a count.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

Established by the three preceding probes in this directory:

  * the SHIPPED seed ({"max_symbols": 5, "regression_target": "return"}) is refused
    outright -- InputTooLargeError, universe 503 over a cap of 5;
  * an explicit ``symbols`` list clears the cap (juniper-ml's own
    tests/test_equities_symbol_cap_operator.py names this as THE remedy and warns off
    enabling allow_truncation stack-wide);
  * but the fit then dies on ``u must be finite``: X_train alone is 9.1% non-finite,
    in exactly 3 of 16 features, in 49.4% of windows. juniper-data's schema names the
    cause -- ``fundamentals_fill`` is "how to represent PRE-2009 missing total_shares /
    market_cap", and its default is "nan". ``incomplete_rows`` does not touch it.

So the seed needs BOTH halves. Measure each candidate the way §12.4 of the
selection-reachability design requires -- generate, then actually fit -- and report
wall-clock against the 300 s train timeout the registry comment invokes.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_equities_seq_candidate_seeds.py
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
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
BASE: dict[str, Any] = {"symbols": SYMBOLS, "regression_target": "return"}

CANDIDATES: dict[str, dict[str, Any]] = {
    "C1 fundamentals_fill=zero": {**BASE, "fundamentals_fill": "zero"},
    "C2 fundamentals_fill=drop": {**BASE, "fundamentals_fill": "drop"},
    "C3 start_date=2010-01-01": {**BASE, "start_date": "2010-01-01"},
}


def _measure(label: str, overrides: dict[str, Any]) -> None:
    print(f"\n{'=' * 74}\n{label}\n  {overrides}\n{'=' * 74}")
    info = GENERATOR_REGISTRY["equities_seq"]
    try:
        params = info["params_class"](**overrides)
        t0 = time.monotonic()
        result = info["generator"].generate(params)
        gen_s = time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        print(f"  GENERATE FAILED: {type(exc).__name__}: {exc}")
        return

    data = result[0] if isinstance(result, tuple) else result
    arrays: dict[str, Any] = data if isinstance(data, dict) else getattr(data, "__dict__", {})
    x_tr = np.asarray(arrays["X_train"])
    y_tr = np.asarray(arrays["y_train"])
    dt_tr = arrays.get("dt_train")

    nonfinite = int((~np.isfinite(np.asarray(x_tr, dtype=np.float64))).sum())
    print(f"  generate      : {gen_s:7.1f}s")
    print(f"  X_train       : {x_tr.shape}   non-finite={nonfinite:,}")

    if nonfinite:
        print("  -> still non-finite; the LMU will refuse it. Not fitting.")
        return

    model = LMURegressor()  # service effective defaults: d=16, theta data-driven, ridge=0.0
    fit_kw: dict[str, Any] = {}
    if dt_tr is not None:
        fit_kw["dt"] = np.asarray(dt_tr, dtype=float)
    try:
        t1 = time.monotonic()
        train_result = model.fit(x_tr, y_tr, **fit_kw)
        fit_s = time.monotonic() - t1
    except Exception as exc:  # noqa: BLE001
        print(f"  FIT FAILED: {type(exc).__name__}: {exc}")
        return

    total = gen_s + fit_s
    verdict = "OK" if total < TRAIN_TIMEOUT_S else "OVER BUDGET"
    print(f"  fit           : {fit_s:7.1f}s  (d={model.d}, theta={model.theta:.3f})")
    print(f"  TOTAL         : {total:7.1f}s  vs {TRAIN_TIMEOUT_S:.0f}s -> {verdict}")
    metrics = getattr(train_result, "final_metrics", None) or getattr(train_result, "metrics", None)
    print(f"  final_metrics : {metrics}")


def main() -> int:
    for label, overrides in CANDIDATES.items():
        _measure(label, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
