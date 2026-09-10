#!/usr/bin/env python3
"""Test the two documented remedies for canopy's ungenerable equities_seq seed.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

2026-09-09_dt_uniformity_and_equities_cap.py established that canopy's SHIPPED
equities_seq default_params ({"max_symbols": 5, "regression_target": "return"})
are refused by juniper-data with InputTooLargeError, because the default universe
is 503 names and max_symbols is a CAP that refuses rather than truncates.

juniper-data/docs/DEVELOPER_CHEATSHEET.md:305 names two remedies. Measure both, so
canopy's registry seed can be changed on evidence rather than on the doc's say-so:

  R1: allow_truncation=true    (+ max_symbols=5)  -> leading 5 of the universe
  R2: symbols=[...]            (an explicit <=cap universe)

Report, per remedy: whether it generates, the resulting rank/shape, and any
data-quality annotation the dataset carries.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_equities_seq_remedies.py
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from juniper_data.api.routes.generators import GENERATOR_REGISTRY

# canopy's shipped seed, verbatim from model_registry.DATASET_TYPES.
CANOPY_SEED: dict[str, Any] = {"max_symbols": 5, "regression_target": "return"}

REMEDIES: dict[str, dict[str, Any]] = {
    "R0 shipped (control)": dict(CANOPY_SEED),
    "R1 allow_truncation": {**CANOPY_SEED, "allow_truncation": True},
    "R2 explicit symbols": {**CANOPY_SEED, "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]},
}


def _run(name: str, overrides: dict[str, Any]) -> None:
    info = GENERATOR_REGISTRY["equities_seq"]
    print(f"\n--- {name} ---")
    print(f"  params: {overrides}")
    started = time.monotonic()
    try:
        params = info["params_class"](**overrides)
        result = info["generator"].generate(params)
        elapsed = time.monotonic() - started
        data = result[0] if isinstance(result, tuple) else result
        meta = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        arrays = data if isinstance(data, dict) else getattr(data, "__dict__", {})
        x = np.asarray(arrays["X_train"])
        print(f"  RESULT : GENERATED in {elapsed:.1f}s")
        print(f"  X_train: shape={x.shape} ndim={x.ndim} dtype={x.dtype}")
        for key in ("X_val", "X_test"):
            if key in arrays and arrays[key] is not None:
                print(f"  {key:7s}: shape={np.asarray(arrays[key]).shape}")
        dq = None
        if meta is not None:
            dq = getattr(meta, "data_quality", None)
            if dq is None and isinstance(meta, dict):
                dq = meta.get("data_quality")
        print(f"  data_quality: {dq}")
    except Exception as exc:  # noqa: BLE001 — the refusal IS the measurement
        elapsed = time.monotonic() - started
        print(f"  RESULT : {type(exc).__name__} after {elapsed:.1f}s")
        print(f"  detail : {exc}")


def main() -> int:
    print("=" * 78)
    print("equities_seq — canopy's shipped seed vs. the two documented remedies")
    print("=" * 78)
    for name, overrides in REMEDIES.items():
        _run(name, overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
