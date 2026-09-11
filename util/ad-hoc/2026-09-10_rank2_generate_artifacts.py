#!/usr/bin/env python3
"""Stage 1 of the §12.4 rank-2 validation: generate NPZ artifacts for the unseeded rank-2 set.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

canopy#612 seeded the five rank-3 synthetics and left `gaussian`, `checkerboard` and
`equities` on G10's exclusion list pending "generate -> train observed once per new seed,
and that validation runs against the cascor backend rather than the LMU" (design §12.4,
notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md).

juniper_data and juniper-cascor live in different conda environments, so the validation
runs in two stages through the real artifact contract rather than in one process:

  stage 1 (this file, JuniperData)   generate -> NPZ on disk
  stage 2 (2026-09-10_rank2_cascor_fit.py, JuniperCascor1)   NPZ -> CascadeCorrelationNetwork.fit

That split is a feature: it exercises exactly the six-key NPZ contract juniper-data emits
and cascor consumes, instead of passing arrays in memory and assuming the artifact round-trip.

For `equities` the canopy#610 treatment is measured rather than assumed: its sibling
`equities_seq` needed an explicit `symbols` list (max_symbols is a cap juniper-data REFUSES
against) and `fundamentals_fill` (the "nan" default leaves X_train non-finite). Both the
bare and the treated form are attempted here so the seed's default_params rest on evidence.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-10_rank2_generate_artifacts.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from juniper_data.api.routes.generators import GENERATOR_REGISTRY, generator_available, generator_install_hint

OUT_DIR = Path("/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/febb940d-6152-433e-a6e6-6534f75e4907/scratchpad/rank2_artifacts")

# Each case: the generator, and the params canopy would send. A ``None`` params dict means
# "the generator's own defaults", which is what an empty registry ``default_params`` produces.
CASES: dict[str, dict[str, Any]] = {
    "gaussian": {},
    "checkerboard": {},
    # checkerboard's default is only 200 samples over a 4x4 grid (CHECKERBOARD_DEFAULT_N_SAMPLES
    # / _N_SQUARES), ~12 per cell. At that size CasCor recruited ONE unit and sat at chance, so
    # this case separates "the dataset is under-sampled" from "the cascade stalls regardless" --
    # a distinction that decides whether the seed needs a default_params n_samples or not.
    # ``n_samples`` is also exactly what canopy's sidebar sends, so this is an operator-reachable
    # knob rather than a hidden one.
    "checkerboard_2000": {"n_samples": 2000},
    "equities_bare": {},
    "equities_treated": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "fundamentals_fill": "drop",
    },
    # The canopy#610 treatment is necessary but NOT sufficient here. equities' feature
    # columns are raw market quantities -- close prices, volumes, market caps in the 1e11
    # range -- and ``normalize_features`` defaults to False
    # (EQUITIES_DEFAULT_NORMALIZE_FEATURES). Fed to CasCor unnormalised the first output
    # pass reports a loss of 5.8e+21 and top-1 sits at chance. This is the rank-2 analogue
    # of the fundamentals_fill discovery: a third key the seed must pin.
    "equities_normalized": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "fundamentals_fill": "drop",
        "normalize_features": True,
    },
}

GENERATOR_FOR_CASE = {
    "gaussian": "gaussian",
    "checkerboard": "checkerboard",
    "checkerboard_2000": "checkerboard",
    "equities_bare": "equities",
    "equities_treated": "equities",
    "equities_normalized": "equities",
}


def _run(case: str, overrides: dict[str, Any]) -> dict[str, Any]:
    name = GENERATOR_FOR_CASE[case]
    info = GENERATOR_REGISTRY[name]
    row: dict[str, Any] = {
        "case": case,
        "generator": name,
        "params": overrides,
        "declared_task_type": info["task_type"],
        "available": generator_available(info),
        "install_hint": generator_install_hint(info),
    }
    try:
        params = info["params_class"](**overrides)
        t0 = time.monotonic()
        result = info["generator"].generate(params)
        row["generate_s"] = round(time.monotonic() - t0, 2)
    except Exception as exc:  # noqa: BLE001 — a refusal is the measurement
        row["verdict"] = f"GENERATE FAILED: {type(exc).__name__}: {exc}"
        return row

    data = result[0] if isinstance(result, tuple) else result
    arrays: dict[str, Any] = data if isinstance(data, dict) else getattr(data, "__dict__", {})

    payload: dict[str, np.ndarray] = {}
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        arr = arrays.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        payload[key] = arr
        row[f"{key}_shape"] = tuple(int(d) for d in arr.shape)
        row[f"{key}_nonfinite"] = int((~np.isfinite(arr.astype(np.float64))).sum())

    if "X_train" not in payload:
        row["verdict"] = f"NO X_train (keys={sorted(arrays)[:8]})"
        return row

    row["rank"] = int(payload["X_train"].ndim)
    # A one-hot y is the classification tell; a single column is a regression target.
    y = payload.get("y_train")
    if y is not None and y.ndim == 2:
        row["y_width"] = int(y.shape[1])
        row["y_looks_one_hot"] = bool(np.all(np.isin(y, (0.0, 1.0))) and np.allclose(y.sum(axis=1), 1.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / f"{case}.npz"
    np.savez_compressed(npz_path, **payload)
    row["npz"] = str(npz_path)
    row["npz_mb"] = round(npz_path.stat().st_size / 1e6, 2)
    row["verdict"] = "GENERATED"
    return row


def main() -> int:
    rows = [_run(case, overrides) for case, overrides in CASES.items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage1.json").write_text(json.dumps(rows, indent=2, default=str))
    for row in rows:
        print(f"\n--- {row['case']} ({row['generator']}) ---")
        for key in ("verdict", "generate_s", "rank", "X_train_shape", "y_train_shape", "y_width", "y_looks_one_hot", "X_val_shape", "X_test_shape", "X_train_nonfinite", "declared_task_type", "available", "npz_mb"):
            if key in row:
                print(f"  {key:20s}: {row[key]}")
    print(f"\nwrote {OUT_DIR}/stage1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
