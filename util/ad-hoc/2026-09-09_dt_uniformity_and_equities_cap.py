#!/usr/bin/env python3
"""Verify two census results that would change the canopy §12 seeding work.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

(A) dt uniformity. 2026-09-09_generator_gap_census.py reported uniform=False for ALL
    five rank-3 generators, contradicting §12.1 of
    notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md, which
    names only irregular_sine and delay_product as non-uniform. The census took
    np.unique over the WHOLE dt array; if dt[:, 0] is a 0 sentinel (no previous step)
    that alone yields two values and the census is measuring its own convention, not
    the generator's. Re-measure excluding the first column and print the raw head.

(B) The equities cap. The census hit InputTooLargeError at max_symbols=1 because the
    universe (503) exceeds the cap. canopy's SHIPPED equities_seq seed carries
    default_params={"max_symbols": 5, "regression_target": "return"} and nothing else,
    so if the cap refuses before truncation, the one seeded rank-3 dataset cannot
    generate at all. Probe max_symbols=5 exactly as canopy sends it.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_dt_uniformity_and_equities_cap.py
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from juniper_data.api.routes.generators import GENERATOR_REGISTRY

RANK3 = ("multi_sine", "mackey_glass", "ar_p", "irregular_sine", "delay_product")


def _generate(name: str, **overrides: Any) -> dict[str, Any]:
    info = GENERATOR_REGISTRY[name]
    params = info["params_class"](**overrides)
    result = info["generator"].generate(params)
    data = result[0] if isinstance(result, tuple) else result
    return data if isinstance(data, dict) else getattr(data, "__dict__", {})


def part_a() -> None:
    print("=" * 78)
    print("(A) dt uniformity — whole array vs. excluding the first column")
    print("=" * 78)
    for name in RANK3:
        arrays = _generate(name, n_steps=40, window_length=8, n_windows=6)
        dt = np.asarray(arrays["dt_train"], dtype=np.float64)
        whole = np.unique(np.round(dt, 9))
        tail = np.unique(np.round(dt[:, 1:], 9)) if dt.ndim == 2 and dt.shape[1] > 1 else whole
        print(f"\n{name}:")
        print(f"  dt shape           : {dt.shape}")
        print(f"  row 0 head         : {np.round(dt[0][:6], 6).tolist()}")
        print(f"  unique WHOLE       : n={whole.size} {np.round(whole[:6], 6).tolist()}")
        print(f"  unique EXCL col 0  : n={tail.size} {np.round(tail[:6], 6).tolist()}")
        print(f"  VERDICT            : {'REGULAR' if tail.size <= 1 else 'IRREGULAR'} (excluding col 0)")


def part_b() -> None:
    print()
    print("=" * 78)
    print("(B) equities cap — canopy's shipped default_params, verbatim")
    print("=" * 78)
    canopy_seed = {"max_symbols": 5, "regression_target": "return"}
    for name in ("equities_seq", "equities"):
        if name == "equities":
            probe = {"max_symbols": 5}
        else:
            probe = dict(canopy_seed)
        print(f"\n{name} with {probe}:")
        try:
            arrays = _generate(name, **probe)
            x = np.asarray(arrays["X_train"])
            print(f"  OK  X_train.shape={x.shape} ndim={x.ndim}")
        except Exception as exc:  # noqa: BLE001 — reporting the refusal IS the result
            print(f"  {type(exc).__name__}: {exc}")

    print("\n  --- does allow_truncation exist as a param? ---")
    for name in ("equities_seq", "equities"):
        fields = sorted(GENERATOR_REGISTRY[name]["params_class"].model_fields)
        print(f"  {name}: allow_truncation={'allow_truncation' in fields}")
        print(f"    fields: {fields}")


def main() -> int:
    part_a()
    part_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())
