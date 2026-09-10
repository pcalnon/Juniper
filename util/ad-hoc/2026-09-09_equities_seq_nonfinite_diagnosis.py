#!/usr/bin/env python3
"""Locate the non-finite values that stop the LMU fitting canopy's equities_seq seed.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

2026-09-09_equities_seq_fit_budget.py got past the symbol cap and generated 25,586
train sequences, then LMURegressor.fit raised ``ValueError: u must be finite (no
NaN/Inf)``. So fixing the cap is necessary but NOT sufficient: (recurrence,
equities_seq) still cannot train.

Answer, before proposing any params change:
  * is the non-finiteness in X, in y, or both;
  * WHICH of the 16 features carry it, and what fraction of rows;
  * is it concentrated at the start of each window (a warm-up artefact) or spread;
  * do the partial-data policy knobs (fundamentals_fill / incomplete_rows, with
    allow_truncation opening the gate) actually clear it.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_equities_seq_nonfinite_diagnosis.py
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from juniper_data.api.routes.generators import GENERATOR_REGISTRY

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def _generate(**overrides: Any) -> dict[str, Any]:
    info = GENERATOR_REGISTRY["equities_seq"]
    params = info["params_class"](symbols=SYMBOLS, regression_target="return", **overrides)
    result = info["generator"].generate(params)
    data = result[0] if isinstance(result, tuple) else result
    return data if isinstance(data, dict) else getattr(data, "__dict__", {})


def _report(label: str, arrays: dict[str, Any]) -> None:
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        arr = arrays.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        bad = ~np.isfinite(arr)
        n_bad = int(bad.sum())
        print(f"  {key:8s} shape={str(arr.shape):20s} non-finite={n_bad:>10,} ({100.0 * n_bad / arr.size:6.3f}%)")

    x = np.asarray(arrays["X_train"], dtype=np.float64)
    bad = ~np.isfinite(x)
    if not bad.any():
        print("  X_train is entirely finite.")
        return

    # (n, T, F): which features, and where in the window.
    per_feature = bad.any(axis=(0, 1))
    print(f"\n  features carrying non-finite values: {np.flatnonzero(per_feature).tolist()} of {x.shape[2]}")
    for f in np.flatnonzero(per_feature):
        col = bad[:, :, f]
        rows_hit = int(col.any(axis=1).sum())
        print(f"    feature {f:2d}: {int(col.sum()):>9,} cells, in {rows_hit:,}/{x.shape[0]:,} windows ({100.0 * rows_hit / x.shape[0]:5.1f}%)")

    per_step = bad.any(axis=(0, 2))
    hit_steps = np.flatnonzero(per_step)
    print(f"\n  window positions affected: {hit_steps.size}/{x.shape[1]}", end="")
    print(f"  first={hit_steps[:5].tolist()} last={hit_steps[-5:].tolist()}" if hit_steps.size else "")

    all_bad_windows = int(bad.all(axis=(1, 2)).sum())
    any_bad_windows = int(bad.any(axis=(1, 2)).sum())
    print(f"  windows with ANY non-finite: {any_bad_windows:,}/{x.shape[0]:,}; entirely non-finite: {all_bad_windows:,}")


def main() -> int:
    print("Control — the params PR 2 proposes (explicit symbols, nothing else):")
    _report("control: symbols + regression_target", _generate())

    # Does opening the partial-data gate clear it? allow_truncation is the gate;
    # incomplete_rows then says accept (fill) or drop.
    for label, overrides in (
        ("allow_truncation + incomplete_rows='drop'", {"allow_truncation": True, "incomplete_rows": "drop"}),
        ("allow_truncation + incomplete_rows='accept'", {"allow_truncation": True, "incomplete_rows": "accept"}),
    ):
        try:
            _report(label, _generate(**overrides))
        except Exception as exc:  # noqa: BLE001 — a refusal is a result
            print(f"\n{label}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
