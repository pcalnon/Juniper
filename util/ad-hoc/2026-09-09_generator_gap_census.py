#!/usr/bin/env python3
"""Census juniper-data's GENERATOR_REGISTRY for the canopy §12 generator-gap seeding.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

Answers, by EXECUTING the registry rather than reading prose, the questions §12.3 of
notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md leaves open:

  * rank (X_train.ndim) of every registered generator, and whether that rank is
    FIXED or depends on a parameter;
  * task_type as juniper-data emits it (vs what canopy's registry asserts);
  * per-step dt presence (the LMU's requires_dt path) and whether it is uniform;
  * deployment availability + install hint;
  * the schema field count canopy's params panel would render.

Run:  conda run -n JuniperData python util/ad-hoc/2026-09-09_generator_gap_census.py
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

import numpy as np

from juniper_data.api.routes.generators import (
    GENERATOR_REGISTRY,
    generator_available,
    generator_install_hint,
)

# Bound every probe: this is a shape census, not a benchmark.
SMALL: dict[str, Any] = {
    "n_samples": 40,
    "n_steps": 40,
    "window_length": 8,
    "n_windows": 6,
    "max_symbols": 1,
    "n_tasks": 2,
}

# Fields excluded from canopy's rendered form (mirrors dataset_schema.FORM_EXCLUDED_FIELDS
# closely enough for a count; exact parity is asserted by the canopy-side test, not here).
INFRA_FIELDS = {
    "seed",
    "test_size",
    "val_size",
    "shuffle",
    "stratify",
    "dtype",
    "on_partial_data",
    "min_rows",
    "partial_data_policy",
}


def _fit_params(params_class: Any) -> Any:
    """Instantiate ``params_class`` with the smallest accepted value for each knob we cap."""
    fields = params_class.model_fields
    kwargs = {k: v for k, v in SMALL.items() if k in fields}
    return params_class(**kwargs)


def _probe(name: str, info: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "declared_task_type": info["task_type"],
        "version": info["version"],
        "available": generator_available(info),
        "install_hint": generator_install_hint(info),
    }
    params_class = info["params_class"]
    schema = params_class.model_json_schema()
    props = schema.get("properties", {})
    row["schema_fields_total"] = len(props)
    row["schema_fields_rendered"] = len([p for p in props if p not in INFRA_FIELDS])
    row["has_seed"] = "seed" in props

    if not row["available"]:
        row["rank"] = "SKIPPED (unavailable)"
        return row

    try:
        params = _fit_params(params_class)
        result = info["generator"].generate(params)
        data = result[0] if isinstance(result, tuple) else result
        arrays = data if isinstance(data, dict) else getattr(data, "__dict__", {})
        x_train = arrays.get("X_train")
        if x_train is None:
            row["rank"] = f"NO X_train (keys={sorted(arrays)[:8]})"
            return row
        x_train = np.asarray(x_train)
        row["rank"] = int(x_train.ndim)
        row["X_train_shape"] = tuple(int(d) for d in x_train.shape)
        dt = arrays.get("dt_train")
        if dt is None:
            row["dt"] = "absent"
        else:
            dt = np.asarray(dt)
            uniq = np.unique(np.round(dt.astype(np.float64), 9))
            row["dt"] = f"present shape={tuple(int(d) for d in dt.shape)} uniform={uniq.size <= 1}"
    except Exception as exc:  # noqa: BLE001 — a census must report a failure, not die on it
        row["rank"] = f"ERROR {type(exc).__name__}: {exc}"
        row["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
    return row


def main() -> int:
    rows = [_probe(name, info) for name, info in GENERATOR_REGISTRY.items()]
    print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
