#!/usr/bin/env python3
"""Stage 2 of the §12.4 rank-2 validation: fit CasCor on the artifacts stage 1 wrote.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc analysis
Author:      Paul Calnon
License:     MIT

Consumes the NPZ files written by 2026-09-10_rank2_generate_artifacts.py and trains
``CascadeCorrelationNetwork`` on each, which is the "train" half of §12.4's
generate -> stage -> train -> render requirement for a rank-2 seed
(notes/JUNIPER_2026-09-02_JUNIPER-CANOPY_SELECTION-REACHABILITY-DESIGN.md).

Two deliberate choices:

* **Both ``max_epochs`` and ``output_epochs`` are set, to the same value.** juniper-ml's
  AGENTS.md carries this as a resident hazard: the service applies ``max_epochs`` only to
  the INITIAL output pass and reads ``output_epochs`` (defaulting to 10000) for every later
  pass, while the direct CLI aliases the two. Setting only one silently produces a much
  longer run than the config appears to ask for.
* **Threads are capped and the device forced to CPU.** Other sessions run experiments on
  this host, and cascor's first output pass bursts through libgomp under torch. This probe
  is a trainability check, not a benchmark, so it has no business taking the machine.

Run:  conda run -n JuniperCascor1 python util/ad-hoc/2026-09-10_rank2_cascor_fit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Must precede the torch import to bind.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

CASCOR_SRC = Path("/home/pcalnon/Development/python/Juniper/juniper-cascor/src")
if str(CASCOR_SRC) not in sys.path:
    sys.path.insert(0, str(CASCOR_SRC))

from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork  # noqa: E402

ART_DIR = Path("/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/febb940d-6152-433e-a6e6-6534f75e4907/scratchpad/rank2_artifacts")

# Bounded on purpose: enough cascade growth to prove the algorithm engages with the data,
# nowhere near enough to be a performance claim.
#
# ``max_iterations`` was 2 in the first run of this probe, and that was too few to be a
# verdict: checkerboard is an XOR-like problem whose whole point is that it needs several
# cascade units, so it came back at chance (top-1 0.515) with an essentially flat loss.
# A cap that starves the algorithm measures the cap, not the dataset.
BOUNDED_EPOCHS = 60
CONFIG: dict[str, Any] = {
    "max_iterations": 8,  # cascade growth iterations
    "output_epochs": BOUNDED_EPOCHS,  # later output passes -- see the hazard note above
    "candidate_epochs": 40,
    "candidate_pool_size": 4,
    "generate_plots": False,
    "random_seed": 0,
}

# A trainability probe must not be dominated by data volume; subsample large artifacts.
MAX_ROWS = 4000


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float | None:
    """Top-1 agreement when the target is one-hot; None for a regression target."""
    if target.ndim != 2 or target.shape[1] < 2:
        return None
    return float((pred.argmax(dim=1) == target.argmax(dim=1)).float().mean().item())


def _fit_one(npz_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"case": npz_path.stem}
    arrays = np.load(npz_path)
    keys = sorted(arrays.files)
    row["npz_keys"] = keys

    x_train = np.asarray(arrays["X_train"], dtype=np.float32)
    y_train = np.asarray(arrays["y_train"], dtype=np.float32)
    row["X_train_shape"] = tuple(int(d) for d in x_train.shape)
    row["y_train_shape"] = tuple(int(d) for d in y_train.shape)

    if x_train.ndim != 2:
        row["verdict"] = f"NOT RANK-2 (ndim={x_train.ndim}) -- cascor's data_provider refuses this"
        return row

    if len(x_train) > MAX_ROWS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x_train), MAX_ROWS, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
        row["subsampled_to"] = MAX_ROWS

    x_val = arrays["X_val"].astype(np.float32) if "X_val" in arrays.files else None
    y_val = arrays["y_val"].astype(np.float32) if "y_val" in arrays.files else None
    if x_val is not None and len(x_val) > MAX_ROWS:
        x_val, y_val = x_val[:MAX_ROWS], y_val[:MAX_ROWS]

    # input_size / output_size default to the spiral problem's 2 and must be taken from the
    # data. Omitting them is why the first run of this probe reported equities as
    # "FIT FAILED: Expected 2 input features, got 16" -- a statement about the config, not
    # about the dataset, and one that would have wrongly kept equities on G10's exclusion list.
    net = CascadeCorrelationNetwork(
        input_size=int(x_train.shape[1]),
        output_size=int(y_train.shape[1]) if y_train.ndim == 2 else 1,
        **CONFIG,
    )
    row["input_size"] = int(x_train.shape[1])
    row["output_size"] = int(y_train.shape[1]) if y_train.ndim == 2 else 1
    try:
        t0 = time.monotonic()
        history = net.fit(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
            x_val=torch.from_numpy(x_val) if x_val is not None else None,
            y_val=torch.from_numpy(y_val) if y_val is not None else None,
            max_epochs=BOUNDED_EPOCHS,
            max_iterations=CONFIG["max_iterations"],
        )
        row["fit_s"] = round(time.monotonic() - t0, 1)
    except Exception as exc:  # noqa: BLE001 — a refusal to fit IS the measurement
        row["verdict"] = f"FIT FAILED: {type(exc).__name__}: {exc}"
        return row

    if isinstance(history, dict):
        # Report every series the run produced, not just the three names guessed up front --
        # the first run of this probe read only "train_loss" and so could not tell a flat
        # loss apart from a cascade that never grew.
        row["history_keys"] = sorted(history)
        for key in ("train_loss", "loss", "val_loss", "train_accuracy", "val_accuracy"):
            series = history.get(key)
            if isinstance(series, list) and series:
                row[f"{key}_first"] = round(float(series[0]), 6)
                row[f"{key}_last"] = round(float(series[-1]), 6)
                row[f"{key}_n"] = len(series)

    try:
        with torch.no_grad():
            pred = net.forward(torch.from_numpy(x_train))
        acc = _accuracy(pred, torch.from_numpy(y_train))
        if acc is not None:
            row["train_top1"] = round(acc, 4)
    except Exception as exc:  # noqa: BLE001 — scoring is a bonus, not the verdict
        row["score_note"] = f"{type(exc).__name__}: {exc}"

    # ``hidden_units`` is a LIST on the network (cascade_correlation.py:740). The first run of
    # this probe guessed ``num_hidden_units`` / ``n_hidden``, got None for every case, and so
    # could not distinguish "trained and grew" from "trained and never recruited a unit" --
    # which is exactly the question a flat loss raises.
    units = getattr(net, "hidden_units", None)
    row["hidden_units"] = len(units) if isinstance(units, list) else None
    row["verdict"] = "TRAINED"
    return row


def main() -> int:
    if not ART_DIR.exists():
        print(f"no artifacts at {ART_DIR}; run stage 1 first", file=sys.stderr)
        return 2
    rows = []
    for npz_path in sorted(ART_DIR.glob("*.npz")):
        print(f"\n=== {npz_path.stem} ===", flush=True)
        row = _fit_one(npz_path)
        rows.append(row)
        for key, value in row.items():
            if key != "npz_keys":
                print(f"  {key:20s}: {value}")
    (ART_DIR / "stage2.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {ART_DIR}/stage2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
