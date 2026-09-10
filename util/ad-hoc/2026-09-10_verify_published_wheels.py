#!/usr/bin/env python3
"""Verify the decision-11 release-train wheels AS PUBLISHED, not as checked out.

Project:     Juniper
Sub-Project: juniper-ml
Application: ad-hoc release verification
Author:      Paul Calnon
Created:     2026-09-10
Version:     0.1.0
License:     MIT License
Status:      single-use (decision-11 release train, Wave 1 post-publish check)

Why this exists
---------------
The SemVer ruling's last step is to verify each **published** wheel in a clean venv rather
than the working checkout. A checkout is not a deployment: the repo can be correct while the
wheel on PyPI is stale -- which is exactly what juniper-model-core 0.3.1 was, carrying a
docstring the repo had already fixed.

Importability is not behaviour, so this exercises ``derive_full_split`` on real arrays,
including the case the 0.3.0 changelog calls "the whole difficulty": juniper-data's
``equities`` / ``equities_seq`` generators wrote ``*_full`` ENTITY-major while their
partitions are SPLIT-major, so a plain concatenation holds the same rows in a different
permutation -- and walk-forward cross-validation slices by ROW INDEX, so the naive version
would silently redistribute windows across folds.

Run against a venv that has the published wheels installed:

    python3 -m venv /path/to/venv
    /path/to/venv/bin/pip install juniper-data-client==0.5.0 juniper-recurrence-model==0.3.0
    /path/to/venv/bin/python util/ad-hoc/2026-09-10_verify_published_wheels.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import sys

import numpy as np


def check_npz_splits() -> None:
    from juniper_data_client.constants import NPZ_SPLITS

    print(f"  NPZ_SPLITS = {NPZ_SPLITS}")
    assert "full" not in NPZ_SPLITS, f"'full' is still in NPZ_SPLITS: {NPZ_SPLITS}"
    assert NPZ_SPLITS == ("train", "val", "test"), NPZ_SPLITS
    print("  OK: three-way, no 'full'")


def check_derive_full_split() -> None:
    import juniper_recurrence_model as m
    from juniper_recurrence_model.data import derive_full_split

    print(f"  juniper_recurrence_model.__version__ = {m.__version__}")
    assert "derive_full_split" in m.data.__all__, "derive_full_split not exported"

    # 1. No ticker_code -> plain concatenation, in train | val | test order.
    plain = {
        "X_train": np.array([[1.0], [2.0]]),
        "X_val": np.array([[3.0]]),
        "X_test": np.array([[4.0]]),
    }
    got = derive_full_split(dict(plain))["X_full"].ravel().tolist()
    print(f"  no-ticker concat      -> {got}")
    assert got == [1.0, 2.0, 3.0, 4.0], got

    # 2. Multi-entity -> the stable ticker_code sort restores the producer's ENTITY-major
    #    order. A split-major concatenation would give [10, 20, 11, 21, 12, 22].
    panel = {
        "X_train": np.array([[10.0], [20.0]]),
        "ticker_code_train": np.array([0, 1]),
        "X_val": np.array([[11.0], [21.0]]),
        "ticker_code_val": np.array([0, 1]),
        "X_test": np.array([[12.0], [22.0]]),
        "ticker_code_test": np.array([0, 1]),
    }
    got = derive_full_split(dict(panel))["X_full"].ravel().tolist()
    print(f"  entity-major restored -> {got}")
    assert got == [10.0, 11.0, 12.0, 20.0, 21.0, 22.0], got

    # 3. A producer-supplied *_full is never overwritten (legacy artifacts stay byte-identical).
    legacy = dict(plain)
    legacy["X_full"] = np.array([[99.0]])
    got = derive_full_split(dict(legacy))["X_full"].ravel().tolist()
    print(f"  legacy *_full kept    -> {got}")
    assert got == [99.0], got

    print("  OK: order, panel restoration and legacy passthrough all correct")


def main() -> int:
    print("juniper-data-client (published wheel):")
    check_npz_splits()
    print("juniper-recurrence-model (published wheel):")
    check_derive_full_split()
    print("\nALL PUBLISHED-WHEEL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
