"""
Label processing utilities for different experimental scenarios.

Scenario 1:
    Binary classification: class 0 vs class 4.
    Keep only labels {0, 4}, map:
        0 → 0
        4 → 1

Scenario 2:
    Binary classification: first four classes vs last class.
    Group:
        {0, 1, 2, 3} → 0
        {4}          → 1

Scenario 3:
    Multiclass classification with all 5 classes.
    Classes:
        0, 1, 2, 3, 4  (identity mapping)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def prepare_split_for_scenario(
    X: np.ndarray,
    Y: np.ndarray,
    scenario: int,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[int, int]]:
    """
    Process a single split (train/val/test) for a given scenario.

    Args:
        X: Data array (N, H, W, C).
        Y: Label array (N,) with original classes in {0,1,2,3,4}.
        scenario:
            1 → binary (0 vs 4), filter to only these classes.
            2 → binary (0–3 vs 4).
            3 → multiclass (0–4).

    Returns:
        X_proc: Possibly filtered data.
        Y_proc: Reindexed labels.
        num_classes: 1 for binary, or 5 for multiclass.
        mapping: dict {old_label → new_label}.
    """
    if scenario == 1:
        # Keep only classes 0 and 4
        mask = np.isin(Y, [0, 4])
        X_proc = X[mask]
        Y_proc_raw = Y[mask]
        mapping = {0: 0, 4: 1}
        Y_proc = np.array([mapping[int(y)] for y in Y_proc_raw], dtype=np.int64)
        num_classes = 1

    elif scenario == 2:
        # Group 0–3 vs 4
        mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
        X_proc = X
        Y_proc = np.array([mapping[int(y)] for y in Y], dtype=np.int64)
        num_classes = 1

    elif scenario == 3:
        # All 5 classes intact
        mapping = {int(c): int(c) for c in np.unique(Y)}
        X_proc = X
        Y_proc = Y.astype(np.int64)
        num_classes = len(mapping)  # should be 5

    else:
        raise ValueError("Scenario must be 1, 2, or 3.")

    return X_proc, Y_proc, num_classes, mapping
