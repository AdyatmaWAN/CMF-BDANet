"""
Metrics utilities for binary and multiclass classification.

Includes:
- Accuracy
- Precision
- Recall
- Macro-F1
- Matthews Correlation Coefficient (MCC), normalized to [0, 1]
- Confusion matrix
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Compute metrics for binary or multiclass classification.

    Args:
        y_true: Ground-truth labels (N,).
        y_pred: Predicted labels (N,), already discretized (0..K-1).
        num_classes:
            1  → binary
            >1 → multiclass (macro-averaged precision/recall/F1)

    Returns:
        (accuracy, precision, recall, macro_f1, mcc_normalized, confusion_matrix)
    """
    acc = accuracy_score(y_true, y_pred)

    if num_classes == 1:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
    else:
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(y_true, y_pred, average="macro")

    mcc_raw = matthews_corrcoef(y_true, y_pred)
    mcc_norm = (mcc_raw + 1.0) / 2.0  # normalize from [-1,1] → [0,1]

    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, mcc_norm, cm
