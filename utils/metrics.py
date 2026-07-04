"""
Metrics utilities for binary, multiclass, and ordinal classification.

Includes:
- Accuracy
- Precision
- Recall
- Macro-F1
- Matthews Correlation Coefficient (MCC), normalized to [0, 1]
- Confusion matrix
- Ordinal regression: MAE, RMSE, off-by-one accuracy, quadratic/linear
  weighted kappa, Spearman rank correlation (see compute_ordinal_metrics)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
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


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Ordinal-regression-specific metrics, computed on decoded integer class
    labels (0..K-1). These complement, not replace, `compute_metrics` — a
    model can be evaluated with both nominal (accuracy/F1/MCC) and ordinal
    metrics on the same predictions. Every metric here treats getting a rank
    off by more classes as worse, which plain accuracy/F1 cannot express.

    Args:
        y_true: Ground-truth ordinal ranks (N,).
        y_pred: Predicted ordinal ranks (N,), already decoded to 0..K-1.

    Returns a dict with:
        mae: Mean Absolute Error — average |predicted rank - true rank|.
            Directly interpretable as "how many damage levels off, on
            average".
        rmse: Root Mean Squared Error — like MAE but squares errors before
            averaging, so a few large misses (e.g. off by 4) inflate it far
            more than many small ones (off by 1); a useful companion to MAE
            for spotting whether errors are small-and-frequent or
            rare-and-severe.
        off_by_one_accuracy: fraction of predictions within +/-1 rank of the
            truth. A common, easily-communicated ordinal metric ("94% of
            predictions are within one damage class").
        qwk: Quadratic Weighted Kappa (`cohen_kappa_score(weights="quadratic")`)
            — chance-corrected agreement that penalizes an error quadratically
            more the further apart the ranks are. The standard single-number
            summary for ordinal agreement; used as the training/checkpoint
            monitor metric for CORAL runs (see models/ordinal.py).
        linear_kappa: Same idea as QWK but penalizes errors linearly instead
            of quadratically — reported alongside QWK since the two disagree
            most when a model makes a few severe misses versus many mild
            ones, and reporting only one hides that distinction.
        spearman: Spearman rank correlation between true and predicted ranks.
            Measures whether predictions are *monotonically* related to the
            truth independent of the exact magnitude of errors — a model can
            score well here while still having a high MAE if its mistakes
            are consistently in the same direction.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    errors = y_pred - y_true

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    off_by_one_accuracy = float(np.mean(np.abs(errors) <= 1))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    linear_kappa = float(cohen_kappa_score(y_true, y_pred, weights="linear"))

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        # Spearman is undefined when either side is constant (e.g. every
        # prediction landed on the same class) - report NaN rather than
        # letting scipy emit a divide-by-zero warning.
        spearman = float("nan")
    else:
        spearman = float(spearmanr(y_true, y_pred).correlation)

    return {
        "mae": mae,
        "rmse": rmse,
        "off_by_one_accuracy": off_by_one_accuracy,
        "qwk": qwk,
        "linear_kappa": linear_kappa,
        "spearman": spearman,
    }
