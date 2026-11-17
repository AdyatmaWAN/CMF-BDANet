"""Bootstrap confidence intervals and paired significance tests.

Two separate things live here, and they answer different questions:

  bootstrap_ci(...)      "how precise is this arm's score?"
                         Resamples the test set with replacement and recomputes
                         the metric. Reported as a 95% CI per arm.

  paired_bootstrap(...)  "is arm A actually better than arm B?"
                         Resamples the *same* test indices for both arms on
                         every draw, so the sampling noise common to the two is
                         cancelled, and counts how often the difference changes
                         sign. This is the test the architecture comparisons use.

Both read per-sample predictions (predictions.csv), not summary numbers, so a
difference is measured on the samples the two arms actually disagree about.

Holm-Bonferroni correction is applied across the comparison family, which is
stricter than reporting raw p-values and far less conservative than plain
Bonferroni. McNemar is deliberately absent: it tests marginal homogeneity of a
2x2 table, not a difference in F-measure, and it has no multi-seed notion.

Run directly for a self-check:
    python tools/stats_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from utils.experiment import compute_f2
from utils.metrics import compute_balanced_accuracy, compute_metrics

DEFAULT_RESAMPLES = 2000


# ============================================================
# Metric adapters
# ============================================================


def multi_metric_fn(
    names: Sequence[str], num_classes: int
) -> Callable[[np.ndarray, np.ndarray], Dict[str, float]]:
    """(y_true, y_pred) -> {metric: value} for several metrics in ONE pass.

    `compute_metrics` already computes accuracy/precision/recall/macro-F1/MCC
    together, so asking for them one at a time re-runs the whole sklearn stack
    once per metric. At 2000 bootstrap draws x ~40 arms that is the difference
    between minutes and hours, which is why this exists alongside `metric_fn`.
    """
    names = list(names)
    tuple_index = {"accuracy": 0, "precision": 1, "recall": 2, "macro_f1": 3, "mcc": 4}
    wanted_tuple = {n: i for n, i in tuple_index.items() if n in names}
    want_f2 = "macro_f2" in names
    want_bacc = "balanced_accuracy" in names

    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if wanted_tuple:
            values = compute_metrics(y_true, y_pred, num_classes)
            for name, i in wanted_tuple.items():
                out[name] = float(values[i])
        if want_f2:
            out["macro_f2"] = compute_f2(y_true, y_pred, num_classes)
        if want_bacc:
            out["balanced_accuracy"] = compute_balanced_accuracy(y_true, y_pred)
        return out

    return compute


def bootstrap_ci_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: Sequence[str],
    num_classes: int,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Dict[str, Tuple[float, float, float]]:
    """Percentile bootstrap CIs for several metrics, sharing the resamples.

    Returns {metric: (point, ci_low, ci_high)}. Sharing one set of resampled
    index draws across metrics is not just faster — it also means every
    metric's interval describes the same resampled test sets.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = [n for n in names if n in SUPPORTED_METRICS]
    if len(y_true) == 0 or not names:
        return {}

    compute = multi_metric_fn(names, num_classes)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: Dict[str, List[float]] = {name: [] for name in names}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            for name, value in compute(y_true[idx], y_pred[idx]).items():
                draws[name].append(value)
        except Exception:
            continue  # a resample can be single-class; some metrics reject that

    point = compute(y_true, y_pred)
    out: Dict[str, Tuple[float, float, float]] = {}
    for name in names:
        values = np.asarray(draws[name], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            out[name] = (point.get(name, float("nan")), float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[name] = (point.get(name, float("nan")), float(lo), float(hi))
    return out


SUPPORTED_METRICS = (
    "accuracy", "precision", "recall", "macro_f1", "macro_f2",
    "balanced_accuracy", "mcc",
)


def metric_fn(name: str, num_classes: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """(y_true, y_pred) -> float, for the metrics the paper tables report."""
    if name == "macro_f2":
        return lambda yt, yp: compute_f2(yt, yp, num_classes)
    if name == "balanced_accuracy":
        return lambda yt, yp: compute_balanced_accuracy(yt, yp)
    index = {"accuracy": 0, "precision": 1, "recall": 2, "macro_f1": 3, "mcc": 4}
    if name in index:
        i = index[name]
        return lambda yt, yp: compute_metrics(yt, yp, num_classes)[i]
    raise ValueError(f"Unsupported metric: {name}")


# ============================================================
# Confidence interval for one arm
# ============================================================


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point_estimate, ci_low, ci_high).

    `y_true`/`y_pred` may be pooled across seeds — the point estimate is then
    the pooled score, and the interval reflects test-set sampling noise.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            draws[i] = score(y_true[idx], y_pred[idx])
        except Exception:
            # A resample can contain a single class, which some metrics reject.
            draws[i] = np.nan

    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return float(score(y_true, y_pred)), float("nan"), float("nan")

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(score(y_true, y_pred)), float(lo), float(hi)


# ============================================================
# Paired comparison between two arms
# ============================================================


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 12345,
) -> Dict[str, float]:
    """Two-sided paired bootstrap on (A - B).

    Both arms are scored on the *same* resampled indices each draw, so the
    shared test-set noise cancels and what is left is the difference between
    the arms. The p-value is the proportion of draws whose difference has the
    opposite sign to the observed one (doubled, two-sided), with the standard
    +1 correction so it can never be exactly zero.

    Requires `pred_a` and `pred_b` to be aligned to the same test rows — see
    `align_predictions`.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must be the same length (aligned by sample)")

    observed = score(y_true, pred_a) - score(y_true, pred_b)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            diffs.append(score(y_true[idx], pred_a[idx]) - score(y_true[idx], pred_b[idx]))
        except Exception:
            continue

    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return {"diff": float(observed), "p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    # Count draws contradicting the observed direction.
    opposite = np.sum(diffs <= 0) if observed > 0 else np.sum(diffs >= 0)
    p = min(1.0, 2.0 * (opposite + 1) / (diffs.size + 1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(observed), "p_value": float(p), "ci_low": float(lo), "ci_high": float(hi)}


def align_predictions(
    rows_a: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    rows_b: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool seed-matched (source_index, y_true, y_pred) triples from two arms,
    keeping only the samples both arms predicted, in the same order.

    Seeds are matched positionally: arm A's i-th seed against arm B's i-th
    seed. That is what makes the comparison paired across seeds as well as
    across samples.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(f"arms have different seed counts: {len(rows_a)} vs {len(rows_b)}")

    true_parts, a_parts, b_parts = [], [], []
    for (idx_a, true_a, pa), (idx_b, true_b, pb) in zip(rows_a, rows_b):
        common, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True)
        if common.size == 0:
            continue
        if not np.array_equal(np.asarray(true_a)[ia], np.asarray(true_b)[ib]):
            raise ValueError("the two arms disagree on ground truth for the same source rows")
        true_parts.append(np.asarray(true_a)[ia])
        a_parts.append(np.asarray(pa)[ia])
        b_parts.append(np.asarray(pb)[ib])

    if not true_parts:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(true_parts), np.concatenate(a_parts), np.concatenate(b_parts)


# ============================================================
# Multiple-comparison correction
# ============================================================


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni step-down correction.

    Returns (adjusted_p_values, rejected) in the *input* order. Adjusted values
    are made monotonically non-decreasing along the sorted order, which is what
    keeps the procedure coherent — without it a later, smaller adjusted value
    could contradict an earlier rejection.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return [], []

    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * p[i]
        running = max(running, value)
        adjusted[i] = min(1.0, running)

    return adjusted.tolist(), (adjusted <= alpha).tolist()


# ============================================================
# Self-check
# ============================================================


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 600
    y_true = rng.integers(0, 2, size=n)

    # A is right 90% of the time, B 60% — the test must find this.
    good = np.where(rng.random(n) < 0.90, y_true, 1 - y_true)
    poor = np.where(rng.random(n) < 0.60, y_true, 1 - y_true)
    score = metric_fn("macro_f2", num_classes=1)

    point, lo, hi = bootstrap_ci(y_true, good, score, n_resamples=300)
    assert lo <= point <= hi, (lo, point, hi)
    assert hi - lo < 0.2, (lo, hi)
    print(f"OK  bootstrap_ci: {point:.3f} [{lo:.3f}, {hi:.3f}]")

    res = paired_bootstrap(y_true, good, poor, score, n_resamples=300)
    assert res["diff"] > 0, res
    assert res["p_value"] < 0.05, res
    print(f"OK  paired_bootstrap detects a real gap: diff={res['diff']:.3f} p={res['p_value']:.4f}")

    null = paired_bootstrap(y_true, good, good.copy(), score, n_resamples=300)
    assert abs(null["diff"]) < 1e-9, null
    assert null["p_value"] > 0.05, null
    print(f"OK  paired_bootstrap finds nothing between identical arms: p={null['p_value']:.3f}")

    adj, rejected = holm_bonferroni([0.001, 0.04, 0.3])
    assert adj[0] < adj[1] < adj[2], adj
    assert rejected[0] and not rejected[2], (adj, rejected)
    assert holm_bonferroni([0.05])[0] == [0.05]
    print(f"OK  holm_bonferroni: {[round(a, 4) for a in adj]} rejected={rejected}")

    idx = np.arange(n)
    t, a, b = align_predictions([(idx, y_true, good)], [(idx[::-1], y_true[::-1], poor[::-1])])
    assert np.array_equal(t, y_true) and np.array_equal(a, good) and np.array_equal(b, poor)
    print("OK  align_predictions matches on source_index regardless of row order")

    print("\nstats_compare self-check passed")


if __name__ == "__main__":
    _self_check()
