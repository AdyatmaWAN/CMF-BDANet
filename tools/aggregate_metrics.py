"""
Aggregate all metrics.csv files under a results directory into a single
table, in two stages:

1. Per-ablation-config best: collapse the hyperparameter grid (optimizer/lr/
   batch_size for the NN models, kernel/decision_function/C/gamma for the
   SVM baselines) down to the single best HPO combo for every
   (model, scenario, dataset, ablation-config) group. This answers "what's
   the best result for this model configuration/ablation?".
2. Best overall: within stage 1's output, collapse further across every
   model to the single best row per scenario (+ dataset). This answers
   "which model/config wins outright?".

Every metrics.csv already carries its own metadata columns (scenario,
model, dataset, dsm_mode_tag, ...), so grouping just needs to know which
columns are hyperparameters (to collapse over) vs. bookkeeping (to ignore) —
everything else is treated as part of the ablation identity. This means new
models/columns need no code changes here, only column-name additions to the
two sets below if they introduce a genuinely new hyperparameter.

Usage:
    python tools/aggregate_metrics.py [--results ROOT] [--best-metric f1|f2]

Writes `<results>/aggregated_metrics.csv` (+ .xlsx) and
`<results>/best_overall.csv` (+ .xlsx) by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Hyperparameter-tuning columns: collapsed within each ablation-config group.
HPO_COLUMNS = {
    "optimizer", "learning_rate", "batch_size",
    "kernel", "decision_function", "dec_func", "C", "gamma",
}

# Per-run metrics/bookkeeping columns: never part of the grouping identity.
NON_GROUP_COLUMNS = {
    "accuracy", "precision", "recall", "macro_f1", "macro_f2", "mcc",
    "cm00", "cm01", "cm10", "cm11", "confusion_matrix_json",
    "best_val_metric", "best_val_epoch", "monitor_metric",
    "train_time_sec", "num_test_samples",
    "result_dir", "predictions_csv", "selected_metric",
    # Ordinal-only metrics (see utils/metrics.py::compute_ordinal_metrics) -
    # present only on scenario-5 rows, always excluded from grouping either way.
    "mae", "rmse", "off_by_one_accuracy", "qwk", "linear_kappa", "spearman",
}


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _score_column_name(best_metric: str) -> str:
    return "macro_f1" if best_metric == "f1" else "macro_f2"


def _best_metric_series(df: pd.DataFrame, best_metric: str) -> pd.Series:
    if best_metric == "f1":
        if "macro_f1" not in df.columns:
            raise ValueError("Input data does not contain macro_f1")
        return _coerce_numeric(df["macro_f1"])
    if best_metric == "f2":
        if "macro_f2" in df.columns:
            return _coerce_numeric(df["macro_f2"])
        precision = _coerce_numeric(df.get("precision", pd.Series(dtype=float)))
        recall = _coerce_numeric(df.get("recall", pd.Series(dtype=float)))
        denom = (4.0 * precision + recall).replace(0, pd.NA)
        return (5.0 * precision * recall) / denom
    raise ValueError(f"Unsupported best_metric: {best_metric}")


def _ablation_group_columns(df: pd.DataFrame, score_col: str) -> list[str]:
    excluded = HPO_COLUMNS | NON_GROUP_COLUMNS | {score_col}
    return [c for c in df.columns if c not in excluded]


def load_all_metrics(results_root: Path) -> pd.DataFrame:
    """Read every metrics.csv under results_root into one flat table."""
    csv_paths = sorted(results_root.rglob("metrics.csv"))
    if not csv_paths:
        print(f"No metrics.csv files found under {results_root}")
        return pd.DataFrame()

    rows = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue
        rows.extend(df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def collapse_best(df: pd.DataFrame, group_cols: list[str], score_col: str) -> pd.DataFrame:
    """Keep one row per group_cols combo: the row with the highest score_col."""
    if df.empty:
        return df
    if not group_cols:
        scored = df.dropna(subset=[score_col])
        return scored.sort_values(score_col, ascending=False).iloc[[0]] if not scored.empty else df.iloc[0:0]

    ranked_rows = []
    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        scored = group.dropna(subset=[score_col])
        if scored.empty:
            ranked_rows.append(group.iloc[[0]])
            continue
        ranked_rows.append(scored.sort_values(score_col, ascending=False).iloc[[0]])
    return pd.concat(ranked_rows, ignore_index=True)


def aggregate_metrics(results_root: Path, best_metric: str = "f1") -> pd.DataFrame:
    """Stage 1: best HPO combo per (model, scenario, dataset, ablation-config)."""
    agg = load_all_metrics(results_root.resolve())
    if agg.empty:
        return agg

    score_col = _score_column_name(best_metric)
    agg[score_col] = _best_metric_series(agg, best_metric)
    group_cols = _ablation_group_columns(agg, score_col)

    result = collapse_best(agg, group_cols, score_col)
    result["selected_metric"] = score_col
    return _reorder(result, score_col)


def select_best_overall(per_ablation_best: pd.DataFrame, best_metric: str = "f1") -> pd.DataFrame:
    """Stage 2: collapse the stage-1 table further across every model, down
    to the single best row per scenario (+ dataset).
    """
    if per_ablation_best.empty:
        return per_ablation_best

    score_col = _score_column_name(best_metric)
    group_cols = [c for c in ("scenario", "dataset") if c in per_ablation_best.columns]
    result = collapse_best(per_ablation_best, group_cols, score_col)
    return _reorder(result, score_col)


def _reorder(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    front_cols = [
        c for c in (
            "scenario", "model", "dataset", "dsm_mode_tag", "variant_tag",
            "residual", "fusion", "epochs", "num_classes",
            "dsm_post_concat_with_rgb", "use_four_stream",
            "optimizer", "learning_rate", "batch_size",
            "kernel", "decision_function", "C", "gamma",
            "selected_metric", score_col, "result_dir",
        )
        if c in df.columns
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df.loc[:, front_cols + other_cols]


def _write(df: pd.DataFrame, out_csv: Path, out_xlsx: Path) -> None:
    df.to_csv(out_csv, index=False)
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="sheet1", index=False)
        print(f"Wrote {out_csv} and {out_xlsx}")
    except ModuleNotFoundError:
        print(f"Wrote {out_csv} (skipped .xlsx: no openpyxl/xlsxwriter installed)")
    except Exception as e:
        print(f"Wrote {out_csv} (failed to write {out_xlsx}: {e})")


def aggregate_and_write(results_root: Path, best_metric: str = "f1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both aggregation stages over everything under `results_root` and
    write their CSV/XLSX outputs directly under it (`aggregated_metrics.*`,
    `best_overall.*`). This is the single entrypoint both the CLI below and
    every `train_*.py` script's automatic post-run aggregation call into —
    point it at `results/scenario_N/` to scope it to one scenario across
    every model, or at `results/` for an all-scenarios view.

    Returns (per_ablation_best, best_overall); both empty if no metrics.csv
    was found under results_root.
    """
    per_ablation_best = aggregate_metrics(results_root, best_metric=best_metric)
    if per_ablation_best.empty:
        return per_ablation_best, per_ablation_best

    _write(
        per_ablation_best,
        results_root / "aggregated_metrics.csv",
        results_root / "aggregated_metrics.xlsx",
    )
    best_overall = select_best_overall(per_ablation_best, best_metric=best_metric)
    _write(
        best_overall,
        results_root / "best_overall.csv",
        results_root / "best_overall.xlsx",
    )
    return per_ablation_best, best_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results", help="Path to results root")
    parser.add_argument(
        "--best-metric", type=str, default="f1", choices=("f1", "f2"),
        help="Metric used to rank hyperparameter combos and, in stage 2, models.",
    )
    args = parser.parse_args()

    results_root = Path(args.results)
    if not results_root.exists():
        print(f"Results path does not exist: {results_root}")
        return

    per_ablation_best, best_overall = aggregate_and_write(results_root, best_metric=args.best_metric)
    if per_ablation_best.empty:
        print("No data aggregated.")
        return

    print(f"Aggregated {len(per_ablation_best)} ablation-config rows; {len(best_overall)} best-overall rows.")


if __name__ == "__main__":
    main()
