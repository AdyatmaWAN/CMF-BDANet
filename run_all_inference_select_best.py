"""
Select best inference runs per parameter combination.

For each unique combination of: dataset, model, residual, fusion, epochs,
num_classes, dsm_mode_tag, scenario, dsm_post_concat_with_rgb, use_four_stream
the script selects the run (different optimizer / learning_rate / batch_size)
that maximizes either macro_f1 or macro_f2. Two sets of outputs are produced:
 - best_by_f1
 - best_by_f2

Outputs are written as CSV and HTML in an output folder and additionally
split into subfolders per scenario and per model (SNN / MMF).

Usage:
    python run_all_inference_select_best.py \
        --input inference_results/inference_summary.csv \
        --output inference_results/best

The script will create:
  <output>/best_by_f1.csv, best_by_f1.html
  <output>/best_by_f2.csv, best_by_f2.html
  <output>/scenario_<n>/<model>/best_by_f1.csv/html and best_by_f2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT = Path("inference_results") / "inference_summary.csv"
DEFAULT_OUTPUT = Path("inference_results") / "best"

EXPORT_COLUMNS = [
    "model",
    "residual",
    "fusion",
    "dsm_mode_tag",
    "dsm_post_concat_with_rgb",
    "optimizer",
    "learning_rate",
    "batch_size",
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "best_val_f1",
    "mcc",
    "macro_f2",
    "result_dir",
    "confusion_matrix_json",
    "num_test_samples",
    "checkpoint_path",
    "inference_dir",
    "predictions_csv",
    "status",
    "source_metrics_path",
    "inference_metrics_csv",
]


def validate_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def select_best(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    # Within each group choose row with highest metric; break ties with mcc if available
    def pick_best(g: pd.DataFrame) -> pd.Series:
        if metric not in g.columns:
            # if metric missing, return first
            return g.iloc[0]
        # drop rows with NaN metric to avoid idxmax errors
        sub = g.dropna(subset=[metric])
        if sub.empty:
            # fallback to original group
            sub = g
        if "mcc" in sub.columns:
            sub = sub.sort_values([metric, "mcc"], ascending=[False, False])
        else:
            sub = sub.sort_values(metric, ascending=False)
        return sub.iloc[0]

    picked = df.groupby(group_cols, dropna=False, as_index=False).apply(pick_best)
    # groupby + apply produces a hierarchical index in older pandas; normalize it
    if isinstance(picked, pd.DataFrame):
        picked = picked.reset_index(drop=True)
    return picked


def write_outputs(df: pd.DataFrame, output_root: Path, prefix: str):
    ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True)
    ensure_dir(output_root)
    csv_path = output_root / f"{prefix}.csv"
    html_path = output_root / f"{prefix}.html"
    df.to_csv(csv_path, index=False)
    df.to_html(html_path, index=False)
    print(f"Wrote {prefix}: {csv_path}")


def export_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df.loc[:, cols].copy()


def main():
    parser = argparse.ArgumentParser(description="Select best inference runs by F1 or F2")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Global inference summary CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder for best selections")
    args = parser.parse_args()

    input_csv = args.input
    out_root = args.output
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        print("Input CSV is empty; nothing to do.")
        return

    # Required grouping columns
    group_cols = [
        "dataset",
        "model",
        "residual",
        "fusion",
        "epochs",
        "num_classes",
        "dsm_mode_tag",
        "scenario",
        "dsm_post_concat_with_rgb",
        "use_four_stream",
    ]
    # Validate presence but don't fail if some are missing; adjust accordingly
    present_group_cols = [c for c in group_cols if c in df.columns]
    if not present_group_cols:
        raise ValueError("None of the expected grouping columns are present in the input CSV.")

    # Produce both best-by-F1 and best-by-F2
    for metric, prefix in [("macro_f1", "best_by_f1"), ("macro_f2", "best_by_f2")]:
        try:
            best_df = select_best(df, present_group_cols, metric)
        except Exception as e:
            print(f"Failed to select best by {metric}: {e}")
            continue

        # Top-level outputs
        write_outputs(export_view(best_df), out_root, prefix)

        # Additionally, if MMF model is present and use_four_stream is a column,
        # split MMF rows into separate files for each use_four_stream value
        if "model" in best_df.columns and "use_four_stream" in best_df.columns:
            mmf_df = best_df[best_df["model"] == "MMF"]
            if not mmf_df.empty:
                for val in sorted(mmf_df["use_four_stream"].dropna().unique(), key=lambda x: str(x)):
                    mask = mmf_df["use_four_stream"] == val
                    df_mmf_split = mmf_df[mask]
                    # Create top-level files named like: best_by_f1_MMF_use_four_stream_True.csv
                    write_outputs(export_view(df_mmf_split), out_root, f"{prefix}_MMF_use_four_stream_{str(val)}")

        # Split per scenario and per model
        scenarios = sorted(best_df["scenario"].unique()) if "scenario" in best_df.columns else [None]
        models = sorted(best_df["model"].unique()) if "model" in best_df.columns else [None]

        for scen in scenarios:
            scen_mask = best_df["scenario"] == scen if scen is not None else pd.Series([True] * len(best_df))
            df_s = best_df[scen_mask]
            scen_dir = out_root / (f"scenario_{scen}" if scen is not None else "scenario_unknown")
            ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True)
            ensure_dir(scen_dir)
            write_outputs(export_view(df_s), scen_dir, prefix)

            for model in models:
                m = model
                df_sm = df_s[df_s["model"] == m]
                model_dir = scen_dir / str(m)
                ensure_dir(model_dir)
                # If this is the MMF model and use_four_stream exists, split into separate files
                if m == "MMF" and "use_four_stream" in df_sm.columns:
                    for val in sorted(df_sm["use_four_stream"].dropna().unique(), key=lambda x: str(x)):
                        df_sm_split = df_sm[df_sm["use_four_stream"] == val]
                        write_outputs(export_view(df_sm_split), model_dir, f"{prefix}_use_four_stream_{str(val)}")
                else:
                    write_outputs(export_view(df_sm), model_dir, prefix)

    print("Done selecting best runs.")


if __name__ == "__main__":
    main()

