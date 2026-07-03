"""
Utility to aggregate all metrics.csv files under the `results` directory
and write a single CSV/Excel file. It extracts semantic columns from the
results folder structure and also splits optimizer_lr_bs folder names into
three separate columns: optimizer, learning_rate and batch_size.

The aggregator can also collapse multiple optimizer / learning-rate /
batch-size runs down to the single best row per experiment configuration.
Choose the ranking metric with `--best-metric f1` or `--best-metric f2`.

Usage:
    python tools/aggregate_metrics.py [--results ROOT]

The script will write `results/aggregated_metrics.csv` and
`results/aggregated_metrics.xlsx` (Excel) by default.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


OPT_LR_BS_RE = re.compile(r"(?P<optimizer>[^_]+)_lr(?P<lr>[-+.0-9eE]+)_bs(?P<bs>\d+)")


GROUP_COLUMNS = [
    "scenario",
    "model",
    "dataset",
    "residual",
    "fusion",
    "epochs",
    "num_classes",
    "dsm_mode_tag",
    "dsm_post_concat_with_rgb",
    "use_four_stream",
    "variant_tag",
]


def parse_metadata_from_path(p: Path, results_root: Path) -> dict:
    """Extract semantic metadata from a metrics.csv file path.

    The function looks for common folder name patterns created by the
    training script, e.g.:
      results/scenario_1/SNN/dataset_16/dsm_density/residual_True/fusion_concat/Adam_lr0.0001_bs256/metrics.csv

    It returns a dict with parsed fields plus the full result_dir.
    """
    meta = {}
    rel = p.relative_to(results_root)
    parts = list(rel.parent.parts)  # exclude the file name

    # Find scenario
    for part in parts:
        if part.startswith("scenario_"):
            try:
                meta["scenario"] = int(part.split("_")[1])
            except Exception:
                meta["scenario"] = part
            break

    # Model (SNN / MMF)
    for m in ("SNN", "MMF"):
        if m in parts:
            meta["model"] = m
            break

    # dataset e.g. dataset_16
    for part in parts:
        if part.startswith("dataset_"):
            meta["dataset"] = part
            break

    # dsm mode tag (match known patterns used in the project)
    possible_dsm_tags = (
        "dsm_density_uncertainty",
        "dsm_density",
        "dsm_uncertainty",
        "dsm_only",
    )
    for part in parts:
        if part in possible_dsm_tags:
            meta["dsm_mode_tag"] = part
            break

    # variant tag for MMF (3stream_concat, 3stream_no_concat, 4stream)
    for part in parts:
        if part in ("3stream_concat", "3stream_no_concat", "4stream"):
            meta["variant_tag"] = part
            break

    # residual_True / residual_False
    for part in parts:
        if part.startswith("residual_"):
            meta["residual"] = part.split("residual_")[1]
            break

    # fusion_concat / fusion_mcmaf
    for part in parts:
        if part.startswith("fusion_"):
            meta["fusion"] = part.split("fusion_")[1]
            break

    # optimizer_lr{lr}_bs{bs} folder
    for part in parts:
        m = OPT_LR_BS_RE.match(part)
        if m:
            meta["optimizer"] = m.group("optimizer")
            try:
                meta["learning_rate"] = float(m.group("lr"))
            except Exception:
                meta["learning_rate"] = m.group("lr")
            meta["batch_size"] = int(m.group("bs"))
            break

    meta["result_dir"] = str(p.parent)
    return meta


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _score_column_name(best_metric: str) -> str:
    return "macro_f1" if best_metric == "f1" else "macro_f2"


def _compute_macro_f2(df: pd.DataFrame) -> pd.Series:
    """Compute F2 from precision and recall.

    This uses the standard F-beta formula with beta=2.
    If either precision or recall is missing, the result is NaN.
    """
    precision = _coerce_numeric(df.get("precision", pd.Series(dtype=float)))
    recall = _coerce_numeric(df.get("recall", pd.Series(dtype=float)))
    denom = 4.0 * precision + recall
    with pd.option_context("mode.use_inf_as_na", True):
        return (5.0 * precision * recall) / denom.replace(0, pd.NA)


def _best_metric_series(df: pd.DataFrame, best_metric: str) -> pd.Series:
    if best_metric == "f1":
        if "macro_f1" not in df.columns:
            raise ValueError("Input data does not contain macro_f1")
        return _coerce_numeric(df["macro_f1"])
    if best_metric == "f2":
        return _compute_macro_f2(df)
    raise ValueError(f"Unsupported best_metric: {best_metric}")


def aggregate_metrics(results_root: Path, best_metric: str = "f1") -> pd.DataFrame:
    results_root = results_root.resolve()
    csv_paths = list(results_root.rglob("metrics.csv"))
    rows = []
    if not csv_paths:
        print(f"No metrics.csv files found under {results_root}")
        return pd.DataFrame()

    for p in sorted(csv_paths):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        meta = parse_metadata_from_path(p, results_root)
        # For each row in df (usually 1) attach metadata
        for _, row in df.iterrows():
            combined = row.to_dict()
            # don't overwrite existing metric columns unless missing
            for k, v in meta.items():
                if k not in combined or combined.get(k) in (None, ""):
                    combined[k] = v
            rows.append(combined)

    if not rows:
        return pd.DataFrame()
    agg = pd.DataFrame(rows)

    score_col = _score_column_name(best_metric)
    agg[score_col] = _best_metric_series(agg, best_metric)

    group_cols = [c for c in GROUP_COLUMNS if c in agg.columns]

    # Keep one row per experiment configuration by selecting the row with the
    # best score across optimizer / learning-rate / batch-size combinations.
    # Drop rows with missing score before ranking, but retain them if there is
    # no comparable row in that group.
    ranked_rows = []
    if group_cols:
        for _, group in agg.groupby(group_cols, dropna=False, sort=False):
            scored = group.dropna(subset=[score_col]).copy()
            if scored.empty:
                ranked_rows.append(group.iloc[[0]].copy())
                continue
            scored = scored.sort_values(
                by=[score_col, "best_val_f1" if "best_val_f1" in scored.columns else score_col],
                ascending=[False, False],
                kind="mergesort",
            )
            ranked_rows.append(scored.iloc[[0]].copy())
        agg = pd.concat(ranked_rows, ignore_index=True)

    agg["selected_metric"] = score_col

    # Reorder columns to put common metadata first if present
    front_cols = [
        c
        for c in (
            "scenario",
            "model",
            "dataset",
            "dsm_mode_tag",
            "variant_tag",
            "residual",
            "fusion",
            "epochs",
            "num_classes",
            "dsm_post_concat_with_rgb",
            "use_four_stream",
            "optimizer",
            "learning_rate",
            "batch_size",
            "selected_metric",
            score_col,
            "result_dir",
        )
        if c in agg.columns
    ]
    other_cols = [c for c in agg.columns if c not in front_cols]
    agg = agg.loc[:, front_cols + other_cols]
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results", help="Path to results root")
    parser.add_argument("--out-csv", type=str, default=None, help="CSV output path (optional)")
    parser.add_argument("--out-xlsx", type=str, default=None, help="Excel output path (optional)")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="f1",
        choices=("f1", "f2"),
        help="Metric used to pick the best optimizer/lr/bs combination within each experiment configuration.",
    )
    args = parser.parse_args()

    results_root = Path(args.results)
    if not results_root.exists():
        print(f"Results path does not exist: {results_root}")
        return

    agg = aggregate_metrics(results_root, best_metric=args.best_metric)
    if agg.empty:
        print("No data aggregated.")
        return

    out_csv = Path(args.out_csv) if args.out_csv else results_root / "aggregated_metrics.csv"
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else results_root / "aggregated_metrics.xlsx"

    agg.to_csv(out_csv, index=False)
    # Try writing Excel; if the environment lacks Excel engines (openpyxl/xlsxwriter)
    # gracefully skip Excel output and notify the user.
    wrote_xlsx = False
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            agg.to_excel(w, sheet_name="aggregated", index=False)
        wrote_xlsx = True
    except ModuleNotFoundError:
        try:
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
                agg.to_excel(w, sheet_name="aggregated", index=False)
            wrote_xlsx = True
        except ModuleNotFoundError:
            print(
                "Warning: no Excel writer engine available (openpyxl/xlsxwriter). Skipping .xlsx output."
            )
    except Exception as e:
        print(f"Failed to write Excel file: {e}")

    print(f"Wrote aggregated CSV: {out_csv}")
    if wrote_xlsx:
        print(f"Wrote aggregated Excel: {out_xlsx}")
    else:
        print("Skipped .xlsx output (no Excel writer installed). You can install openpyxl or xlsxwriter to enable Excel export.")
    print(f"Aggregated {len(agg)} rows from {len(list(results_root.rglob('metrics.csv')))} files.")


if __name__ == "__main__":
    main()



