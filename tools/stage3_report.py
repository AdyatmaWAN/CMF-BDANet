"""Stage 3 — the paper tables.

Two aggregates, both written per scenario (one file each) and combined (one
file with a `scenario` column):

  Aggregate A, table_main        cross-model comparison, 8 rows per scenario:
                                 the 4 FCSNN cell winners + BDDNet + Fujita +
                                 Moya + Hajeb. All at bf=32. The capacity
                                 control is deliberately NOT here — it is not a
                                 competing model.

  Aggregate B, table_ablation    FCSNN only, 17 rows per scenario: all 16
                                 ablation configs plus the capacity control,
                                 flagged control=True, with the Stage-1
                                 validation score that drove selection.

Plus table_hpo.csv (Stage-0 winners, for the reproducibility appendix) and
significance.csv (Holm-corrected paired bootstrap over the comparison family).

Every metric is reported as mean +/- std across seeds, with a bootstrap 95% CI
computed on predictions pooled across those seeds.

Usage:
    python tools/stage3_report.py
    python tools/stage3_report.py --scenarios 1,2
    python tools/stage3_report.py --models FCSNN,FUJITA
    python tools/stage3_report.py --no-significance      # skip the slow part
    python tools/stage3_report.py --resamples 500
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tools.stage_lib import (  # noqa: E402
    ROOT,
    SVM_MODELS,
    as_bool,
    collect_runs,
    load_config,
    parse_scenarios,
    write_table,
)
from tools.stats_compare import (  # noqa: E402
    DEFAULT_RESAMPLES,
    align_predictions,
    bootstrap_ci_multi,
    holm_bonferroni,
    metric_fn,
    paired_bootstrap,
)

OUT_DIR = "results/paper"

NOMINAL_METRICS = [
    "accuracy", "precision", "recall", "macro_f1", "macro_f2",
    "balanced_accuracy", "mcc",
]
ORDINAL_METRICS = ["mae", "rmse", "off_by_one_accuracy", "qwk", "linear_kappa", "spearman"]

# Which metric the significance tests compare arms on.
COMPARISON_METRIC = "macro_f2"


def front(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Move `columns` to the front, leaving the rest in order. (`scenario` is
    already produced by build_rows, so it is reordered rather than inserted.)"""
    lead = [c for c in columns if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]

IDENTITY_COLUMNS = [
    "scenario", "model", "fusion", "shared", "dropout", "dense", "base_filters",
    "kernel", "decision_function", "C", "gamma",
    "optimizer", "learning_rate", "batch_size",
]


# ============================================================
# Grouping runs into arms
# ============================================================


def arm_key(row) -> tuple:
    """Everything that identifies an arm *except* the seed."""
    return tuple(str(row.get(c, "")) for c in IDENTITY_COLUMNS)


def summarise_arm(runs: pd.DataFrame, metrics: List[str]) -> Dict:
    """mean / std / 'mean +/- std' per metric across this arm's seeds."""
    out: Dict = {"n_seeds": int(len(runs))}
    for metric in metrics:
        if metric not in runs.columns:
            continue
        values = pd.to_numeric(runs[metric], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        # ddof=1 is the sample std (what "+/- std across 5 seeds" means);
        # it is undefined for a single seed, so report 0.0 there and let
        # n_seeds=1 tell the reader why.
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = std
        out[metric] = f"{mean:.4f} +/- {std:.4f}"
    return out


def load_predictions(result_dir: str) -> tuple | None:
    """(source_index, y_true, y_pred) from one run's predictions.csv."""
    path = Path(result_dir)
    if not path.is_absolute():
        path = ROOT / path
    path = path / "predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {"source_index", "y_true", "y_pred"}
    if not needed.issubset(df.columns):
        return None
    return (
        df["source_index"].to_numpy(),
        df["y_true"].to_numpy(),
        df["y_pred"].to_numpy(),
    )


def arm_predictions(runs: pd.DataFrame) -> List[tuple]:
    """One (source_index, y_true, y_pred) triple per seed, seed-sorted so two
    arms line up seed-for-seed in a paired test."""
    triples = []
    for _, row in runs.sort_values("seed").iterrows():
        loaded = load_predictions(str(row["result_dir"]))
        if loaded is not None:
            triples.append(loaded)
    return triples


def add_bootstrap_ci(summary: Dict, runs: pd.DataFrame, metrics: List[str], resamples: int) -> None:
    """95% CIs on predictions pooled across this arm's seeds.

    All metrics share one set of resample draws — both because it is far
    cheaper than bootstrapping each metric separately, and because the
    intervals then describe the same resampled test sets.

    Ordinal metrics (QWK, MAE, ...) have no bootstrap adapter and are reported
    as mean +/- std only.
    """
    triples = arm_predictions(runs)
    if not triples:
        return
    y_true = np.concatenate([t[1] for t in triples])
    y_pred = np.concatenate([t[2] for t in triples])
    num_classes = int(pd.to_numeric(runs["num_classes"], errors="coerce").iloc[0])

    for metric, (_, lo, hi) in bootstrap_ci_multi(
        y_true, y_pred, metrics, num_classes, n_resamples=resamples
    ).items():
        summary[f"{metric}_ci_low"] = lo
        summary[f"{metric}_ci_high"] = hi


def per_class_columns(runs: pd.DataFrame) -> Dict:
    """Per-class precision/recall/F1, averaged across seeds, read from each
    run's classification_report.csv."""
    frames = []
    for _, row in runs.iterrows():
        path = Path(str(row["result_dir"]))
        if not path.is_absolute():
            path = ROOT / path
        path = path / "classification_report.csv"
        if path.exists():
            frames.append(pd.read_csv(path, index_col=0))
    if not frames:
        return {}

    out: Dict = {}
    for label in frames[0].index:
        if str(label) in ("accuracy", "macro avg", "weighted avg"):
            continue
        for field in ("precision", "recall", "f1-score"):
            values = [f.loc[label, field] for f in frames if label in f.index and field in f.columns]
            if values:
                name = f"class{label}_{field.replace('-score', '')}"
                out[f"{name}_mean"] = float(np.mean(values))
                out[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def summed_confusion(runs: pd.DataFrame) -> str:
    """Confusion matrices summed across seeds, as a JSON string."""
    import json

    total = None
    for value in runs.get("confusion_matrix_json", []):
        try:
            matrix = np.array(json.loads(value))
        except Exception:
            continue
        total = matrix if total is None else total + matrix
    return json.dumps(total.tolist()) if total is not None else ""


def build_rows(df: pd.DataFrame, metrics: List[str], resamples: int) -> pd.DataFrame:
    """One row per arm (seeds collapsed)."""
    rows = []
    for _, runs in df.groupby(df.apply(arm_key, axis=1), sort=False):
        first = runs.iloc[0]
        row = {c: first.get(c) for c in IDENTITY_COLUMNS if c in runs.columns}
        for extra in ("params", "ordinal", "monitor_metric"):
            if extra in runs.columns:
                row[extra] = first.get(extra)
        row["seeds"] = ",".join(str(s) for s in sorted(runs["seed"].unique()))
        row["best_val_metric_mean"] = float(
            pd.to_numeric(runs["best_val_metric"], errors="coerce").mean()
        )
        row.update(summarise_arm(runs, metrics))
        add_bootstrap_ci(row, runs, metrics, resamples)
        row.update(per_class_columns(runs))
        row["confusion_matrix_summed"] = summed_confusion(runs)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Aggregate A and B
# ============================================================


def metrics_for(scenario: int) -> List[str]:
    return NOMINAL_METRICS + (ORDINAL_METRICS if scenario == 5 else [])


def filter_to_hpo(runs: pd.DataFrame, arm) -> pd.DataFrame:
    """Keep only the runs at this arm's frozen hyperparameters.

    Essential, not cosmetic: `results/` still holds every Stage-0 HPO candidate
    for the reference configs. Without this filter an arm would pool runs from
    *different* hyperparameter settings into one mean +/- std, silently
    inflating both n_seeds and the spread.
    """
    for column in ("optimizer", "batch_size", "kernel", "decision_function", "gamma"):
        if column in runs.columns and pd.notna(arm.get(column)):
            runs = runs[runs[column].astype(str) == str(arm[column])]
    for column in ("learning_rate", "C"):
        if column in runs.columns and pd.notna(arm.get(column)):
            runs = runs[
                np.isclose(
                    pd.to_numeric(runs[column], errors="coerce").fillna(np.nan),
                    float(arm[column]),
                    equal_nan=False,
                )
            ]
    return runs


def aggregate_a(
    df: pd.DataFrame, stage1: pd.DataFrame, scenario: int, resamples: int
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Cross-model table: the 4 FCSNN cell winners + every baseline arm.

    Also returns {arm -> the exact runs behind it}, so the significance tests
    compare precisely the rows that were tabulated instead of re-deriving the
    filter and risking drift between the two.
    """
    metrics = metrics_for(scenario)
    wanted = stage1[(stage1["scenario"] == scenario) & (~stage1["control"].map(as_bool))]
    parts = []
    arm_runs: Dict[str, pd.DataFrame] = {}

    for _, arm in wanted.iterrows():
        model = arm["model"]
        runs = df[(df["model"] == model) & (df["scenario"] == scenario)]
        if model == "FCSNN":
            runs = runs[
                (runs["fusion"] == arm["fusion"])
                & (runs["shared"].map(as_bool) == as_bool(arm["shared"]))
                & (runs["dropout"].map(as_bool) == as_bool(arm["dropout"]))
                & (runs["dense"].map(as_bool) == as_bool(arm["dense"]))
                & (pd.to_numeric(runs["base_filters"], errors="coerce") == 32)
            ]
        elif model == "BDD" and "base_filters" in runs.columns:
            runs = runs[pd.to_numeric(runs["base_filters"], errors="coerce") == int(arm["base_filters"])]
        runs = filter_to_hpo(runs, arm)
        if runs.empty:
            print(f"  !! scenario {scenario}: no Stage-2 runs for arm '{arm['arm']}'")
            continue
        built = build_rows(runs, metrics, resamples)
        built.insert(0, "arm", arm["arm"])
        parts.append(built)
        arm_runs[arm["arm"]] = runs

    if not parts:
        return pd.DataFrame(), {}
    out = pd.concat(parts, ignore_index=True)
    return front(out, ["scenario", "arm", "model"]), arm_runs


def aggregate_b(df: pd.DataFrame, stage1: pd.DataFrame, scenario: int, resamples: int) -> pd.DataFrame:
    """FCSNN ablation table: all 16 configs + the capacity control."""
    metrics = metrics_for(scenario)
    all_fcsnn = df[(df["model"] == "FCSNN") & (df["scenario"] == scenario)]
    if all_fcsnn.empty:
        return pd.DataFrame()

    winners = stage1[(stage1["scenario"] == scenario) & (stage1["model"] == "FCSNN")]
    control_bf = load_config("configs/fcsnn.json")["capacity_control"]["base_filters"]

    # Each cell's four dropout x dense configs share that cell's frozen
    # hyperparameters, so select per cell and drop Stage-0's other candidates.
    selected = []
    for _, arm in winners.iterrows():
        cell = all_fcsnn[
            (all_fcsnn["fusion"] == arm["fusion"])
            & (all_fcsnn["shared"].map(as_bool) == as_bool(arm["shared"]))
            & (pd.to_numeric(all_fcsnn["base_filters"], errors="coerce") == int(arm["base_filters"]))
        ]
        cell = filter_to_hpo(cell, arm)
        if as_bool(arm.get("control", False)):
            # The control is the single config Stage 1 picked, not a sub-grid.
            cell = cell[
                (cell["dropout"].map(as_bool) == as_bool(arm["dropout"]))
                & (cell["dense"].map(as_bool) == as_bool(arm["dense"]))
            ]
        if not cell.empty:
            selected.append(cell)

    if not selected:
        return pd.DataFrame()
    runs = pd.concat(selected).drop_duplicates(subset=["result_dir"])

    out = build_rows(runs, metrics, resamples)
    out["control"] = pd.to_numeric(out["base_filters"], errors="coerce") == control_bf

    winner_keys = {
        (r["fusion"], as_bool(r["shared"]), as_bool(r["dropout"]), as_bool(r["dense"]))
        for _, r in winners.iterrows()
    }
    out["is_cell_winner"] = out.apply(
        lambda r: (r["fusion"], as_bool(r["shared"]), as_bool(r["dropout"]), as_bool(r["dense"]))
        in winner_keys,
        axis=1,
    )
    out = out.sort_values(["fusion", "shared", "dropout", "dense"]).reset_index(drop=True)
    return front(out, ["scenario", "fusion", "shared", "dropout", "dense", "base_filters", "control"])


# ============================================================
# Significance
# ============================================================


def comparison_pairs(arms: List[str], table_a: pd.DataFrame, family: str) -> List[tuple]:
    """Which arm pairs enter the Holm-corrected family.

    Family size directly costs power: Holm multiplies the smallest p-value by
    the family size, and a paired bootstrap with B resamples cannot report a
    p below 2/(B+1). With B=2000 that floor is 0.001, so a family of 28 (every
    pair of 8 arms) can still reach 0.028 — but a chunk of that family is
    comparisons the paper never makes, such as Moya vs Hajeb.

    "architecture" (default) keeps only what the paper claims: the four FCSNN
    arms against each other (the sub-vs-concat and shared-vs-unshared
    contrasts), plus the best FCSNN arm against each baseline.
    "all" is every pair, for an appendix.
    """
    if family == "all":
        return list(itertools.combinations(arms, 2))

    fcsnn = [a for a in arms if a.startswith("FCSNN")]
    baselines = [a for a in arms if not a.startswith("FCSNN")]
    pairs = list(itertools.combinations(fcsnn, 2))

    if fcsnn and baselines and "macro_f2_mean" in table_a.columns:
        rows = table_a[table_a["arm"].isin(fcsnn)]
        best = rows.loc[pd.to_numeric(rows["macro_f2_mean"], errors="coerce").idxmax(), "arm"]
        pairs += [(best, b) for b in baselines]
    return pairs


def significance(
    df: pd.DataFrame,
    arm_runs: Dict[str, pd.DataFrame],
    table_a: pd.DataFrame,
    scenario: int,
    resamples: int,
    family: str = "architecture",
) -> pd.DataFrame:
    """Paired bootstrap over every pair of arms in Aggregate A, Holm-corrected
    across the whole family for this scenario.

    `arm_runs` comes straight from aggregate_a, so each arm's predictions are
    exactly the runs its table row summarised.
    """
    arms = list(arm_runs)
    if len(arms) < 2:
        return pd.DataFrame()

    num_classes_series = pd.to_numeric(df[df["scenario"] == scenario]["num_classes"], errors="coerce")
    if num_classes_series.empty:
        return pd.DataFrame()
    try:
        score = metric_fn(COMPARISON_METRIC, int(num_classes_series.iloc[0]))
    except ValueError:
        return pd.DataFrame()

    preds = {}
    for arm, runs in arm_runs.items():
        triples = arm_predictions(runs)
        if triples:
            preds[arm] = triples

    pairs = comparison_pairs([a for a in arms if a in preds], table_a, family)
    rows = []
    for a, b in pairs:
        try:
            y_true, pa, pb = align_predictions(preds[a], preds[b])
        except ValueError as exc:
            print(f"  (skipping {a} vs {b}: {exc})")
            continue
        if len(y_true) == 0:
            continue
        res = paired_bootstrap(y_true, pa, pb, score, n_resamples=resamples)
        rows.append(
            {
                "scenario": scenario,
                "metric": COMPARISON_METRIC,
                "arm_a": a,
                "arm_b": b,
                "diff_a_minus_b": res["diff"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "p_value": res["p_value"],
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    adjusted, rejected = holm_bonferroni(out["p_value"].tolist())
    out["p_holm"] = adjusted
    out["significant_at_0.05"] = rejected
    out["family"] = family
    out["family_size"] = len(out)
    # The smallest p a paired bootstrap can report, so a reader can tell
    # "not significant" apart from "not enough resamples to show it".
    out["p_floor"] = 2.0 / (resamples + 1)
    return out.sort_values("p_value").reset_index(drop=True)


# ============================================================
# Main
# ============================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", type=str, default=None, help="e.g. 1,2,5 (default: all)")
    p.add_argument("--models", type=str, default=None, help="restrict Aggregate A to these models")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, help="bootstrap draws")
    p.add_argument("--no-significance", action="store_true", help="skip the paired tests")
    p.add_argument(
        "--family", choices=("architecture", "all"), default="architecture",
        help="which comparisons enter the Holm family: 'architecture' (the 4 FCSNN "
             "arms pairwise + best FCSNN vs each baseline) or 'all' pairs",
    )
    args = p.parse_args()

    scenarios = parse_scenarios(args.scenarios)

    stage1_path = ROOT / "results" / "stage1_winners.csv"
    if not stage1_path.exists():
        raise SystemExit("results/stage1_winners.csv not found — run tools/stage1_ablation.py first.")
    stage1 = pd.read_csv(stage1_path)
    if args.models:
        stage1 = stage1[stage1["model"].isin(args.models.split(","))]

    df = collect_runs(scenarios=scenarios)
    if df.empty:
        raise SystemExit("No runs found under results/.")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")

    print(f"Stage 3 — report | scenarios={scenarios} resamples={args.resamples}")

    all_a, all_b, all_sig = [], [], []
    for scenario in scenarios:
        print(f"\nscenario {scenario}:")
        table_a, arm_runs = aggregate_a(df, stage1, scenario, args.resamples)
        if not table_a.empty:
            write_table(table_a, f"{OUT_DIR}/table_main_scenario_{scenario}.csv")
            all_a.append(table_a)

        table_b = aggregate_b(df, stage1, scenario, args.resamples)
        if not table_b.empty:
            write_table(table_b, f"{OUT_DIR}/table_ablation_scenario_{scenario}.csv")
            all_b.append(table_b)

        if not args.no_significance and not table_a.empty:
            sig = significance(df, arm_runs, table_a, scenario, args.resamples, args.family)
            if not sig.empty:
                all_sig.append(sig)

    if all_a:
        combined = pd.concat(all_a, ignore_index=True)
        write_table(combined, f"{OUT_DIR}/table_main.csv")
        try:
            combined.to_excel(ROOT / OUT_DIR / "table_main.xlsx", index=False)
            print(f"  wrote {OUT_DIR}/table_main.xlsx")
        except Exception as exc:
            print(f"  (xlsx skipped: {exc})")
    if all_b:
        write_table(pd.concat(all_b, ignore_index=True), f"{OUT_DIR}/table_ablation.csv")
    if all_sig:
        write_table(pd.concat(all_sig, ignore_index=True), f"{OUT_DIR}/significance.csv")

    stage0_path = ROOT / "results" / "stage0_hpo.csv"
    if stage0_path.exists():
        hpo = pd.read_csv(stage0_path)
        write_table(hpo[hpo["scenario"].isin(scenarios)], f"{OUT_DIR}/table_hpo.csv")

    print(f"\nTables written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
