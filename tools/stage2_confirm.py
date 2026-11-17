"""Stage 2 — confirmation: multi-seed training, test set evaluated.

Every FCSNN ablation config is retrained on all five seeds, not only the cell
winners. The ablation table is a *reported* table, so leaving its non-winning
rows at one seed would reproduce the single-run weakness this whole protocol
exists to remove. `--winners-only` falls back to winners + control if the run
budget has to be cut; those rows then carry n_seeds=1 and must be labelled as
such in the paper.

Hyperparameters stay frozen per cell from Stage 0. Each seed stops
independently on its own validation curve — Stage 1's epoch count is never
reused. Seed 12345 already exists from Stages 0-1 and is skipped automatically.

Nothing is aggregated here; tools/stage3_report.py reads the runs back.

Usage:
    python tools/stage2_confirm.py
    python tools/stage2_confirm.py --scenarios 1,2 --models FCSNN
    python tools/stage2_confirm.py --winners-only
    python tools/stage2_confirm.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tools.stage_lib import (  # noqa: E402
    FCSNN_CELLS,
    NN_MODELS,
    ROOT,
    SVM_CONFIG,
    SVM_MODELS,
    SVM_SCRIPT,
    as_bool,
    cell_tag,
    hpo_from_row,
    load_config,
    narrowed_config,
    native,
    parse_scenarios,
    run_trainer,
)

FCSNN_SCRIPT, FCSNN_CONFIG = NN_MODELS["FCSNN"]
ALL_MODELS = (*NN_MODELS, *SVM_MODELS)


def load_stage1() -> pd.DataFrame:
    path = ROOT / "results" / "stage1_winners.csv"
    if not path.exists():
        raise SystemExit("results/stage1_winners.csv not found — run tools/stage1_ablation.py first.")
    return pd.read_csv(path)


def seeds() -> list[int]:
    return load_config(FCSNN_CONFIG)["seeds"]


def fcsnn_jobs(stage1: pd.DataFrame, scenarios, winners_only: bool):
    """One job per (scenario, cell): the 4 configs of that cell share its frozen
    hyperparameters, so they go in a single invocation."""
    jobs = []
    for scenario in scenarios:
        rows = stage1[(stage1["model"] == "FCSNN") & (stage1["scenario"] == scenario)]
        if rows.empty:
            print(f"  !! no Stage-1 FCSNN arms for scenario {scenario} — skipping")
            continue

        for cell in FCSNN_CELLS:
            match = rows[
                (~rows["control"].map(as_bool))
                & (rows["fusion"] == cell["fusion"])
                & (rows["shared"].map(as_bool) == cell["shared"])
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            if winners_only:
                ablation = {
                    "dropout": [as_bool(row["dropout"])],
                    "dense": [as_bool(row["dense"])],
                }
            else:
                ablation = {"dropout": [True, False], "dense": [True, False]}
            jobs.append(
                (
                    FCSNN_SCRIPT,
                    narrowed_config(
                        FCSNN_CONFIG,
                        scenarios=[scenario],
                        ablation={
                            "fusion": [cell["fusion"]],
                            "shared": [cell["shared"]],
                            "base_filters": [cell["base_filters"]],
                            **ablation,
                        },
                        hpo=hpo_from_row(row),
                        seeds=seeds(),
                    ),
                    f"stage2_s{scenario}_{cell_tag(cell)}",
                )
            )

        # Capacity control: exactly the one config Stage 1 selected.
        ctrl = rows[rows["control"].map(as_bool)]
        if not ctrl.empty:
            row = ctrl.iloc[0]
            jobs.append(
                (
                    FCSNN_SCRIPT,
                    narrowed_config(
                        FCSNN_CONFIG,
                        scenarios=[scenario],
                        ablation={
                            "fusion": [row["fusion"]],
                            "shared": [as_bool(row["shared"])],
                            "base_filters": [native(row["base_filters"])],
                            "dropout": [as_bool(row["dropout"])],
                            "dense": [as_bool(row["dense"])],
                        },
                        hpo=hpo_from_row(row),
                        seeds=seeds(),
                    ),
                    f"stage2_s{scenario}_control",
                )
            )
    return jobs


def baseline_jobs(stage1: pd.DataFrame, scenarios, models):
    jobs = []
    for scenario in scenarios:
        for model in models:
            rows = stage1[(stage1["model"] == model) & (stage1["scenario"] == scenario)]
            if rows.empty:
                continue
            row = rows.iloc[0]

            if model in SVM_MODELS:
                cfg = narrowed_config(SVM_CONFIG, scenarios=[scenario], ablation={}, seeds=seeds())
                cfg["models"] = [model]
                cfg["hpo_grid"] = {
                    "kernel": [native(row["kernel"])],
                    "decision_function": [native(row["decision_function"])],
                    "C": [native(row["C"])],
                    "gamma": [native(row["gamma"])],
                }
                jobs.append((SVM_SCRIPT, cfg, f"stage2_s{scenario}_{model}"))
                continue

            script, config = NN_MODELS[model]
            ablation = {}
            if model == "BDD":
                ablation["base_filters"] = [native(row["base_filters"])]
            jobs.append(
                (
                    script,
                    narrowed_config(
                        config,
                        scenarios=[scenario],
                        ablation=ablation,
                        hpo=hpo_from_row(row),
                        seeds=seeds(),
                    ),
                    f"stage2_s{scenario}_{model}",
                )
            )
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", type=str, default=None, help="e.g. 1,2,5 (default: all)")
    p.add_argument("--models", type=str, default=None, help=f"comma-separated subset of {ALL_MODELS}")
    p.add_argument(
        "--winners-only", action="store_true",
        help="multi-seed only the cell winners + control, leaving the other 12 "
             "ablation configs at one seed (saves ~240 runs; label them in the paper)",
    )
    p.add_argument("--dry-run", action="store_true", help="list the invocations and exit")
    args = p.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    models = args.models.split(",") if args.models else list(ALL_MODELS)
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(ALL_MODELS)}")

    stage1 = load_stage1()
    print(f"Stage 2 — confirmation | scenarios={scenarios} models={models} seeds={seeds()}")
    if args.winners_only:
        print("  (--winners-only: non-winning ablation configs stay at a single seed)")

    jobs = []
    if "FCSNN" in models:
        jobs += fcsnn_jobs(stage1, scenarios, args.winners_only)
    jobs += baseline_jobs(stage1, scenarios, [m for m in models if m != "FCSNN"])

    if args.dry_run:
        for script, cfg, tag in jobs:
            print(f"  {tag}: {script} ablation={cfg['ablation_grid']} hpo={cfg['hpo_grid']}")
        print(f"\n{len(jobs)} invocation(s)")
        return

    for script, cfg, tag in jobs:
        run_trainer(script, cfg, tag)

    print(f"\nStage 2 done ({len(jobs)} invocations). Next: python tools/stage3_report.py")


if __name__ == "__main__":
    main()
