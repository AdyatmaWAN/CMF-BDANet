"""Stage 0 — per-cell hyperparameter search (validation only, one seed).

For every model, the full HPO grid is searched once per *cell*, at a fixed
reference architecture. A "cell" is:

    FCSNN   -> each (sharing x fusion) combination  -> 4 cells
    BDDNet  -> each base_filters value              -> 2 cells
    FUJITA  -> the model itself                     -> 1 cell
    Moya    -> the model itself                     -> 1 cell (its own grid)
    Hajeb   -> the model itself                     -> 1 cell (its own grid)

Tuning per cell rather than once on a reference architecture matters here: the
paper compares shared-vs-unshared and sub-vs-concat, so hyperparameters chosen
on one arm and applied to its competitor would bias the comparison.

Selection is on validation (`best_val_metric`, i.e. macro-F2, or QWK on the
ordinal scenario). The test split is not read until Stage 2.

Output: results/stage0_hpo.csv — the winning (optimizer, learning_rate,
batch_size) per (scenario, model, cell). Frozen for Stages 1 and 2.

Usage:
    python tools/stage0_hpo.py
    python tools/stage0_hpo.py --scenarios 1,2 --models FCSNN
    python tools/stage0_hpo.py --select-only      # skip training, re-select
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tools.stage_lib import (  # noqa: E402
    FCSNN_CELLS,
    FIRST_SEED,
    NN_MODELS,
    SVM_CONFIG,
    SVM_MODELS,
    SVM_SCRIPT,
    cell_tag,
    collect_runs,
    load_config,
    narrowed_config,
    parse_scenarios,
    pick_best,
    run_trainer,
    write_table,
)

ALL_MODELS = (*NN_MODELS, *SVM_MODELS)

# Reference architecture per cell: dropout and dense on, residual on, bf=32.
FCSNN_REFERENCE = {"dropout": [True], "dense": [True], "base_filters": [32]}


def train(models, scenarios) -> None:
    for model in models:
        if model == "FCSNN":
            for cell in FCSNN_CELLS:
                cfg = narrowed_config(
                    NN_MODELS["FCSNN"][1],
                    scenarios=scenarios,
                    ablation={
                        "fusion": [cell["fusion"]],
                        "shared": [cell["shared"]],
                        **FCSNN_REFERENCE,
                    },
                    seeds=[FIRST_SEED],
                )
                run_trainer(NN_MODELS["FCSNN"][0], cfg, f"stage0_FCSNN_{cell_tag(cell)}")

        elif model == "BDD":
            base_cfg = load_config(NN_MODELS["BDD"][1])
            for bf in base_cfg["ablation_grid"]["base_filters"]:
                cfg = narrowed_config(
                    NN_MODELS["BDD"][1],
                    scenarios=scenarios,
                    ablation={"base_filters": [bf]},
                    seeds=[FIRST_SEED],
                )
                run_trainer(NN_MODELS["BDD"][0], cfg, f"stage0_BDD_bf{bf}")

        elif model == "FUJITA":
            cfg = narrowed_config(
                NN_MODELS["FUJITA"][1], scenarios=scenarios, ablation={}, seeds=[FIRST_SEED]
            )
            run_trainer(NN_MODELS["FUJITA"][0], cfg, "stage0_FUJITA")

        else:  # Moya / Hajeb — one script, selected by --model
            svm_scenarios = [s for s in scenarios if s in load_config(SVM_CONFIG)["scenarios"]]
            if not svm_scenarios:
                print(f"  ({model}: no SVM-capable scenario in {scenarios}, skipping)")
                continue
            cfg = narrowed_config(SVM_CONFIG, scenarios=svm_scenarios, ablation={}, seeds=[FIRST_SEED])
            cfg["models"] = [model]
            run_trainer(SVM_SCRIPT, cfg, f"stage0_{model}")


def select(models, scenarios) -> pd.DataFrame:
    df = collect_runs(scenarios=scenarios)
    if df.empty:
        raise SystemExit("No runs found under results/. Run Stage 0 training first.")
    df = df[df["model"].isin(list(models))]
    if df.empty:
        raise SystemExit(f"No runs found for models {list(models)}.")

    winners = []

    fcsnn = df[df["model"] == "FCSNN"]
    if not fcsnn.empty:
        # Restrict to the reference architecture so a Stage-1 ablation run does
        # not get mistaken for an HPO candidate when this is re-selected later.
        ref = fcsnn[
            fcsnn["dropout"].astype(str).str.lower().eq("true")
            & fcsnn["dense"].astype(str).str.lower().eq("true")
            & (pd.to_numeric(fcsnn["base_filters"], errors="coerce") == 32)
        ]
        best = pick_best(ref, ["scenario", "model", "fusion", "shared"])
        best["cell"] = best.apply(lambda r: f"fusion_{r['fusion']}_shared_{r['shared']}", axis=1)
        winners.append(best)

    bdd = df[df["model"] == "BDD"]
    if not bdd.empty:
        best = pick_best(bdd, ["scenario", "model", "base_filters"])
        # pandas reads the column as float when other models leave it blank
        best["base_filters"] = pd.to_numeric(best["base_filters"], errors="coerce").astype("Int64")
        best["cell"] = "bf" + best["base_filters"].astype(str)
        winners.append(best)

    rest = df[df["model"].isin(["FUJITA", *SVM_MODELS])]
    if not rest.empty:
        best = pick_best(rest, ["scenario", "model"])
        best["cell"] = best["model"]
        winners.append(best)

    if not winners:
        raise SystemExit("Nothing selectable — no matching runs.")

    out = pd.concat(winners, ignore_index=True)
    keep = [
        c
        for c in (
            "scenario", "model", "cell", "fusion", "shared", "dropout", "dense",
            "base_filters", "kernel", "decision_function", "C", "gamma",
            "optimizer", "learning_rate", "batch_size", "seed",
            "best_val_metric", "monitor_metric", "result_dir",
        )
        if c in out.columns
    ]
    return out[keep].sort_values(["scenario", "model", "cell"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", type=str, default=None, help="e.g. 1,2,5 (default: all)")
    p.add_argument("--models", type=str, default=None, help=f"comma-separated subset of {ALL_MODELS}")
    p.add_argument("--select-only", action="store_true", help="skip training, just re-select")
    args = p.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    models = args.models.split(",") if args.models else list(ALL_MODELS)
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(ALL_MODELS)}")

    print(f"Stage 0 — per-cell HPO | scenarios={scenarios} models={models} seed={FIRST_SEED}")
    if not args.select_only:
        train(models, scenarios)

    winners = select(models, scenarios)
    write_table(winners, "results/stage0_hpo.csv")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
