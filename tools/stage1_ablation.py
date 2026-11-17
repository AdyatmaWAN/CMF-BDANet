"""Stage 1 — FCSNN ablation + capacity control (validation only, one seed).

With hyperparameters frozen per cell by Stage 0, this runs the 16-config FCSNN
ablation:

    fusion{sub,concat} x shared{T,F} x dropout{T,F} x dense{T,F}

four configs per (sharing x fusion) cell, and picks each cell's best by
validation -> 4 reported arms per scenario. The 4 reference configs
(dropout=on, dense=on) already exist from Stage 0 and are skipped automatically.

It then runs the capacity control: the winning fusion, unshared, at the
parameter-matched width (base_filters=22, which is 1.038x one shared tower at
32 — verified by tests/test_fcsnn_shapes.py), over its own dropout x dense
sub-grid, inheriting the unshared cell's hyperparameters. That is 1 more arm,
reported in the ablation table only, never as a competing model.

Baselines have no ablation axis left once class weighting is gone, so their
Stage-0 winner is their arm and no run happens here.

Output: results/stage1_winners.csv — every arm that Stage 2 will multi-seed.
Nothing is reported from this stage; the test split is still untouched.

Usage:
    python tools/stage1_ablation.py
    python tools/stage1_ablation.py --scenarios 1,2
    python tools/stage1_ablation.py --select-only
    python tools/stage1_ablation.py --no-control
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
    REPORT_BF,
    ROOT,
    SVM_MODELS,
    as_bool,
    cell_tag,
    collect_runs,
    control_bf,
    hpo_from_row,
    load_config,
    narrowed_config,
    parse_scenarios,
    pick_best,
    require_columns,
    run_trainer,
    write_table,
)

FCSNN_SCRIPT, FCSNN_CONFIG = NN_MODELS["FCSNN"]
ABLATION_AXES = {"dropout": [True, False], "dense": [True, False]}


def load_stage0() -> pd.DataFrame:
    path = ROOT / "results" / "stage0_hpo.csv"
    if not path.exists():
        raise SystemExit("results/stage0_hpo.csv not found — run tools/stage0_hpo.py first.")
    df = pd.read_csv(path)
    require_columns(df, ["scenario", "model", "optimizer", "learning_rate", "batch_size"], "stage0")
    return df


def fcsnn_hpo(stage0: pd.DataFrame, scenario: int, cell: dict) -> dict | None:
    """The hyperparameters Stage 0 chose for one (scenario, cell).

    `base_filters` is part of the cell identity: the capacity control was tuned
    at its own width, so matching on fusion/shared alone would hand it bf=32's
    hyperparameters and undo the reason it was tuned separately.
    """
    match = stage0[
        (stage0["model"] == "FCSNN")
        & (stage0["scenario"] == scenario)
        & (stage0["fusion"] == cell["fusion"])
        & (stage0["shared"].map(as_bool) == cell["shared"])
        & (pd.to_numeric(stage0["base_filters"], errors="coerce") == cell["base_filters"])
    ]
    if match.empty:
        print(f"  !! no Stage-0 winner for scenario {scenario} {cell_tag(cell)} — skipping")
        return None
    return hpo_from_row(match.iloc[0])


def train_ablation(stage0: pd.DataFrame, scenarios) -> None:
    """The 16-config grid, one invocation per (scenario, cell) since each cell
    carries its own frozen hyperparameters."""
    for scenario in scenarios:
        for cell in FCSNN_CELLS:
            hpo = fcsnn_hpo(stage0, scenario, cell)
            if hpo is None:
                continue
            cfg = narrowed_config(
                FCSNN_CONFIG,
                scenarios=[scenario],
                ablation={
                    "fusion": [cell["fusion"]],
                    "shared": [cell["shared"]],
                    "base_filters": [cell["base_filters"]],
                    **ABLATION_AXES,
                },
                hpo=hpo,
                seeds=[FIRST_SEED],
            )
            run_trainer(FCSNN_SCRIPT, cfg, f"stage1_s{scenario}_{cell_tag(cell)}")


def select_cell_winners(scenarios) -> pd.DataFrame:
    df = collect_runs(scenarios=scenarios)
    fcsnn = df[df["model"] == "FCSNN"] if not df.empty else df
    if fcsnn.empty:
        raise SystemExit("No FCSNN runs found — run Stage 1 training first.")
    reported = fcsnn[pd.to_numeric(fcsnn["base_filters"], errors="coerce") == REPORT_BF]
    best = pick_best(reported, ["scenario", "fusion", "shared"])
    best["arm"] = best.apply(
        lambda r: f"FCSNN {'shared' if as_bool(r['shared']) else 'unshared'} + {r['fusion']}", axis=1
    )
    best["control"] = False
    return best


def train_control(cell_winners: pd.DataFrame, stage0: pd.DataFrame, scenarios) -> None:
    """Capacity control: winning fusion, unshared, parameter-matched width."""
    bf = control_bf()
    winners = winning_fusion_per_scenario(cell_winners)
    for scenario in scenarios:
        if scenario not in winners:
            continue
        winning_fusion = winners[scenario]
        # Stage 0 tuned both control fusions; take the one that won, with its
        # own hyperparameters rather than bf=32's.
        cell = {"fusion": winning_fusion, "shared": False, "base_filters": bf}
        hpo = fcsnn_hpo(stage0, scenario, cell)
        if hpo is None:
            continue
        print(f"  scenario {scenario}: winning fusion = {winning_fusion}, control at bf={bf}")
        cfg = narrowed_config(
            FCSNN_CONFIG,
            scenarios=[scenario],
            ablation={
                "fusion": [winning_fusion],
                "shared": [False],
                "base_filters": [bf],
                **ABLATION_AXES,
            },
            hpo=hpo,
            seeds=[FIRST_SEED],
        )
        run_trainer(FCSNN_SCRIPT, cfg, f"stage1_s{scenario}_control_bf{bf}")


def winning_fusion_per_scenario(cell_winners: pd.DataFrame) -> dict:
    """The fusion with the best cell winner, per scenario. This is the single
    source of truth for which control arm is reported — both `train_control`
    and `select_control` read it, so they cannot disagree."""
    out = {}
    for scenario, rows in cell_winners.groupby("scenario"):
        out[scenario] = rows.loc[rows["best_val_metric"].idxmax(), "fusion"]
    return out


def select_control(scenarios, cell_winners: pd.DataFrame) -> pd.DataFrame:
    """The single best capacity-control config per scenario.

    Stage 0 tuned the control at *both* fusions, so `results/` holds bf=22 runs
    for the losing fusion too. Those are not a reported arm: filter to the
    winning fusion explicitly rather than inferring it from how many runs each
    has, which would be a 21-vs-18 coin flip after a real Stage 1.
    """
    bf = control_bf()
    df = collect_runs(scenarios=scenarios)
    if df.empty or "base_filters" not in df.columns:
        return pd.DataFrame()
    ctrl = df[
        (df["model"] == "FCSNN")
        & (pd.to_numeric(df["base_filters"], errors="coerce") == bf)
    ]
    if ctrl.empty:
        return pd.DataFrame()

    winners = winning_fusion_per_scenario(cell_winners)
    kept = [
        group[group["fusion"] == winners[scenario]]
        for scenario, group in ctrl.groupby("scenario")
        if scenario in winners
    ]
    kept = [k for k in kept if not k.empty]
    if not kept:
        return pd.DataFrame()

    best = pick_best(pd.concat(kept), ["scenario"])
    best["arm"] = f"FCSNN capacity control (bf={bf})"
    best["control"] = True
    return best


def baseline_arms(scenarios) -> pd.DataFrame:
    """One arm per baseline model.

    Stage 0 searched BDDNet per `base_filters` cell, so it contributes two rows
    per scenario; collapsing them to the better one *is* BDDNet's ablation
    selection — the step that would otherwise have no home now that class
    weighting is gone. Fujita and the SVMs have a single cell each, so for them
    this is a pass-through.
    """
    stage0 = load_stage0()
    rows = stage0[
        stage0["model"].isin(["BDD", "FUJITA", *SVM_MODELS])
        & stage0["scenario"].isin(list(scenarios))
    ].copy()
    if rows.empty:
        return rows
    rows = pick_best(rows, ["scenario", "model"])
    rows["arm"] = rows["model"]
    rows["control"] = False
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", type=str, default=None, help="e.g. 1,2,5 (default: all)")
    p.add_argument("--select-only", action="store_true", help="skip training, just re-select")
    p.add_argument("--no-control", action="store_true", help="skip the capacity-control arm")
    args = p.parse_args()

    scenarios = parse_scenarios(args.scenarios)
    stage0 = load_stage0()
    print(f"Stage 1 — FCSNN ablation | scenarios={scenarios} seed={FIRST_SEED}")

    if not args.select_only:
        train_ablation(stage0, scenarios)

    cell_winners = select_cell_winners(scenarios)

    control = pd.DataFrame()
    if not args.no_control:
        if not args.select_only:
            train_control(cell_winners, stage0, scenarios)
        control = select_control(scenarios, cell_winners)

    parts = [p_ for p_ in (cell_winners, control, baseline_arms(scenarios)) if not p_.empty]
    winners = pd.concat(parts, ignore_index=True)

    keep = [
        c
        for c in (
            "scenario", "model", "arm", "control", "fusion", "shared", "dropout", "dense",
            "base_filters", "params", "kernel", "decision_function", "C", "gamma",
            "optimizer", "learning_rate", "batch_size",
            "best_val_metric", "monitor_metric", "result_dir",
        )
        if c in winners.columns
    ]
    winners = winners[keep].sort_values(["scenario", "model", "arm"]).reset_index(drop=True)
    write_table(winners, "results/stage1_winners.csv")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
