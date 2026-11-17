"""Shared helpers for the staged experiment protocol (docs/EXPERIMENT_PLAN.md).

Design note: the stage scripts do not reimplement training. They narrow a
model's config (pinning the ablation axes and/or hyperparameters for one cell),
write it to `results/_stage_configs/`, and invoke the existing `train_*.py` with
`--config`. Those scripts already handle data loading, the grid, resumption
(`metrics.csv` exists -> skip) and metric writing, so a stage is just "which
subset of the grid, and what gets selected from it".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STAGE_CONFIG_DIR = Path("results") / "_stage_configs"

# Stage 0 and Stage 1 run at a single seed: the first of the five, so Stage 2
# inherits those runs instead of retraining them.
FIRST_SEED = 12345

# (script, config) per model label as it appears in metrics.csv's `model` column.
NN_MODELS: Dict[str, tuple[str, str]] = {
    "FCSNN": ("train_fcsnn.py", "configs/fcsnn.json"),
    "BDD": ("train_bddnet.py", "configs/bddnet.json"),
    "FUJITA": ("train_fujita.py", "configs/fujita.json"),
}
SVM_MODELS = ("Moya", "Hajeb")
SVM_SCRIPT, SVM_CONFIG = "train_svm.py", "configs/svm.json"

REPORT_BF = 32


def control_bf() -> int:
    """The parameter-matched width for the capacity control, from the config."""
    return int(load_config("configs/fcsnn.json")["capacity_control"]["base_filters"])


# The four reported FCSNN arms: one per (sharing x fusion) cell, at bf=32.
FCSNN_CELLS: List[Dict] = [
    {"shared": shared, "fusion": fusion, "base_filters": REPORT_BF}
    for fusion in ("sub", "concat")
    for shared in (True, False)
]


def fcsnn_control_cells() -> List[Dict]:
    """The capacity-control cells: unshared at the matched width, one per
    fusion.

    Both fusions are tuned in Stage 0 even though only the winning one is
    reported, because Stage 0 runs before Stage 1 has decided which fusion
    wins. Tuning the control in its own right is the point: a narrower network
    can genuinely prefer a different learning rate or batch size, so inheriting
    bf=32's hyperparameters would leave "the control was tuned for a wider
    model" as an alternative explanation for any gap.
    """
    bf = control_bf()
    return [
        {"shared": False, "fusion": fusion, "base_filters": bf}
        for fusion in ("sub", "concat")
    ]

# Hyperparameter columns a cell's winner pins for later stages.
HPO_KEYS = ("optimizer", "learning_rate", "batch_size")

SELECT_COLUMN = "best_val_metric"


def cell_tag(cell: Dict) -> str:
    return f"fusion_{cell['fusion']}_shared_{cell['shared']}_bf{cell['base_filters']}"


# ============================================================
# Config narrowing + trainer invocation
# ============================================================


def load_config(path: str) -> Dict:
    with open(ROOT / path) as f:
        return json.load(f)


def narrowed_config(
    base_config: str,
    *,
    scenarios: Sequence[int],
    ablation: Dict[str, list],
    hpo: Dict[str, list] | None = None,
    seeds: Sequence[int],
) -> Dict:
    """A copy of `base_config` with its grids replaced by the given subsets."""
    cfg = load_config(base_config)
    cfg["scenarios"] = list(scenarios)
    cfg["ablation_grid"] = {**cfg["ablation_grid"], **ablation}
    if hpo is not None:
        cfg["hpo_grid"] = {**cfg["hpo_grid"], **hpo}
    cfg["seeds"] = list(seeds)
    return cfg


def run_trainer(script: str, cfg: Dict, tag: str, *, extra_args: Iterable[str] = ()) -> int:
    """Write `cfg` to results/_stage_configs/<tag>.json and run `script` on it.

    Returns the subprocess exit code. A non-zero code is reported but does not
    abort the stage: `grid_search` already logs per-run failures to error.log
    and continues, so the usual cause of a non-zero exit is something the next
    invocation can retry.
    """
    config_dir = ROOT / STAGE_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{tag}.json"
    # `default=native` is a backstop: any numpy scalar that slipped through a
    # pandas round-trip would otherwise raise here rather than at the call site.
    config_path.write_text(json.dumps(cfg, indent=2, default=native))

    cmd = [sys.executable, script, "--config", str(config_path.relative_to(ROOT)), *extra_args]
    print(f"\n>>> [{tag}] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"!!! [{tag}] exited with code {proc.returncode}", flush=True)
    return proc.returncode


# ============================================================
# Reading results back
# ============================================================


def collect_runs(results_root: str = "results", scenarios: Sequence[int] | None = None) -> pd.DataFrame:
    """Every metrics.csv under results/scenario_*/ as one DataFrame."""
    root = ROOT / results_root
    frames = []
    for path in root.glob("scenario_*/**/metrics.csv"):
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # a half-written row should not kill the stage
            print(f"  (skipping unreadable {path}: {exc})")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if scenarios is not None:
        df = df[df["scenario"].isin(list(scenarios))]
    return df


def require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{context}: results are missing column(s) {missing}. "
            f"Has the previous stage been run?"
        )


def pick_best(df: pd.DataFrame, group_cols: Sequence[str], metric: str = SELECT_COLUMN) -> pd.DataFrame:
    """One winning row per group, by validation metric (higher is better).

    Ties break on the first row after a stable sort, which given the grid order
    is the earlier-listed hyperparameter — deterministic, which is what matters.
    """
    require_columns(df, [*group_cols, metric], "selection")
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])
    if d.empty:
        return d
    idx = d.groupby(list(group_cols), dropna=False)[metric].idxmax()
    return d.loc[idx].reset_index(drop=True)


def write_table(df: pd.DataFrame, path: str) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def as_bool(value) -> bool:
    """metrics.csv round-trips booleans as the strings 'True'/'False'."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def native(value):
    """A pandas/numpy scalar as a plain Python value.

    Two reasons this matters: `json.dumps` rejects numpy scalars outright, and
    a float that was originally an int (pandas widens `C: 1` to `1.0` when the
    column has blanks) would build the result directory `C_1.0` instead of
    `C_1` — a different path, so the already-trained Stage-0 run would not be
    found and would be retrained.
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def hpo_from_row(row) -> Dict[str, list]:
    """The pinned one-value HPO grid implied by a winning row."""
    return {
        "optimizer": [str(row["optimizer"])],
        "learning_rate": [float(row["learning_rate"])],
        "batch_size": [int(row["batch_size"])],
    }


def parse_scenarios(arg: str | None, default: Sequence[int] = (1, 2, 3, 4, 5)) -> List[int]:
    if not arg:
        return list(default)
    return [int(s) for s in arg.split(",")]
