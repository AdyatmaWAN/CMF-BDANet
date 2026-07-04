"""
FUJITA (baseline Siamese CNN) grid-search training entrypoint.

Same shape as train_fcsnn.py, driven by configs/fujita.json. FUJITA has no
residual/fusion ablation axis, only dsm_mode.

models.fujita.snn treats num_of_class in {1, 2} as binary and always builds
a 1-unit sigmoid head (this pipeline's convention represents binary
scenarios as num_classes == 1); anything else gets a softmax head. See
models/fujita.py for the self-check.

Scenario 5 is the ordinal-regression version of scenario 3's data (same 5
classes, no row filtering) - see utils/label_processing.py::is_ordinal_scenario
and models/ordinal.py's CORAL head for the mechanics, and the README's
"Ordinal regression (scenario 5)" section for the full explanation.

Usage:
    python train_fujita.py
    python train_fujita.py --config configs/fujita.json --scenario 1
    python train_fujita.py --scenario 5   # ordinal regression (CORAL)
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from typing import Dict

from utils.experiment import set_global_determinism

set_global_determinism()

import pandas as pd  # noqa: E402

from utils.experiment import (  # noqa: E402
    DSM_MODES,
    aggregate_scenarios_after_run,
    dsm_mode_channels,
    grid_search,
    make_siamese_dsm_dataset,
    set_tf_determinism,
    train_and_evaluate_nn,
)
from models import fujita as baseline_fujita  # noqa: E402
from models.fcsnn import load_dataset  # noqa: E402
from models.mmfemsnet import resolve_dsm_channel_indices  # noqa: E402
from models.ordinal import (  # noqa: E402
    coral_probs_to_class_probs,
    decode_coral_predictions,
    decode_coral_true_labels,
    ordinal_extra_metrics,
    ordinal_monitor_metric,
    ordinal_summary_log,
)
from utils.label_processing import is_ordinal_scenario, prepare_split_with_indices  # noqa: E402

set_tf_determinism()


def build_result_dir(
    results_root: str,
    scenario: int,
    dataset_name: str,
    *,
    dsm_mode: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    **_ignored,
) -> str:
    return os.path.join(
        results_root,
        f"scenario_{scenario}",
        "FUJITA",
        dataset_name,
        dsm_mode,
        f"{optimizer}_lr{learning_rate}_bs{batch_size}",
    )


def train_one(
    *,
    result_dir: str,
    dsm_mode: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    scenario: int,
    dataset_name: str,
    num_classes: int,
    seed: int,
    epochs: int,
    callback_cfg: Dict,
    X_train, Y_train, X_val, Y_val, X_test, Y_test, test_indices,
) -> Dict:
    include_density, include_unc = dsm_mode_channels(dsm_mode)
    ordinal = is_ordinal_scenario(scenario)

    print(
        f"\n===== [FUJITA] scenario={scenario} dataset={dataset_name} dsm={dsm_mode} "
        f"opt={optimizer} lr={learning_rate} bs={batch_size} ordinal={ordinal} ====="
    )

    train_ds = make_siamese_dsm_dataset(
        X_train, Y_train, batch_size, True, include_density, include_unc, seed,
        ordinal=ordinal, num_classes=num_classes,
    )
    val_ds = make_siamese_dsm_dataset(
        X_val, Y_val, batch_size, False, include_density, include_unc, seed,
        ordinal=ordinal, num_classes=num_classes,
    )
    test_ds = make_siamese_dsm_dataset(
        X_test, Y_test, batch_size, False, include_density, include_unc, seed,
        ordinal=ordinal, num_classes=num_classes,
    )

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(resolve_dsm_channel_indices(include_density, include_unc)[0])

    set_tf_determinism(seed)
    model = baseline_fujita.snn(num_classes, (H, W, dsm_channels), ordinal=ordinal).get_model()

    return train_and_evaluate_nn(
        result_dir=result_dir,
        model=model,
        model_label="FUJITA",
        num_classes=num_classes,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        test_indices=test_indices,
        epochs=epochs,
        callback_cfg=callback_cfg,
        loss="binary_crossentropy" if ordinal else None,
        decode_pred_fn=decode_coral_predictions if ordinal else None,
        decode_true_fn=decode_coral_true_labels if ordinal else None,
        class_probs_fn=coral_probs_to_class_probs if ordinal else None,
        extra_metrics_fn=ordinal_extra_metrics if ordinal else None,
        monitor_name="val_qwk" if ordinal else "val_f1",
        monitor_fn=ordinal_monitor_metric if ordinal else None,
        summary_log_fn=ordinal_summary_log if ordinal else None,
        extra_summary={
            "ordinal": ordinal,
            "scenario": scenario,
            "dataset": dataset_name,
            "dsm_mode_tag": dsm_mode,
            "batch_size": batch_size,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fujita.json")
    parser.add_argument("--scenario", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--dsm-mode", type=str, default=None, choices=[m["key"] for m in DSM_MODES])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    results_root = config["results_root"]
    seed = config.get("seed", 1234)
    epochs = config["epochs"]
    callback_cfg = config.get("callbacks", {})
    scenarios = [args.scenario] if args.scenario is not None else config["scenarios"]
    ablation_grid = dict(config["ablation_grid"])
    if args.dsm_mode is not None:
        ablation_grid["dsm_mode"] = [args.dsm_mode]

    all_results = []
    for scenario in scenarios:
        for dataset_name, npz_path in config["datasets"].items():
            print(f"\n{'=' * 70}\nScenario {scenario} | dataset {dataset_name}\n{'=' * 70}")
            (
                X_train_raw, Y_train_raw,
                X_val_raw, Y_val_raw,
                X_test_raw, Y_test_raw,
            ) = load_dataset(npz_path)

            X_train, Y_train, _, num_classes, _ = prepare_split_with_indices(X_train_raw, Y_train_raw, scenario)
            X_val, Y_val, _, _, _ = prepare_split_with_indices(X_val_raw, Y_val_raw, scenario)
            X_test, Y_test, test_indices, _, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)

            results = grid_search(
                ablation_grid=ablation_grid,
                hpo_grid=config["hpo_grid"],
                result_dir_fn=functools.partial(build_result_dir, results_root, scenario, dataset_name),
                train_one_fn=functools.partial(
                    train_one,
                    scenario=scenario,
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    seed=seed,
                    epochs=epochs,
                    callback_cfg=callback_cfg,
                    X_train=X_train, Y_train=Y_train,
                    X_val=X_val, Y_val=Y_val,
                    X_test=X_test, Y_test=Y_test,
                    test_indices=test_indices,
                ),
            )
            all_results.extend(results)

    if all_results:
        os.makedirs(results_root, exist_ok=True)
        pd.DataFrame(all_results).to_csv(os.path.join(results_root, "FUJITA_grid_summary.csv"), index=False)

    aggregate_scenarios_after_run(results_root, scenarios)


if __name__ == "__main__":
    main()
