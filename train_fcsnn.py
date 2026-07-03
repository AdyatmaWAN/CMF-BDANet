"""
FCSNN grid-search training entrypoint.

Reads hyperparameter/ablation grids from a JSON config (default
configs/fcsnn.json). For every combination it trains, evaluates on the test
split, and writes metrics.csv + predictions.csv + a classification report +
model artifacts into a config-derived result directory. Runs whose
metrics.csv already exists are skipped automatically (see
utils/experiment.py::grid_search), so re-running this script resumes where
it left off.

Usage:
    python train_fcsnn.py
    python train_fcsnn.py --config configs/fcsnn.json --scenario 1
    python train_fcsnn.py --dsm-mode dsm_only
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from typing import Dict

import numpy as np

from utils.experiment import set_global_determinism

set_global_determinism()

import pandas as pd  # noqa: E402

from utils.experiment import (  # noqa: E402
    DSM_MODES,
    build_predictions_frame,
    compute_f2,
    dsm_mode_channels,
    get_optimizer,
    grid_search,
    make_siamese_dsm_dataset as make_fcsnn_dataset,
    make_training_callbacks,
    save_classification_report,
    set_tf_determinism,
)
from models.fcsnn import FCSNN, load_dataset  # noqa: E402
from models.mmfemsnet import resolve_dsm_channel_indices  # noqa: E402
from utils.label_processing import prepare_split_with_indices  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402

set_tf_determinism()


def build_result_dir(
    results_root: str,
    scenario: int,
    dataset_name: str,
    *,
    dsm_mode: str,
    residual: bool,
    fusion: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    **_ignored,
) -> str:
    return os.path.join(
        results_root,
        f"scenario_{scenario}",
        "FCSNN",
        dataset_name,
        dsm_mode,
        f"residual_{residual}",
        f"fusion_{fusion}",
        f"{optimizer}_lr{learning_rate}_bs{batch_size}",
    )


def train_one(
    *,
    result_dir: str,
    dsm_mode: str,
    residual: bool,
    fusion: str,
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
    os.makedirs(result_dir, exist_ok=True)
    include_density, include_unc = dsm_mode_channels(dsm_mode)

    print(
        f"\n===== [FCSNN] scenario={scenario} dataset={dataset_name} dsm={dsm_mode} "
        f"residual={residual} fusion={fusion} opt={optimizer} lr={learning_rate} bs={batch_size} ====="
    )

    train_ds = make_fcsnn_dataset(X_train, Y_train, batch_size, True, include_density, include_unc, seed)
    val_ds = make_fcsnn_dataset(X_val, Y_val, batch_size, False, include_density, include_unc, seed)
    test_ds = make_fcsnn_dataset(X_test, Y_test, batch_size, False, include_density, include_unc, seed)

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(resolve_dsm_channel_indices(include_density, include_unc)[0])

    set_tf_determinism(seed)
    model = FCSNN(
        num_of_class=num_classes,
        residual=residual,
        dropout=True,
        dense=True,
        num_of_layer=3,
        input_shape=(H, W, dsm_channels),
        substraction=True,
        shared=True,
        fusion=fusion,
    ).get_model()

    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer=get_optimizer(optimizer, learning_rate), loss=loss, metrics=["accuracy"])

    training_callbacks, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=result_dir, val_data=val_ds, num_classes=num_classes, **callback_cfg
    )

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=training_callbacks)
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
    train_time = time.time() - t0

    y_prob = model.predict(test_ds)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1) if num_classes == 1 else np.argmax(y_prob, axis=-1)
    y_true = np.concatenate([y for (_, y) in test_ds], axis=0)

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)
    macro_f2 = compute_f2(y_true, y_pred, num_classes)

    pred_df = build_predictions_frame(y_true, y_pred, y_prob, num_classes, test_indices)
    pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)
    save_classification_report(
        y_true, y_pred,
        os.path.join(result_dir, "classification_report.txt"),
        os.path.join(result_dir, "classification_report.csv"),
    )

    model.save(os.path.join(result_dir, "model.keras"))
    model.save_weights(os.path.join(result_dir, "weights.h5"))

    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": "FCSNN",
        "dsm_mode_tag": dsm_mode,
        "residual": residual,
        "fusion": fusion,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": num_classes,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "macro_f2": macro_f2,
        "mcc": mcc,
        "cm00": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "cm01": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "cm10": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "cm11": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        "confusion_matrix_json": json.dumps(cm.tolist()),
        "best_val_f1": f1_callback.best_val_f1,
        "best_val_epoch": f1_callback.best_epoch,
        "monitor_metric": "val_f1",
        "train_time_sec": train_time,
        "num_test_samples": int(len(y_true)),
        "predictions_csv": os.path.join(result_dir, "predictions.csv"),
        "result_dir": result_dir,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

    print(f"[FCSNN] scenario={scenario} ACC={acc:.4f} F1={f1:.4f} MCC={mcc:.4f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fcsnn.json")
    parser.add_argument("--scenario", type=int, default=None, choices=[1, 2, 3])
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
        pd.DataFrame(all_results).to_csv(os.path.join(results_root, "FCSNN_grid_summary.csv"), index=False)


if __name__ == "__main__":
    main()
