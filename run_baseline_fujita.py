"""
FUJITA baseline HPO runner.

Trains FUJITA (Siamese NN) baseline model using the same dataset and output
conventions as `run_all_hpo.py` so downstream processing can be reused.

Usage:
    python run_baseline_fujita.py --scenario all --dsm-mode all
    python run_baseline_fujita.py --scenario 1 --dsm-mode dsm_only
"""

from __future__ import annotations

import os
import time
import random
import shutil
from typing import List, Dict

from run_all_hpo import (
    OPTIMIZERS,
    LEARNING_RATES,
    BATCH_SIZES,
    EPOCHS,
    RESULTS_ROOT,
    make_training_callbacks,
    get_optimizer,
    make_snn_dataset,
    make_snn_test_dataset,
    get_dsm_mode_tag,
    load_checkpoint,
    save_checkpoint,
    generate_dashboard,
    DSM_MODES,
    DATASETS,
)

# Use a separate results root for FUJITA to avoid colliding with the main HPO outputs
RESULTS_ROOT = os.path.join(RESULTS_ROOT + "_baseline_fujita")

from models.snn import load_dataset
from models.mmfemsnet import resolve_dsm_channel_indices
from utils.label_processing import prepare_split_for_scenario
from utils.metrics import compute_metrics
from run_all_hpo import save_classification_report

import argparse
import numpy as np
import pandas as pd

# Reproduce the same seeding technique as `run_all_hpo.py`
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import shutil
from baseline.models import fujita as baseline_fujita
from sklearn.metrics import fbeta_score
import json


def prepare_split_with_indices(X: np.ndarray, Y: np.ndarray, scenario: int):
    """Return processed split and source indices for a scenario."""
    if scenario == 1:
        mask = np.isin(Y, [0, 4])
        indices = np.flatnonzero(mask)
        X_proc = X[mask]
        Y_raw = Y[mask]
        mapping = {0: 0, 4: 1}
        Y_proc = np.array([mapping[int(y)] for y in Y_raw], dtype=np.int64)
        num_classes = 1
    elif scenario == 2:
        indices = np.arange(len(Y), dtype=np.int64)
        mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
        X_proc = X
        Y_proc = np.array([mapping[int(y)] for y in Y], dtype=np.int64)
        num_classes = 1
    elif scenario == 3:
        indices = np.arange(len(Y), dtype=np.int64)
        mapping = {int(c): int(c) for c in np.unique(Y)}
        X_proc = X
        Y_proc = Y.astype(np.int64)
        num_classes = len(mapping)
    else:
        raise ValueError("Scenario must be 1, 2, or 3.")
    return X_proc, Y_proc, indices, num_classes, mapping


def build_predictions_frame(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int, source_indices: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true), dtype=np.int64),
            "source_index": source_indices.astype(np.int64),
            "y_true": y_true.astype(np.int64),
            "y_pred": y_pred.astype(np.int64),
            "is_correct": (y_true == y_pred),
        }
    )
    if num_classes == 1:
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 1:
            prob_pos = prob_arr
            prob_neg = 1.0 - prob_pos
        elif prob_arr.ndim == 2 and prob_arr.shape[1] == 1:
            prob_pos = prob_arr[:, 0]
            prob_neg = 1.0 - prob_pos
        elif prob_arr.ndim == 2 and prob_arr.shape[1] >= 2:
            prob_neg = prob_arr[:, 0]
            prob_pos = prob_arr[:, 1]
        else:
            raise ValueError(f"Unsupported binary probability shape: {prob_arr.shape}")

        if len(prob_pos) != len(y_true):
            raise ValueError(f"Binary probability length mismatch: probs={len(prob_pos)} y_true={len(y_true)}")

        df["prob_class_0"] = prob_neg
        df["prob_class_1"] = prob_pos
        df["pred_confidence"] = np.maximum(prob_neg, prob_pos)
        df["pred_probability"] = prob_pos
    else:
        prob = np.asarray(y_prob)
        for class_idx in range(prob.shape[1]):
            df[f"prob_class_{class_idx}"] = prob[:, class_idx]
        df["pred_probability"] = prob[np.arange(len(y_pred)), y_pred]
        df["pred_confidence"] = prob.max(axis=1)
    return df


def compute_f2(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if num_classes == 1:
        return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    return float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))


def run_fujita_experiment(
    scenario: int,
    dataset_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    test_indices: np.ndarray,
    num_classes: int,
    optimizer_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    results_dir: str,
    all_results_list: List[Dict],
    include_density: bool,
    include_unc: bool,
):
    """Train and evaluate FUJITA baseline model."""
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"\n===== [FUJITA] Scenario {scenario} | {dataset_name} | OPT={optimizer_name} | LR={lr} | BS={batch_size} ====="
    )

    train_ds = make_snn_dataset(X_train, Y_train, batch_size, shuffle=True, include_density=include_density, include_unc=include_unc)
    val_ds = make_snn_dataset(X_val, Y_val, batch_size, shuffle=False, include_density=include_density, include_unc=include_unc)
    test_ds = make_snn_test_dataset(X_test, Y_test, batch_size=batch_size, include_density=include_density, include_unc=include_unc)

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(resolve_dsm_channel_indices(include_density, include_unc)[0])
    dsm_mode_tag = get_dsm_mode_tag(include_density, include_unc)
    input_shape = (H, W, dsm_channels)

    tf.random.set_seed(1234)
    # baseline_fujita.snn expects 2 to activate sigmoid+single-output for binary;
    # our pipeline uses num_classes=1 for binary, so remap here.
    fujita_n_class = 2 if num_classes == 1 else num_classes
    fmodel = baseline_fujita.snn(fujita_n_class, input_shape)
    model = fmodel.get_model()

    if num_classes == 1:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer=get_optimizer(optimizer_name, lr), loss=loss, metrics=["accuracy"])

    training_callbacks, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=results_dir, val_data=val_ds, num_classes=num_classes
    )

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=training_callbacks)

    if os.path.exists(best_weights_path):
        try:
            model.load_weights(best_weights_path)
        except Exception:
            pass

    train_time = time.time() - t0

    y_prob = model.predict(test_ds)
    if num_classes == 1:
        y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    else:
        y_pred = np.argmax(y_prob, axis=-1)

    y_true = np.concatenate([y for (_, y) in test_ds], axis=0)

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)

    inference_dir = os.path.join(results_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    inference_metrics_path = os.path.join(inference_dir, "inference_metrics.csv")

    y_prob_arr = np.asarray(y_prob)
    pred_df = build_predictions_frame(y_true, y_pred, y_prob_arr, num_classes, test_indices)
    predictions_path = os.path.join(inference_dir, "predictions.csv")
    pred_df.to_csv(predictions_path, index=False)

    save_classification_report(y_true, y_pred, os.path.join(inference_dir, "classification_report.txt"), os.path.join(inference_dir, "classification_report.csv"))

    macro_f2 = compute_f2(y_true, y_pred, num_classes)

    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": "FUJITA",
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": num_classes,
        "dsm_mode_tag": dsm_mode_tag,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "macro_f2": macro_f2,
        "mcc": mcc,
        "best_val_f1": f1_callback.best_val_f1,
        "best_val_epoch": f1_callback.best_epoch,
        "num_test_samples": int(len(y_true)),
        "checkpoint_path": str(best_weights_path) if best_weights_path else "",
        "inference_dir": inference_dir,
        "predictions_csv": predictions_path,
        "status": "ok",
    }

    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    pd.DataFrame([summary]).to_csv(inference_metrics_path, index=False)

    all_results_list.append(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all")
    parser.add_argument("--dsm-mode", type=str, default="all")
    args = parser.parse_args()

    if args.dsm_mode == "all":
        selected = DSM_MODES
    else:
        selected = [m for m in DSM_MODES if m["key"] == args.dsm_mode]

    if args.scenario == "all":
        scenarios = [1, 2, 3]
    else:
        scenarios = [int(args.scenario)]

    all_results = []

    try:
        for mode in selected:
            include_density = bool(mode["include_density"])
            include_unc = bool(mode["include_unc"])
            dsm_mode_tag = str(mode["tag"])

            for scenario in scenarios:
                for ds_name, npz_path in {'dataset_16': DATASETS['dataset_16']}.items():
                    print(f"Running FUJITA: scenario={scenario}, dataset={ds_name}, dsm_mode={dsm_mode_tag}")

                    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw = load_dataset(npz_path)

                    X_train_proc, Y_train_proc, num_classes, _ = prepare_split_for_scenario(X_train_raw, Y_train_raw, scenario)
                    X_val_proc, Y_val_proc, _, _ = prepare_split_for_scenario(X_val_raw, Y_val_raw, scenario)
                    X_test_proc, Y_test_proc, _, _ = prepare_split_for_scenario(X_test_raw, Y_test_raw, scenario)
                    _, _, test_indices, _, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)

                    # collect scenario-specific results so we can write a per-scenario summary
                    scenario_results: List[Dict] = []

                    completed = load_checkpoint()
                    for opt in OPTIMIZERS:
                        for lr in LEARNING_RATES:
                            for bs in BATCH_SIZES:
                                result_dir = os.path.join(
                                    RESULTS_ROOT,
                                    f"scenario_{scenario}",
                                    "FUJITA",
                                    ds_name,
                                    dsm_mode_tag,
                                    f"{opt}_lr{lr}_bs{bs}",
                                )
                                run_id = f"scenario={scenario}|dataset={ds_name}|model=FUJITA|dsm={dsm_mode_tag}|opt={opt}|lr={lr}|bs={bs}"
                                if run_id in completed:
                                    print(f"Skipping completed run: {run_id}")
                                    metrics_path = os.path.join(result_dir, "metrics.csv")
                                    if os.path.exists(metrics_path):
                                        try:
                                            df = pd.read_csv(metrics_path)
                                            all_results.extend(df.to_dict(orient="records"))
                                            scenario_results.extend(df.to_dict(orient="records"))
                                        except Exception:
                                            pass
                                    continue

                                try:
                                    run_fujita_experiment(
                                        scenario=scenario,
                                        dataset_name=ds_name,
                                        X_train=X_train_proc,
                                        Y_train=Y_train_proc,
                                        X_val=X_val_proc,
                                        Y_val=Y_val_proc,
                                        X_test=X_test_proc,
                                        test_indices=test_indices,
                                        Y_test=Y_test_proc,
                                        num_classes=num_classes,
                                        optimizer_name=opt,
                                        lr=lr,
                                        batch_size=bs,
                                        epochs=EPOCHS,
                                        results_dir=result_dir,
                                        all_results_list=scenario_results,
                                        include_density=include_density,
                                        include_unc=include_unc,
                                    )

                                    # append to global list and persist checkpoint
                                    all_results.extend(scenario_results[-1:])
                                    completed.add(run_id)
                                    save_checkpoint(completed)
                                except Exception:
                                    os.makedirs(result_dir, exist_ok=True)
                                    with open(os.path.join(result_dir, "error.log"), "a") as f:
                                        import traceback
                                        f.write(traceback.format_exc())
                                    print(f"FUJITA run failed: {run_id}")
                                    continue

                    # After finishing this dataset/scenario, write aggregated per-scenario CSV and copy predictions
                    out_root = os.path.join("inference_results_fujita", f"scenario_{scenario}")
                    os.makedirs(out_root, exist_ok=True)
                    try:
                        if scenario_results:
                            df = pd.DataFrame(scenario_results)
                            df.to_csv(os.path.join(out_root, "inference_summary_fujita.csv"), index=False)
                            df.to_html(os.path.join(out_root, "inference_summary_fujita.html"), index=False)

                            # collect predictions per run
                            preds_dir = os.path.join(out_root, "predictions")
                            os.makedirs(preds_dir, exist_ok=True)
                            for r in scenario_results:
                                ppath = r.get("predictions_csv")
                                if ppath and os.path.exists(ppath):
                                    # build safe filename
                                    safe_name = f"{r.get('model','FUJITA')}_{r.get('optimizer','')}_lr{r.get('learning_rate','')}_bs{r.get('batch_size','')}".replace("/","_").replace(' ','')
                                    safe_name = safe_name.replace('.', '_')
                                    dst = os.path.join(preds_dir, f"{safe_name}.csv")
                                    try:
                                        shutil.copy(ppath, dst)
                                    except Exception:
                                        pass
                    except Exception as e:
                        print("Failed to write per-scenario FUJITA summary:", e)
    finally:
        try:
            os.makedirs("inference_results", exist_ok=True)
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(os.path.join("inference_results", "inference_summary_fujita.csv"), index=False)
            summary_df.to_html(os.path.join("inference_results", "inference_summary_fujita.html"), index=False)
        except Exception as e:
            print("Failed to write FUJITA summary:", e)

        try:
            generate_dashboard(all_results, save_dir=RESULTS_ROOT)
        except Exception as e:
            print("Failed to generate dashboard:", e)


if __name__ == "__main__":
    main()


