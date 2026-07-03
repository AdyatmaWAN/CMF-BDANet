"""
Lightweight HPO runner for baseline models (FUJITA, Moya, Hajeb) using the
same dataset and output conventions as `run_all_hpo.py` so downstream
processing can be reused.

Usage examples:
    python run_baseline_hpo.py --scenario all --dsm-mode all
    python run_baseline_hpo.py --scenario 1 --dsm-mode dsm_only

Notes:
- FUJITA: trained like the SNN in `run_all_hpo.py` and HPO over optimizer / lr / bs
- Moya / Hajeb: use the NPZ dataset (selecting the DSM nDSM channel) and run
  a single SVM fit per scenario+DSM-mode (keeps runtime reasonable). Outputs
  metrics.csv and classification reports in the same schema as the main
  HPO script so the results can be processed nearly identically.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict

# Reuse helpers and constants from the main HPO script to ensure outputs match
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
)

from models.snn import load_dataset
from models.mmfemsnet import resolve_dsm_channel_indices, extract_inputs
from utils.label_processing import prepare_split_for_scenario
from utils.metrics import compute_metrics
from run_all_hpo import save_classification_report

import argparse
import numpy as np
import pandas as pd

import tensorflow as tf

# Baseline models
from baseline.models import fujita as baseline_fujita
from baseline.models import Moya as moya_module
from baseline.models import Hajeb as hajeb_module

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Reduced shared SVM search space for Moya and Hajeb (smaller, identical grids)
SVM_KERNELS = ["rbf", "linear"]
SVM_DEC_FUNCS = ["ovo", "ovr"]
SVM_CS = [1, 10]
SVM_GAMMAS = ["scale", "auto"]


from sklearn.metrics import fbeta_score
import json


def prepare_split_with_indices(X: np.ndarray, Y: np.ndarray, scenario: int):
    """Return processed split and source indices for a scenario (mirrors run_all_inference)."""
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
        # Accept either (N,), (N,1), or (N,2) probability-like inputs for binary runs.
        # Some SVM branches already provide [p0, p1], so avoid flattening to length 2N.
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
            raise ValueError(
                f"Binary probability length mismatch: probs={len(prob_pos)} y_true={len(y_true)}"
            )

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


def write_inference_summary(all_results: List[Dict], output_dir: str = "inference_results"):
    os.makedirs(output_dir, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_dir, "inference_summary.csv"), index=False)
    summary_df.to_html(os.path.join(output_dir, "inference_summary.html"), index=False)

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
    """Train and evaluate FUJITA baseline model using the same callbacks/metrics
    as in `run_all_hpo.py` so outputs are compatible.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"\n===== [FUJITA] Scenario {scenario} | {dataset_name} | OPT={optimizer_name} | LR={lr} | BS={batch_size} ====="
    )

    train_ds = make_snn_dataset(
        X_train,
        Y_train,
        batch_size,
        shuffle=True,
        include_density=include_density,
        include_unc=include_unc,
    )
    val_ds = make_snn_dataset(
        X_val,
        Y_val,
        batch_size,
        shuffle=False,
        include_density=include_density,
        include_unc=include_unc,
    )
    test_ds = make_snn_test_dataset(
        X_test,
        Y_test,
        batch_size=batch_size,
        include_density=include_density,
        include_unc=include_unc,
    )

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(resolve_dsm_channel_indices(include_density, include_unc)[0])
    dsm_mode_tag = get_dsm_mode_tag(include_density, include_unc)
    input_shape = (H, W, dsm_channels)

    # Build FUJITA model (their class is named `snn` and returns a two-input model)
    tf.random.set_seed(1234)
    fmodel = baseline_fujita.snn(num_classes, input_shape)
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

    # Save inference outputs so downstream inference pipeline is not required
    inference_dir = os.path.join(results_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    inference_metrics_path = os.path.join(inference_dir, "inference_metrics.csv")

    # y_prob -> array for build_predictions_frame
    y_prob_arr = np.asarray(y_prob)
    # Build and save per-sample predictions
    pred_df = build_predictions_frame(y_true, y_pred, y_prob_arr, num_classes, test_indices)
    predictions_path = os.path.join(inference_dir, "predictions.csv")
    pred_df.to_csv(predictions_path, index=False)

    # Save classification report with explicit labels
    labels = [0, 1] if num_classes == 1 else list(range(num_classes))
    save_classification_report(y_true, y_pred, os.path.join(inference_dir, "classification_report.txt"), os.path.join(inference_dir, "classification_report.csv"))

    # Compute F2
    macro_f2 = compute_f2(y_true, y_pred, num_classes)

    # Compose inference summary similar to run_all_inference
    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": "SNN",
        "residual": False,
        "fusion": "concat",
        "dsm_mode_tag": dsm_mode_tag,
        "dsm_post_concat_with_rgb": False,
        "use_four_stream": False,
        "num_classes": num_classes,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "macro_f2": macro_f2,
        "mcc": mcc,
        "cm00": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "cm01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
        "cm10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
        "cm11": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        "confusion_matrix_shape": f"{cm.shape[0]}x{cm.shape[1]}",
        "confusion_matrix_json": json.dumps(cm.tolist()),
        "num_test_samples": int(len(y_true)),
        "checkpoint_path": str(best_weights_path) if best_weights_path else "",
        "checkpoint_kind": "weights",
        "inference_dir": inference_dir,
        "predictions_csv": predictions_path,
        "inference_metrics_csv": inference_metrics_path,
        "status": "ok",
        "source_metrics_path": os.path.join(results_dir, "metrics.csv"),
    }

    # Save a copy of the original metrics.csv for compatibility
    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame([{
        "scenario": scenario,
        "dataset": dataset_name,
        "model": "FUJITA",
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": num_classes,
        "dsm_mode_tag": dsm_mode_tag,
        "dsm_post_concat_with_rgb": False,
        "use_four_stream": False,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "macro_f2": macro_f2,
        "best_val_f1": f1_callback.best_val_f1,
        "best_val_epoch": f1_callback.best_epoch,
        "monitor_metric": "val_f1",
        "mcc": mcc,
        "cm00": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "cm01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
        "cm10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
        "cm11": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        "train_time_sec": train_time,
        "result_dir": results_dir,
    }]).to_csv(metrics_path, index=False)

    # Save inference metrics file
    pd.DataFrame([summary]).to_csv(inference_metrics_path, index=False)

    all_results_list.append(summary)


def run_svm_baseline(
    name: str,
    feature_func,
    scenario: int,
    dataset_name: str,
    X_train_raw,
    Y_train_raw,
    X_val_raw,
    Y_val_raw,
    X_test_raw,
    Y_test_raw,
    num_classes: int,
    results_dir: str,
    dsm_mode_tag: str,
):
    """Run a simple SVM baseline using a feature extraction function that
    accepts X arrays shaped like the fused NPZ (N,H,W,C) and returns features.
    We select only the nDSM channel (first DSM channel) to keep behavior
    consistent across DSM mode variations and to keep inputs compatible with
    the original baseline feature extractors.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Extract DSM channels (use extract_inputs to respect DSM-mode channel mapping)
    # We'll pick the primary nDSM channel (channel 0 after extract_inputs)
    dsm_pre_train, dsm_post_train, _ = extract_inputs(X_train_raw, return_rgb_pre=False)
    dsm_pre_val, dsm_post_val, _ = extract_inputs(X_val_raw, return_rgb_pre=False)
    dsm_pre_test, dsm_post_test, _ = extract_inputs(X_test_raw, return_rgb_pre=False)

    # For compatibility with original baseline functions, collapse to single-channel
    X_train_pair = np.stack([dsm_pre_train[..., 0], dsm_post_train[..., 0]], axis=1)
    X_val_pair = np.stack([dsm_pre_val[..., 0], dsm_post_val[..., 0]], axis=1)
    X_test_pair = np.stack([dsm_pre_test[..., 0], dsm_post_test[..., 0]], axis=1)

    # Compute features using provided function (which expects shape (N,2,H,W) or similar)
    feat_train = feature_func(X_train_pair)
    feat_val = feature_func(X_val_pair)
    feat_test = feature_func(X_test_pair)

    # Standardize features
    scaler = StandardScaler()
    feat_train_s = scaler.fit_transform(feat_train)
    feat_val_s = scaler.transform(feat_val)
    feat_test_s = scaler.transform(feat_test)

    # Train a single SVM (keeps runtime reasonable); use balanced class weights
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=False)
    clf.fit(feat_train_s, Y_train_raw)

    pred = clf.predict(feat_test_s)

    # Compute metrics
    if num_classes == 1 or len(np.unique(Y_train_raw)) == 2:
        # binary
        acc, prec, rec, f1, mcc, cm = compute_metrics(Y_test_raw, pred, 1)
    else:
        acc, prec, rec, f1, mcc, cm = compute_metrics(Y_test_raw, pred, len(np.unique(Y_train_raw)))

    # Save metrics similar to other experiments
    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame([
        {
            "scenario": scenario,
            "dataset": dataset_name,
            "model": name,
            "optimizer": "SVM",
            "learning_rate": float('nan'),
            "batch_size": int(0),
            "epochs": int(0),
            "num_classes": len(np.unique(Y_train_raw)),
            "dsm_mode_tag": dsm_mode_tag,
            "dsm_post_concat_with_rgb": False,
            "use_four_stream": False,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": f1,
            "best_val_f1": float('nan'),
            "best_val_epoch": int(0),
            "monitor_metric": "f1",
            "mcc": mcc,
            "cm00": cm[0][0] if cm.shape == (2,2) else 0,
            "cm01": cm[0][1] if cm.shape == (2,2) else 0,
            "cm10": cm[1][0] if cm.shape == (2,2) else 0,
            "cm11": cm[1][1] if cm.shape == (2,2) else 0,
            "train_time_sec": 0.0,
            "result_dir": results_dir,
        }
    ]).to_csv(metrics_path, index=False)

    # classification report
    cr_txt = os.path.join(results_dir, "classification_report.txt")
    cr_csv = os.path.join(results_dir, "classification_report.csv")
    # reuse run_all_hpo helper
    save_classification_report(Y_test_raw, pred, cr_txt, cr_csv)


def run_moya_grid(
    scenario: int,
    ds_name: str,
    dsm_mode_tag: str,
    X_train_pair,
    Y_train,
    X_test_pair,
    Y_test,
    test_indices: np.ndarray,
    results_root: str,
    num_classes: int,
):
    """Run Moya's original grid search over SVM hyperparameters and save per-config results.
    Uses `moya_module.feature_difference` for feature extraction.
    """
    # Reduced param grids (shared with Hajeb)
    kernels = SVM_KERNELS
    dec_funcs = SVM_DEC_FUNCS
    slacks = SVM_CS
    gammas = SVM_GAMMAS

    # Precompute features
    feat_train = moya_module.feature_difference(X_train_pair.copy())
    feat_test = moya_module.feature_difference(X_test_pair.copy())

    # class weights
    from sklearn.utils import class_weight

    weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    d_class_weights = dict(enumerate(weight))

    best_fm = -99999
    best_model = None
    results: list[Dict] = []

    for func in dec_funcs:
        for kern in kernels:
            for c in slacks:
                for g in gammas:
                    run_tag = f"dec_{func}_kernel_{kern}_C_{c}_gamma_{g}"
                    result_dir = os.path.join(results_root, f"Moya", ds_name, dsm_mode_tag, run_tag)
                    os.makedirs(result_dir, exist_ok=True)

                    run_id = f"scenario={scenario}|dataset={ds_name}|model=Moya|dsm={dsm_mode_tag}|dec={func}|kernel={kern}|C={c}|gamma={g}"
                    completed = load_checkpoint()
                    if run_id in completed:
                        # try to recover metrics and include in returned list
                        metrics_path = os.path.join(result_dir, "metrics.csv")
                        if os.path.exists(metrics_path):
                            try:
                                df = pd.read_csv(metrics_path)
                                results.extend(df.to_dict(orient="records"))
                            except Exception:
                                pass
                        continue

                    clf = SVC(decision_function_shape=func, kernel=kern, C=c, gamma=g, class_weight=d_class_weights)
                    clf.fit(feat_train, Y_train)
                    pred = clf.predict(feat_test)

                    # decision scores -> convert to probabilistic-like outputs
                    if hasattr(clf, "decision_function"):
                        scores = clf.decision_function(feat_test)
                        if scores is None:
                            probs = None
                        else:
                            scores = np.asarray(scores)
                            if scores.ndim == 1:
                                # binary scores -> sigmoid
                                prob_pos = 1.0 / (1.0 + np.exp(-scores))
                                probs = np.vstack([1.0 - prob_pos, prob_pos]).T
                            else:
                                # multiclass scores -> softmax
                                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                                probs = exps / np.sum(exps, axis=1, keepdims=True)
                    elif hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(feat_test)
                    else:
                        probs = None

                    # compute metrics
                    if num_classes == 1 or len(np.unique(Y_train)) == 2:
                        acc, prec, rec, fm, mcc, cm = compute_metrics(Y_test, pred, 1)
                    else:
                        acc, prec, rec, fm, mcc, cm = compute_metrics(Y_test, pred, len(np.unique(Y_train)))
                    # Save per-config metrics and inference outputs
                    inference_dir = os.path.join(result_dir, "inference")
                    os.makedirs(inference_dir, exist_ok=True)

                    metrics_path = os.path.join(result_dir, "metrics.csv")

                    # ensure we have probability-like outputs
                    if probs is None:
                        # fallback: create one-hot from predictions
                        if num_classes == 1:
                            probs = np.vstack([1 - pred, pred]).T
                        else:
                            probs = np.zeros((len(pred), len(np.unique(Y_train))), dtype=float)
                            for i, p in enumerate(pred):
                                probs[i, int(p)] = 1.0

                    pred_df = build_predictions_frame(np.asarray(Y_test).astype(int), np.asarray(pred).astype(int), probs, num_classes, test_indices)
                    predictions_path = os.path.join(inference_dir, "predictions.csv")
                    pred_df.to_csv(predictions_path, index=False)

                    # compute f2
                    macro_f2 = compute_f2(np.asarray(Y_test).astype(int), np.asarray(pred).astype(int), num_classes)

                    summary = {
                        "scenario": scenario,
                        "dataset": ds_name,
                        "model": "Moya",
                        "residual": False,
                        "fusion": "concat",
                        "dsm_mode_tag": dsm_mode_tag,
                        "dsm_post_concat_with_rgb": False,
                        "use_four_stream": False,
                        "num_classes": len(np.unique(Y_train)),
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "macro_f1": fm,
                        "macro_f2": macro_f2,
                        "mcc": mcc,
                        "cm00": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                        "cm01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
                        "cm10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
                        "cm11": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
                        "confusion_matrix_shape": f"{cm.shape[0]}x{cm.shape[1]}",
                        "confusion_matrix_json": json.dumps(cm.tolist()),
                        "num_test_samples": int(len(Y_test)),
                        "checkpoint_path": "",
                        "checkpoint_kind": "svm",
                        "inference_dir": inference_dir,
                        "predictions_csv": predictions_path,
                        "inference_metrics_csv": os.path.join(inference_dir, "inference_metrics.csv"),
                        "status": "ok",
                        "source_metrics_path": metrics_path,
                    }

                    # Sanitize summary values so pandas receives only scalars/strings
                    for kk, vv in list(summary.items()):
                        if isinstance(vv, (np.ndarray, list, tuple)):
                            try:
                                summary[kk] = json.dumps(np.asarray(vv).tolist())
                            except Exception:
                                summary[kk] = str(vv)
                        elif isinstance(vv, (np.generic,)):
                            try:
                                summary[kk] = vv.item()
                            except Exception:
                                summary[kk] = float(vv)

                    # Write summary and metrics
                    metrics_path = os.path.join(result_dir, "metrics.csv")
                    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
                    pd.DataFrame([summary]).to_csv(os.path.join(inference_dir, "inference_metrics.csv"), index=False)
                    results.append(summary)

                    # classification report
                    save_classification_report(Y_test, pred, os.path.join(inference_dir, "classification_report.txt"), os.path.join(inference_dir, "classification_report.csv"))

                    # checkpoint
                    completed.add(run_id)
                    save_checkpoint(completed)

                    # track best model (by F1)
                    if fm > best_fm:
                        best_fm = fm
                        best_model = clf

    # save best model
    if best_model is not None:
        os.makedirs(os.path.join("saved_model", "Moya"), exist_ok=True)
        import pickle

        with open(os.path.join("saved_model", "Moya", f"best_{best_fm}_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
    return results


def run_hajeb_grid(
    scenario: int,
    ds_name: str,
    dsm_mode_tag: str,
    X_train_pair,
    Y_train,
    X_test_pair,
    Y_test,
    test_indices: np.ndarray,
    results_root: str,
    num_classes: int,
):
    """Run Hajeb's original grid search over SVM hyperparameters and save per-config results.
    Uses `hajeb_module.dsm_difference` as in the original script.
    """
    # Reduced param grids (shared with Moya)
    kernels = SVM_KERNELS
    dec_funcs = SVM_DEC_FUNCS
    slacks = SVM_CS
    gammas = SVM_GAMMAS

    feat_train = hajeb_module.dsm_difference(X_train_pair.copy())
    feat_test = hajeb_module.dsm_difference(X_test_pair.copy())

    from sklearn.utils import class_weight
    weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    d_class_weights = dict(enumerate(weight))

    best_fm = -99999
    best_model = None
    results: list[Dict] = []

    for func in dec_funcs:
        for kern in kernels:
            for c in slacks:
                for g in gammas:
                    run_tag = f"dec_{func}_kernel_{kern}_C_{c}_gamma_{g}"
                    result_dir = os.path.join(results_root, f"Hajeb", ds_name, dsm_mode_tag, run_tag)
                    os.makedirs(result_dir, exist_ok=True)

                    run_id = f"scenario={scenario}|dataset={ds_name}|model=Hajeb|dsm={dsm_mode_tag}|dec={func}|kernel={kern}|C={c}|gamma={g}"
                    completed = load_checkpoint()
                    if run_id in completed:
                        metrics_path = os.path.join(result_dir, "metrics.csv")
                        if os.path.exists(metrics_path):
                            try:
                                df = pd.read_csv(metrics_path)
                                results.extend(df.to_dict(orient="records"))
                            except Exception:
                                pass
                        continue

                    clf = SVC(decision_function_shape=func, kernel=kern, C=c, gamma=g, class_weight=d_class_weights)
                    clf.fit(feat_train, Y_train)
                    pred = clf.predict(feat_test)

                    # decision scores -> convert to probabilistic-like outputs
                    if hasattr(clf, "decision_function"):
                        scores = clf.decision_function(feat_test)
                        if scores is None:
                            probs = None
                        else:
                            scores = np.asarray(scores)
                            if scores.ndim == 1:
                                # binary scores -> sigmoid
                                prob_pos = 1.0 / (1.0 + np.exp(-scores))
                                probs = np.vstack([1.0 - prob_pos, prob_pos]).T
                            else:
                                # multiclass scores -> softmax
                                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                                probs = exps / np.sum(exps, axis=1, keepdims=True)
                    elif hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(feat_test)
                    else:
                        probs = None

                    if num_classes == 1 or len(np.unique(Y_train)) == 2:
                        acc, prec, rec, fm, mcc, cm = compute_metrics(Y_test, pred, 1)
                    else:
                        acc, prec, rec, fm, mcc, cm = compute_metrics(Y_test, pred, len(np.unique(Y_train)))

                    inference_dir = os.path.join(result_dir, "inference")
                    os.makedirs(inference_dir, exist_ok=True)

                    metrics_path = os.path.join(result_dir, "metrics.csv")

                    if probs is None:
                        if num_classes == 1:
                            probs = np.vstack([1 - pred, pred]).T
                        else:
                            probs = np.zeros((len(pred), len(np.unique(Y_train))), dtype=float)
                            for i, p in enumerate(pred):
                                probs[i, int(p)] = 1.0

                    pred_df = build_predictions_frame(np.asarray(Y_test).astype(int), np.asarray(pred).astype(int), probs, num_classes, test_indices)
                    predictions_path = os.path.join(inference_dir, "predictions.csv")
                    pred_df.to_csv(predictions_path, index=False)

                    macro_f2 = compute_f2(np.asarray(Y_test).astype(int), np.asarray(pred).astype(int), num_classes)

                    summary = {
                        "scenario": scenario,
                        "dataset": ds_name,
                        "model": "Hajeb",
                        "residual": False,
                        "fusion": "concat",
                        "dsm_mode_tag": dsm_mode_tag,
                        "dsm_post_concat_with_rgb": False,
                        "use_four_stream": False,
                        "num_classes": len(np.unique(Y_train)),
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "macro_f1": fm,
                        "macro_f2": macro_f2,
                        "mcc": mcc,
                        "cm00": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                        "cm01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
                        "cm10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
                        "cm11": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
                        "confusion_matrix_shape": f"{cm.shape[0]}x{cm.shape[1]}",
                        "confusion_matrix_json": json.dumps(cm.tolist()),
                        "num_test_samples": int(len(Y_test)),
                        "checkpoint_path": "",
                        "checkpoint_kind": "svm",
                        "inference_dir": inference_dir,
                        "predictions_csv": predictions_path,
                        "inference_metrics_csv": os.path.join(inference_dir, "inference_metrics.csv"),
                        "status": "ok",
                        "source_metrics_path": metrics_path,
                    }

                    # Sanitize summary values so pandas receives only scalars/strings
                    for kk, vv in list(summary.items()):
                        if isinstance(vv, (np.ndarray, list, tuple)):
                            try:
                                summary[kk] = json.dumps(np.asarray(vv).tolist())
                            except Exception:
                                summary[kk] = str(vv)
                        elif isinstance(vv, (np.generic,)):
                            try:
                                summary[kk] = vv.item()
                            except Exception:
                                summary[kk] = float(vv)

                    metrics_path = os.path.join(result_dir, "metrics.csv")
                    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
                    pd.DataFrame([summary]).to_csv(os.path.join(inference_dir, "inference_metrics.csv"), index=False)
                    results.append(summary)

                    save_classification_report(Y_test, pred, os.path.join(inference_dir, "classification_report.txt"), os.path.join(inference_dir, "classification_report.csv"))

                    completed.add(run_id)
                    save_checkpoint(completed)

                    if fm > best_fm:
                        best_fm = fm
                        best_model = clf

    if best_model is not None:
        os.makedirs(os.path.join("saved_model", "Hajeb"), exist_ok=True)
        import pickle

        with open(os.path.join("saved_model", "Hajeb", f"best_{best_fm}_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=str, default="all")
    p.add_argument("--dsm-mode", type=str, default="all")
    return p.parse_args()


def main():
    args = parse_args()

    # Copy DSM modes from run_all_hpo to preserve tags
    from run_all_hpo import DSM_MODES, DATASETS

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
                    print(f"Running baseline models: scenario={scenario}, dataset={ds_name}, dsm_mode={dsm_mode_tag}")

                    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw = load_dataset(npz_path)

                    X_train_proc, Y_train_proc, num_classes, _ = prepare_split_for_scenario(X_train_raw, Y_train_raw, scenario)
                    X_val_proc, Y_val_proc, _, _ = prepare_split_for_scenario(X_val_raw, Y_val_raw, scenario)
                    X_test_proc, Y_test_proc, _, _ = prepare_split_for_scenario(X_test_raw, Y_test_raw, scenario)
                    # get source indices for test split so we can save per-instance predictions
                    _, _, test_indices, _, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)

                    # ----------------- FUJITA HPO -----------------
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
                                        all_results_list=all_results,
                                        include_density=include_density,
                                        include_unc=include_unc,
                                    )

                                    completed.add(run_id)
                                    save_checkpoint(completed)
                                except Exception:
                                    os.makedirs(result_dir, exist_ok=True)
                                    with open(os.path.join(result_dir, "error.log"), "a") as f:
                                        import traceback

                                        f.write(traceback.format_exc())
                                    print(f"FUJITA run failed: {run_id}")
                                    continue

                    # ----------------- MOYA / HAJEB (SVM baselines) -----------------
                    # We'll run one SVM per baseline per (scenario, dsm_mode)
                    # Run full grid for MOYA and HAJEB using original feature computations
                    try:
                        # Use the processed splits (X_train_proc/X_test_proc) so features align with Y_train/Y_test
                        dsm_pre_train, dsm_post_train, _ = extract_inputs(X_train_proc, return_rgb_pre=False)
                        dsm_pre_test, dsm_post_test, _ = extract_inputs(X_test_proc, return_rgb_pre=False)
                        X_train_pair = np.stack([dsm_pre_train[..., 0], dsm_post_train[..., 0]], axis=1)
                        X_test_pair = np.stack([dsm_pre_test[..., 0], dsm_post_test[..., 0]], axis=1)

                        moya_results = run_moya_grid(
                            scenario=scenario,
                            ds_name=ds_name,
                            dsm_mode_tag=dsm_mode_tag,
                            X_train_pair=X_train_pair,
                            Y_train=Y_train_proc,
                            X_test_pair=X_test_pair,
                            Y_test=Y_test_proc,
                            test_indices=test_indices,
                            results_root=os.path.join(RESULTS_ROOT, f"scenario_{scenario}"),
                            num_classes=num_classes,
                        )
                        all_results.extend(moya_results)
                    except Exception as e:
                        print("Moya full grid failed:", e)

                    try:
                        # Hajeb uses DSM difference features; use processed splits to align counts
                        dsm_pre_train, dsm_post_train, _ = extract_inputs(X_train_proc, return_rgb_pre=False)
                        dsm_pre_test, dsm_post_test, _ = extract_inputs(X_test_proc, return_rgb_pre=False)
                        X_train_pair = np.stack([dsm_pre_train[..., 0], dsm_post_train[..., 0]], axis=1)
                        X_test_pair = np.stack([dsm_pre_test[..., 0], dsm_post_test[..., 0]], axis=1)

                        hajeb_results = run_hajeb_grid(
                            scenario=scenario,
                            ds_name=ds_name,
                            dsm_mode_tag=dsm_mode_tag,
                            X_train_pair=X_train_pair,
                            Y_train=Y_train_proc,
                            X_test_pair=X_test_pair,
                            Y_test=Y_test_proc,
                            test_indices=test_indices,
                            results_root=os.path.join(RESULTS_ROOT, f"scenario_{scenario}"),
                            num_classes=num_classes,
                        )
                        all_results.extend(hajeb_results)
                    except Exception as e:
                        print("Hajeb full grid failed:", e)
    finally:
        # Always write the summary, even if one run fails or dashboard generation errors.
        try:
            write_inference_summary(all_results)
        except Exception as e:
            print("Failed to write inference summary:", e)

        try:
            generate_dashboard(all_results, save_dir=RESULTS_ROOT)
        except Exception as e:
            print("Failed to generate dashboard:", e)


if __name__ == "__main__":
    main()




