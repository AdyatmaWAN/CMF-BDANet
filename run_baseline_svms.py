"""
Moya and Hajeb SVM baseline HPO runner.

Runs SVM grid search over hyperparameters for both Moya and Hajeb baseline models
using the same dataset and output conventions as `run_all_hpo.py`.

Usage:
    python run_baseline_svms.py --scenario all --dsm-mode all
    python run_baseline_svms.py --scenario 1 --dsm-mode dsm_only
"""

from __future__ import annotations

import os
import random
import shutil
from typing import List, Dict

from run_all_hpo import (
    RESULTS_ROOT,
    load_checkpoint,
    save_checkpoint,
    generate_dashboard,
    DSM_MODES,
    DATASETS,
)

# Use a separate results root for the SVM baselines so we don't overwrite main HPO outputs
RESULTS_ROOT = os.path.join(RESULTS_ROOT + "_baseline_svms")

from models.snn import load_dataset
from models.mmfemsnet import extract_inputs
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
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    # tensorflow is optional for SVM runner; ignore if not available
    pass

import shutil

from baseline.models import Moya as moya_module
from baseline.models import Hajeb as hajeb_module
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import fbeta_score
import json


# SVM search space
SVM_KERNELS = ["rbf", "linear"]
SVM_DEC_FUNCS = ["ovo", "ovr"]
SVM_CS = [1, 10]
SVM_GAMMAS = ["scale", "auto"]


def build_moya_features(dsm_pre: np.ndarray, dsm_post: np.ndarray) -> np.ndarray:
    """Apply Moya feature_difference per DSM channel and concatenate.

    dsm_pre / dsm_post: (N, H, W, C) — one entry per channel.
    Returns (N, 3*C) where 3 features per channel = [mean_diff, std, corr].
    """
    n_channels = dsm_pre.shape[-1]
    parts = []
    for c in range(n_channels):
        pair = np.stack([dsm_pre[..., c], dsm_post[..., c]], axis=1)  # (N, 2, H, W)
        parts.append(moya_module.feature_difference(pair))
    return np.concatenate(parts, axis=1)


def build_hajeb_features(dsm_pre: np.ndarray, dsm_post: np.ndarray) -> np.ndarray:
    """Apply Hajeb dsm_difference per DSM channel and concatenate.

    dsm_pre / dsm_post: (N, H, W, C).
    Returns (N, H*W*C) flattened difference per channel.
    """
    n_channels = dsm_pre.shape[-1]
    parts = []
    for c in range(n_channels):
        pair = np.stack([dsm_pre[..., c], dsm_post[..., c]], axis=1)  # (N, 2, H, W)
        parts.append(hajeb_module.dsm_difference(pair))
    return np.concatenate(parts, axis=1)


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


def run_moya_grid(scenario: int, ds_name: str, dsm_mode_tag: str, feat_train: np.ndarray, Y_train, feat_test: np.ndarray, Y_test, test_indices: np.ndarray, results_root: str, num_classes: int):
    """Run Moya SVM grid search."""
    kernels = SVM_KERNELS
    dec_funcs = SVM_DEC_FUNCS
    slacks = SVM_CS
    gammas = SVM_GAMMAS

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

                    if hasattr(clf, "decision_function"):
                        scores = clf.decision_function(feat_test)
                        if scores is not None:
                            scores = np.asarray(scores)
                            if scores.ndim == 1:
                                prob_pos = 1.0 / (1.0 + np.exp(-scores))
                                probs = np.vstack([1.0 - prob_pos, prob_pos]).T
                            else:
                                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                                probs = exps / np.sum(exps, axis=1, keepdims=True)
                        else:
                            probs = None
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
                        "model": "Moya",
                        "num_classes": len(np.unique(Y_train)),
                        "dsm_mode_tag": dsm_mode_tag,
                        "dec_func": func,
                        "kernel": kern,
                        "C": c,
                        "gamma": g,
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
                        "num_test_samples": int(len(Y_test)),
                        "confusion_matrix_json": json.dumps(cm.tolist()),
                        "inference_dir": inference_dir,
                        "predictions_csv": predictions_path,
                        "status": "ok",
                    }

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
        os.makedirs(os.path.join("saved_model", "Moya"), exist_ok=True)
        import pickle
        with open(os.path.join("saved_model", "Moya", f"best_{best_fm}_model.pkl"), "wb") as f:
            pickle.dump(best_model, f)
    return results


def run_hajeb_grid(scenario: int, ds_name: str, dsm_mode_tag: str, feat_train: np.ndarray, Y_train, feat_test: np.ndarray, Y_test, test_indices: np.ndarray, results_root: str, num_classes: int):
    """Run Hajeb SVM grid search."""
    kernels = SVM_KERNELS
    dec_funcs = SVM_DEC_FUNCS
    slacks = SVM_CS
    gammas = SVM_GAMMAS

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

                    if hasattr(clf, "decision_function"):
                        scores = clf.decision_function(feat_test)
                        if scores is not None:
                            scores = np.asarray(scores)
                            if scores.ndim == 1:
                                prob_pos = 1.0 / (1.0 + np.exp(-scores))
                                probs = np.vstack([1.0 - prob_pos, prob_pos]).T
                            else:
                                exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                                probs = exps / np.sum(exps, axis=1, keepdims=True)
                        else:
                            probs = None
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
                        "num_classes": len(np.unique(Y_train)),
                        "dsm_mode_tag": dsm_mode_tag,
                        "dec_func": func,
                        "kernel": kern,
                        "C": c,
                        "gamma": g,
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
                        "num_test_samples": int(len(Y_test)),
                        "confusion_matrix_json": json.dumps(cm.tolist()),
                        "inference_dir": inference_dir,
                        "predictions_csv": predictions_path,
                        "status": "ok",
                    }

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
                    print(f"Running Moya/Hajeb: scenario={scenario}, dataset={ds_name}, dsm_mode={dsm_mode_tag}")

                    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw = load_dataset(npz_path)

                    X_train_proc, Y_train_proc, num_classes, _ = prepare_split_for_scenario(X_train_raw, Y_train_raw, scenario)
                    X_val_proc, Y_val_proc, _, _ = prepare_split_for_scenario(X_val_raw, Y_val_raw, scenario)
                    X_test_proc, Y_test_proc, _, _ = prepare_split_for_scenario(X_test_raw, Y_test_raw, scenario)
                    _, _, test_indices, _, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)

                    # collect scenario results so we can write per-scenario aggregation
                    scenario_results: List[Dict] = []

                    try:
                        dsm_pre_train, dsm_post_train, _ = extract_inputs(X_train_proc, include_density=include_density, include_unc=include_unc, return_rgb_pre=False)
                        dsm_pre_test, dsm_post_test, _ = extract_inputs(X_test_proc, include_density=include_density, include_unc=include_unc, return_rgb_pre=False)
                        moya_feat_train = build_moya_features(dsm_pre_train, dsm_post_train)
                        moya_feat_test = build_moya_features(dsm_pre_test, dsm_post_test)

                        moya_results = run_moya_grid(
                            scenario=scenario,
                            ds_name=ds_name,
                            dsm_mode_tag=dsm_mode_tag,
                            feat_train=moya_feat_train,
                            Y_train=Y_train_proc,
                            feat_test=moya_feat_test,
                            Y_test=Y_test_proc,
                            test_indices=test_indices,
                            results_root=os.path.join(RESULTS_ROOT, f"scenario_{scenario}"),
                            num_classes=num_classes,
                        )
                        all_results.extend(moya_results)
                        scenario_results.extend(moya_results)
                    except Exception as e:
                        print("Moya grid failed:", e)

                    try:
                        dsm_pre_train, dsm_post_train, _ = extract_inputs(X_train_proc, include_density=include_density, include_unc=include_unc, return_rgb_pre=False)
                        dsm_pre_test, dsm_post_test, _ = extract_inputs(X_test_proc, include_density=include_density, include_unc=include_unc, return_rgb_pre=False)
                        hajeb_feat_train = build_hajeb_features(dsm_pre_train, dsm_post_train)
                        hajeb_feat_test = build_hajeb_features(dsm_pre_test, dsm_post_test)

                        hajeb_results = run_hajeb_grid(
                            scenario=scenario,
                            ds_name=ds_name,
                            dsm_mode_tag=dsm_mode_tag,
                            feat_train=hajeb_feat_train,
                            Y_train=Y_train_proc,
                            feat_test=hajeb_feat_test,
                            Y_test=Y_test_proc,
                            test_indices=test_indices,
                            results_root=os.path.join(RESULTS_ROOT, f"scenario_{scenario}"),
                            num_classes=num_classes,
                        )
                        all_results.extend(hajeb_results)
                        scenario_results.extend(hajeb_results)
                    except Exception as e:
                        print("Hajeb grid failed:", e)

                    # write per-scenario aggregation and copy predictions
                    out_root = os.path.join("inference_results_svms", f"scenario_{scenario}")
                    os.makedirs(out_root, exist_ok=True)
                    try:
                        if scenario_results:
                            df = pd.DataFrame(scenario_results)
                            df.to_csv(os.path.join(out_root, "inference_summary_svms.csv"), index=False)
                            df.to_html(os.path.join(out_root, "inference_summary_svms.html"), index=False)

                            preds_dir = os.path.join(out_root, "predictions")
                            os.makedirs(preds_dir, exist_ok=True)
                            for r in scenario_results:
                                ppath = r.get("predictions_csv")
                                if ppath and os.path.exists(ppath):
                                    # build safe filename including grid params
                                    safe_name = f"{r.get('model')}_{r.get('dec_func','')}_{r.get('kernel','')}_C{r.get('C','')}_g{r.get('gamma','')}".replace(' ', '')
                                    safe_name = safe_name.replace('.', '_')
                                    dst = os.path.join(preds_dir, f"{safe_name}.csv")
                                    try:
                                        shutil.copy(ppath, dst)
                                    except Exception:
                                        pass
                    except Exception as e:
                        print("Failed to write per-scenario SVM summary:", e)
    finally:
        try:
            os.makedirs("inference_results", exist_ok=True)
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(os.path.join("inference_results", "inference_summary_svms.csv"), index=False)
            summary_df.to_html(os.path.join("inference_results", "inference_summary_svms.html"), index=False)
        except Exception as e:
            print("Failed to write SVM summary:", e)

        try:
            generate_dashboard(all_results, save_dir=RESULTS_ROOT)
        except Exception as e:
            print("Failed to generate dashboard:", e)


if __name__ == "__main__":
    main()


