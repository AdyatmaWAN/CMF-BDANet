"""
Moya / Hajeb SVM baseline grid-search training entrypoint.

Same shape as the other train_*.py scripts, driven by configs/svm.json.
Both Moya and Hajeb are height-difference SVM baselines that differ only in
their hand-crafted feature extractor; this script runs whichever model
names are listed in the config's "models" field.

Feature extraction respects the DSM-mode ablation axis: whichever DSM
channels a given dsm_mode selects (nDSM, density, uncertainty) are all fed
through the feature extractor and concatenated, so dsm_mode has the same
meaning here as it does for the neural-net models.

Usage:
    python train_svm.py
    python train_svm.py --config configs/svm.json --scenario 1 --model Moya
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import time
from typing import Dict

import numpy as np
import pandas as pd

from utils.experiment import set_global_determinism

set_global_determinism()

from sklearn.svm import SVC  # noqa: E402
from sklearn.utils import class_weight  # noqa: E402

from utils.experiment import (  # noqa: E402
    DSM_MODES,
    build_predictions_frame,
    compute_f2,
    dsm_mode_channels,
    grid_search,
    save_classification_report,
)
from models import Moya as moya_module  # noqa: E402
from models import Hajeb as hajeb_module  # noqa: E402
from models.fcsnn import load_dataset  # noqa: E402
from models.mmfemsnet import extract_inputs  # noqa: E402
from utils.label_processing import prepare_split_with_indices  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402

FEATURE_FUNCS = {
    "Moya": moya_module.feature_difference,
    "Hajeb": hajeb_module.dsm_difference,
}


def build_features(feature_func, dsm_pre: np.ndarray, dsm_post: np.ndarray) -> np.ndarray:
    """Apply a Moya/Hajeb per-channel feature extractor and concatenate across
    every active DSM channel (nDSM plus whichever of density/uncertainty the
    current dsm_mode includes).
    """
    parts = []
    for c in range(dsm_pre.shape[-1]):
        pair = np.stack([dsm_pre[..., c], dsm_post[..., c]], axis=1)  # (N, 2, H, W)
        parts.append(feature_func(pair))
    return np.concatenate(parts, axis=1)


def build_result_dir(
    results_root: str,
    scenario: int,
    dataset_name: str,
    model_name: str,
    *,
    dsm_mode: str,
    kernel: str,
    decision_function: str,
    C,
    gamma,
    **_ignored,
) -> str:
    return os.path.join(
        results_root,
        f"scenario_{scenario}",
        model_name,
        dataset_name,
        dsm_mode,
        f"dec_{decision_function}_kernel_{kernel}_C_{C}_gamma_{gamma}",
    )


def train_one(
    *,
    result_dir: str,
    dsm_mode: str,
    kernel: str,
    decision_function: str,
    C,
    gamma,
    scenario: int,
    dataset_name: str,
    model_name: str,
    num_classes: int,
    feat_train: np.ndarray,
    Y_train: np.ndarray,
    feat_test: np.ndarray,
    Y_test: np.ndarray,
    test_indices: np.ndarray,
) -> Dict:
    os.makedirs(result_dir, exist_ok=True)

    print(
        f"\n===== [{model_name}] scenario={scenario} dataset={dataset_name} dsm={dsm_mode} "
        f"dec={decision_function} kernel={kernel} C={C} gamma={gamma} ====="
    )

    weight = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(Y_train), y=Y_train
    )
    d_class_weights = dict(enumerate(weight))

    t0 = time.time()
    clf = SVC(
        decision_function_shape=decision_function,
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=d_class_weights,
    )
    clf.fit(feat_train, Y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(feat_test)

    scores = clf.decision_function(feat_test)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        prob_pos = 1.0 / (1.0 + np.exp(-scores))
        y_prob = np.vstack([1.0 - prob_pos, prob_pos]).T
    else:
        exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        y_prob = exps / np.sum(exps, axis=1, keepdims=True)

    y_true = np.asarray(Y_test).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)
    macro_f2 = compute_f2(y_true, y_pred, num_classes)

    pred_df = build_predictions_frame(y_true, y_pred, y_prob, num_classes, test_indices)
    pred_df.to_csv(os.path.join(result_dir, "predictions.csv"), index=False)
    save_classification_report(
        y_true, y_pred,
        os.path.join(result_dir, "classification_report.txt"),
        os.path.join(result_dir, "classification_report.csv"),
    )

    with open(os.path.join(result_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": model_name,
        "dsm_mode_tag": dsm_mode,
        "kernel": kernel,
        "decision_function": decision_function,
        "C": C,
        "gamma": gamma,
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
        "train_time_sec": train_time,
        "num_test_samples": int(len(y_true)),
        "predictions_csv": os.path.join(result_dir, "predictions.csv"),
        "result_dir": result_dir,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

    print(f"[{model_name}] scenario={scenario} ACC={acc:.4f} F1={f1:.4f} MCC={mcc:.4f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/svm.json")
    parser.add_argument("--scenario", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--dsm-mode", type=str, default=None, choices=[m["key"] for m in DSM_MODES])
    parser.add_argument("--model", type=str, default=None, choices=["Moya", "Hajeb"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    results_root = config["results_root"]
    scenarios = [args.scenario] if args.scenario is not None else config["scenarios"]
    model_names = [args.model] if args.model is not None else config["models"]
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
            X_test, Y_test, test_indices, _, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)

            for model_name in model_names:
                feature_func = FEATURE_FUNCS[model_name]

                def make_train_one_fn(dsm_mode: str):
                    include_density, include_unc = dsm_mode_channels(dsm_mode)
                    dsm_pre_train, dsm_post_train, _ = extract_inputs(
                        X_train, include_density=include_density, include_unc=include_unc
                    )
                    dsm_pre_test, dsm_post_test, _ = extract_inputs(
                        X_test, include_density=include_density, include_unc=include_unc
                    )
                    feat_train = build_features(feature_func, dsm_pre_train, dsm_post_train)
                    feat_test = build_features(feature_func, dsm_pre_test, dsm_post_test)
                    return functools.partial(
                        train_one,
                        scenario=scenario,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        num_classes=num_classes,
                        feat_train=feat_train,
                        Y_train=Y_train,
                        feat_test=feat_test,
                        Y_test=Y_test,
                        test_indices=test_indices,
                    )

                # dsm_mode changes the extracted feature set, so features must
                # be rebuilt per dsm_mode; everything else (kernel/C/gamma/...)
                # is a pure hyperparameter that reuses the same features.
                for dsm_mode in ablation_grid["dsm_mode"]:
                    results = grid_search(
                        ablation_grid={"dsm_mode": [dsm_mode]},
                        hpo_grid=config["hpo_grid"],
                        result_dir_fn=functools.partial(
                            build_result_dir, results_root, scenario, dataset_name, model_name
                        ),
                        train_one_fn=make_train_one_fn(dsm_mode),
                    )
                    all_results.extend(results)

    if all_results:
        os.makedirs(results_root, exist_ok=True)
        pd.DataFrame(all_results).to_csv(os.path.join(results_root, "SVM_grid_summary.csv"), index=False)


if __name__ == "__main__":
    main()
