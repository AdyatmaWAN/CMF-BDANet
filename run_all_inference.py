"""
Unified inference script for completed SNN and MMF-EMSNet HPO runs.

It scans saved experiment folders, rebuilds the matching architecture, loads the
saved checkpoint/weights, runs test-set inference, recomputes metrics, adds F2,
and saves per-instance prediction CSVs for each model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, fbeta_score

from models.mmfemsnet import build_mmf_emsnet_conv, make_dataset as make_mmf_dataset
from models.mmfemsnet import resolve_dsm_channel_indices
from models.snn import SNN, load_dataset
from utils.metrics import compute_metrics


SEED = 1234
DEFAULT_RESULTS_ROOTS = ["resResults", "resCheck", "results"]
DEFAULT_DATASET_ROOT = Path("Dataset") / "NPZ"
DEFAULT_OUTPUT_ROOT = Path("inference_results")

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
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_bool(value, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def as_int(value, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def load_dataset_by_name(dataset_name: str, dataset_root: Path):
    candidate = dataset_root / f"{dataset_name}.npz"
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    return load_dataset(str(candidate))


def prepare_split_with_indices(X: np.ndarray, Y: np.ndarray, scenario: int):
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


def dsm_mode_to_flags(dsm_mode_tag: str):
    if dsm_mode_tag == "dsm_density_uncertainty":
        return True, True
    if dsm_mode_tag == "dsm_density":
        return True, False
    if dsm_mode_tag == "dsm_uncertainty":
        return False, True
    return False, False


def dsm_channels_count(include_density: bool, include_unc: bool) -> int:
    pre_idx, _ = resolve_dsm_channel_indices(include_density=include_density, include_unc=include_unc)
    return len(pre_idx)


def make_snn_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    include_density: bool = True,
    include_unc: bool = True,
) -> tf.data.Dataset:
    pre_idx, post_idx = resolve_dsm_channel_indices(
        include_density=include_density,
        include_unc=include_unc,
    )
    dsm_pre = X[..., pre_idx]
    dsm_post = X[..., post_idx]
    ds = tf.data.Dataset.from_tensor_slices(((dsm_pre, dsm_post), Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(batch_size)
    try:
        opts = tf.data.Options()
        opts.experimental_threading.private_threadpool_size = 1
        opts.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
    except Exception:
        pass
    return ds.prefetch(1)


def candidate_checkpoint_files(result_dir: Path) -> List[Path]:
    return [
        result_dir / "best_weights.weights.h5",
        result_dir / "best_weights.h5",
        result_dir / "model.weights.h5",
        result_dir / "weights.h5",
        result_dir / "model.keras",
    ]


def load_keras_model(model_path: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(model_path, compile=False)


def build_model_from_metadata(model_name: str, num_classes: int, residual: bool, fusion: str,
                              input_shape_dsm: Tuple[int, int, int], input_shape_rgb: Tuple[int, int, int],
                              concat_post_dsm: bool, four_stream: bool) -> tf.keras.Model:
    if model_name.upper().strip() == "SNN":
        snn = SNN(
            num_of_class=num_classes,
            residual=residual,
            dropout=True,
            dense=True,
            num_of_layer=3,
            input_shape=input_shape_dsm,
            substraction=True,
            shared=True,
            fusion=fusion,
        )
        return snn.get_model()

    if model_name.upper().strip() == "MMF":
        return build_mmf_emsnet_conv(
            input_shape_dsm=input_shape_dsm,
            input_shape_rgb=input_shape_rgb,
            num_classes=num_classes,
            token_dim=128,
            concat_post_dsm=concat_post_dsm,
            four_stream=four_stream,
            residual=residual,
            fusion=fusion,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def load_inference_model(result_dir: Path, model_name: str, num_classes: int, residual: bool, fusion: str,
                         input_shape_dsm: Tuple[int, int, int], input_shape_rgb: Tuple[int, int, int],
                         concat_post_dsm: bool, four_stream: bool):
    for candidate in candidate_checkpoint_files(result_dir):
        if not candidate.exists():
            continue
        if candidate.suffix == ".keras":
            return load_keras_model(candidate), candidate, "keras_model"
        model = build_model_from_metadata(
            model_name=model_name,
            num_classes=num_classes,
            residual=residual,
            fusion=fusion,
            input_shape_dsm=input_shape_dsm,
            input_shape_rgb=input_shape_rgb,
            concat_post_dsm=concat_post_dsm,
            four_stream=four_stream,
        )
        model.load_weights(str(candidate))
        return model, candidate, "weights"

    raise FileNotFoundError(
        f"No checkpoint found in {result_dir}. Tried: " + ", ".join(p.name for p in candidate_checkpoint_files(result_dir))
    )


def collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: List[np.ndarray] = []
    for _, y in dataset:
        labels.append(np.asarray(y))
    return np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, path_txt: Path, path_csv: Path, labels: Sequence[int]) -> None:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
    with path_txt.open("w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=list(labels), zero_division=0))
    pd.DataFrame(report_dict).transpose().to_csv(path_csv, index=True)


def build_predictions_frame(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                            num_classes: int, source_indices: np.ndarray) -> pd.DataFrame:
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
        prob_pos = np.asarray(y_prob).reshape(-1)
        prob_neg = 1.0 - prob_pos
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


def discover_metrics_files(results_roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for root in results_roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("metrics.csv")):
            if metrics_path in seen:
                continue
            seen.add(metrics_path)
            files.append(metrics_path)
    return files


def infer_one_run(metrics_path: Path, dataset_root: Path):
    source_df = pd.read_csv(metrics_path)
    if source_df.empty:
        return {}, {"metrics_path": str(metrics_path), "status": "skipped", "reason": "metrics.csv is empty"}

    if len(source_df) > 1:
        print(f"Warning: {metrics_path} has multiple rows; using the first row only.")

    source = source_df.iloc[0].to_dict()
    result_dir = metrics_path.parent

    scenario = as_int(source.get("scenario"), default=0)
    dataset_name = str(source.get("dataset", ""))
    model_name = str(source.get("model", ""))
    residual = as_bool(source.get("residual"), default=False)
    fusion = str(source.get("fusion", "concat"))
    dsm_mode_tag = str(source.get("dsm_mode_tag", "dsm_only"))
    concat_post_dsm = as_bool(source.get("dsm_post_concat_with_rgb"), default=True)
    four_stream = as_bool(source.get("use_four_stream"), default=False)
    num_classes = as_int(source.get("num_classes"), default=1)
    include_density, include_unc = dsm_mode_to_flags(dsm_mode_tag)

    try:
        _, _, _, _, X_test_raw, Y_test_raw = load_dataset_by_name(dataset_name, dataset_root)
        X_test, Y_test, test_indices, inferred_num_classes, _ = prepare_split_with_indices(X_test_raw, Y_test_raw, scenario)
        if len(X_test) == 0:
            return {}, {
                "metrics_path": str(metrics_path),
                "result_dir": str(result_dir),
                "status": "skipped",
                "reason": f"No test samples remain after scenario {scenario}",
            }

        if inferred_num_classes != num_classes:
            print(f"Warning: {metrics_path} num_classes={num_classes} differs from inferred {inferred_num_classes}")

        H, W = X_test.shape[1], X_test.shape[2]
        input_shape_dsm = (H, W, dsm_channels_count(include_density, include_unc))
        input_shape_rgb = (H, W, 3)
        model, checkpoint_path, checkpoint_kind = load_inference_model(
            result_dir=result_dir,
            model_name=model_name,
            num_classes=num_classes,
            residual=residual,
            fusion=fusion,
            input_shape_dsm=input_shape_dsm,
            input_shape_rgb=input_shape_rgb,
            concat_post_dsm=concat_post_dsm,
            four_stream=four_stream,
        )

        batch_size = as_int(source.get("batch_size"), default=64)
        if model_name.upper() == "SNN":
            test_ds = make_snn_dataset(
                X_test,
                Y_test,
                batch_size=batch_size,
                shuffle=False,
                include_density=include_density,
                include_unc=include_unc,
            )
        else:
            test_ds = make_mmf_dataset(
                X_test,
                Y_test,
                batch=batch_size,
                shuffle=False,
                include_density=include_density,
                include_unc=include_unc,
                four_stream=four_stream,
            )

        y_prob = model.predict(test_ds, verbose=0)
        if num_classes == 1:
            y_prob_arr = np.asarray(y_prob).reshape(-1)
            y_pred = (y_prob_arr >= 0.5).astype(np.int64)
        else:
            y_prob_arr = np.asarray(y_prob)
            y_pred = np.argmax(y_prob_arr, axis=-1).astype(np.int64)

        y_true = collect_labels(test_ds).astype(np.int64)
        acc, prec, rec, macro_f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)
        macro_f2 = compute_f2(y_true, y_pred, num_classes)

        inference_dir = result_dir / "inference"
        ensure_dir(inference_dir)

        pred_df = build_predictions_frame(y_true, y_pred, y_prob_arr, num_classes, test_indices)
        predictions_path = inference_dir / "predictions.csv"
        pred_df.to_csv(predictions_path, index=False)

        labels = [0, 1] if num_classes == 1 else list(range(num_classes))
        save_classification_report(
            y_true,
            y_pred,
            inference_dir / "classification_report.txt",
            inference_dir / "classification_report.csv",
            labels=labels,
        )

        summary = dict(source)
        summary.update(
            {
                "scenario": scenario,
                "dataset": dataset_name,
                "model": model_name,
                "residual": residual,
                "fusion": fusion,
                "dsm_mode_tag": dsm_mode_tag,
                "dsm_post_concat_with_rgb": concat_post_dsm,
                "use_four_stream": four_stream,
                "num_classes": num_classes,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "macro_f1": macro_f1,
                "macro_f2": macro_f2,
                "mcc": mcc,
                "cm00": int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else 0,
                "cm01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
                "cm10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
                "cm11": int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
                "confusion_matrix_shape": f"{cm.shape[0]}x{cm.shape[1]}",
                "confusion_matrix_json": json.dumps(cm.tolist()),
                "num_test_samples": int(len(y_true)),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_kind": checkpoint_kind,
                "inference_dir": str(inference_dir),
                "predictions_csv": str(predictions_path),
                "status": "ok",
                "source_metrics_path": str(metrics_path),
            }
        )
        inference_metrics_path = inference_dir / "inference_metrics.csv"
        pd.DataFrame([summary]).to_csv(inference_metrics_path, index=False)
        summary["inference_metrics_csv"] = str(inference_metrics_path)

        print(f"[OK] {model_name} | scenario {scenario} | {dataset_name} | ACC={acc:.4f} | F1={macro_f1:.4f} | F2={macro_f2:.4f}")
        return summary, None

    except Exception:
        err_path = result_dir / "inference" / "error.log"
        ensure_dir(err_path.parent)
        with err_path.open("a", encoding="utf-8") as f:
            f.write("\nInference failed with exception:\n")
            f.write(traceback.format_exc())
            f.write("\n")
        skipped = {
            "metrics_path": str(metrics_path),
            "result_dir": str(result_dir),
            "scenario": scenario,
            "dataset": dataset_name,
            "model": model_name,
            "status": "skipped",
            "reason": f"Inference failed; see {err_path}",
            "error_log": str(err_path),
        }
        print(f"[SKIP] {metrics_path} -> see {err_path}")
        return {}, skipped


def write_global_dashboard(success_rows: List[Dict], output_root: Path) -> None:
    ensure_dir(output_root)
    if not success_rows:
        print("No successful inference runs to summarize.")
        return
    df = pd.DataFrame(success_rows)
    df.to_csv(output_root / "inference_summary.csv", index=False)
    df.to_html(output_root / "inference_summary.html", index=False)
    print(f"Wrote inference summary: {output_root / 'inference_summary.csv'}")


def write_skipped_dashboard(skipped_rows: List[Dict], output_root: Path) -> None:
    if not skipped_rows:
        return
    ensure_dir(output_root)
    pd.DataFrame(skipped_rows).to_csv(output_root / "inference_skipped.csv", index=False)
    print(f"Wrote skipped-run log: {output_root / 'inference_skipped.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-roots",
        nargs="+",
        default=DEFAULT_RESULTS_ROOTS,
        help="One or more result roots to scan for completed runs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root folder containing the NPZ datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Folder for the global inference dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    results_roots = [Path(p) for p in args.results_roots]
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    metrics_files = discover_metrics_files(results_roots)
    if not metrics_files:
        print("No metrics.csv files found under the selected results roots.")
        return

    print(f"Found {len(metrics_files)} run(s) to infer.")
    success_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    for metrics_path in metrics_files:
        summary, skipped = infer_one_run(metrics_path, dataset_root)
        if summary:
            success_rows.append(summary)
        if skipped:
            skipped_rows.append(skipped)

    write_global_dashboard(success_rows, output_root)
    write_skipped_dashboard(skipped_rows, output_root)


if __name__ == "__main__":
    main()


