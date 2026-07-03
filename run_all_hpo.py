"""
Unified Hyperparameter Optimization script for SNN and MMF-EMSNet.

Runs HPO over:
    - Optimizer ∈ {Adam, SGD, RMSprop, Nadam, Adamax}
    - LR ∈ {1e-4, 1e-3}
    - Batch size ∈ {256, 128, 64}
    - Datasets ∈ {dataset_16, dataset_32, dataset_64}
    - Models ∈ {SNN (DSM), MMF-EMSNet (DSM + RGB)}
    - Scenarios:
        1: class 0 vs class 4 (binary)
        2: classes 0–3 vs class 4 (binary)
        3: all 5 classes (multiclass)

Usage:
    python run_all_hpo.py --scenario 1
    python run_all_hpo.py --scenario 2
    python run_all_hpo.py --scenario 3
    python run_all_hpo.py --scenario all
"""

from __future__ import annotations

from typing import Dict, List, Set

import argparse
import os
import random
import time
import json
import traceback

import numpy as np

# ============================================================
#  GLOBAL DETERMINISM (must run before TF imports)
# ============================================================

SEED = 1234

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"

random.seed(SEED)
np.random.seed(SEED)

# Limit OpenMP / MKL threads early to avoid excessive thread creation by
# low-level numeric libraries (helps prevent pthread_create failures).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from tensorflow.keras import callbacks, optimizers  # noqa: E402

from models.snn import SNN, load_dataset  # noqa: E402
from models.mmfemsnet import (  # noqa: E402
    build_mmf_emsnet_conv,
    make_dataset as make_mmf_dataset,
    resolve_dsm_channel_indices,
)
from utils.label_processing import prepare_split_for_scenario  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

DATASETS: Dict[str, str] = {
    "dataset_16": "Dataset/NPZ/dataset_16.npz",
    # "dataset_32": "Dataset/NPZ/dataset_32.npz",
    # "dataset_64": "Dataset/NPZ/dataset_64.npz",
}

DSM_MODES: List[Dict[str, object]] = [
    {
        "key": "dsm_density_uncertainty",
        "label": "dsm+density+uncertainty",
        "tag": "dsm_density_uncertainty",
        "include_density": True,
        "include_unc": True,
    },
    {
        "key": "dsm_density",
        "label": "dsm+density",
        "tag": "dsm_density",
        "include_density": True,
        "include_unc": False,
    },
    {
        "key": "dsm_uncertainty",
        "label": "dsm+uncertainty",
        "tag": "dsm_uncertainty",
        "include_density": False,
        "include_unc": True,
    },
    {
        "key": "dsm_only",
        "label": "dsm only",
        "tag": "dsm_only",
        "include_density": False,
        "include_unc": False,
    },
]

OPTIMIZERS: List[str] = ["Adam", "SGD", "RMSprop", "Nadam", "Adamax"]
LEARNING_RATES: List[float] = [1e-4, 1e-3]
BATCH_SIZES: List[int] = [256, 128, 64]
# BATCH_SIZES: List[int] = [256]
EPOCHS: int = 100

EARLY_STOPPING_PATIENCE: int = 12
LR_REDUCE_PATIENCE: int = 6
LR_REDUCE_FACTOR: float = 0.5
LR_REDUCE_MIN_LR: float = 1e-6
MIN_DELTA: float = 1e-4

RESULTS_ROOT = "results"  # Root for all results


# ============================================================
# UTILS: OPTIMIZER + DASHBOARD
# ============================================================

def set_seed(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def get_optimizer(name: str, lr: float) -> optimizers.Optimizer:
    """
    Instantiate Keras optimizer by name and learning rate.
    """
    opt_dict: Dict[str, optimizers.Optimizer] = {
        "Adam": optimizers.Adam(lr),
        "SGD": optimizers.SGD(lr),
        "RMSprop": optimizers.RMSprop(lr),
        "Nadam": optimizers.Nadam(lr),
        "Adamax": optimizers.Adamax(lr),
    }
    if name not in opt_dict:
        raise ValueError(f"Unknown optimizer: {name}")
    return opt_dict[name]


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path_txt: str,
    path_csv: str,
) -> None:
    """
    Save classification report in TXT (pretty) and CSV (tabular) formats.
    """
    cr_dict = classification_report(y_true, y_pred, output_dict=True)

    # TXT
    with open(path_txt, "w") as f:
        report_text = classification_report(y_true, y_pred)
        f.write(str(report_text))

    # CSV
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_df.to_csv(path_csv, index=True)


def _collect_dataset_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect labels from a batched tf.data.Dataset into a flat numpy array."""
    labels: List[np.ndarray] = []
    for _, y in dataset:
        if tf.is_tensor(y):
            labels.append(y.numpy())
        else:
            labels.append(np.asarray(y))
    return np.concatenate(labels, axis=0)


def get_dsm_mode_tag(include_density: bool, include_unc: bool) -> str:
    """Return a stable tag for the current DSM feature combination."""
    if include_density and include_unc:
        return "dsm_density_uncertainty"
    if include_density:
        return "dsm_density"
    if include_unc:
        return "dsm_uncertainty"
    return "dsm_only"


class ValidationF1Callback(callbacks.Callback):
    """Compute validation F1 at the end of each epoch and inject it into logs."""

    def __init__(self, val_data: tf.data.Dataset, num_classes: int):
        super().__init__()
        self.val_data = val_data
        self.num_classes = num_classes
        self.y_true = _collect_dataset_labels(val_data)
        self.best_val_f1 = float("-inf")
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            logs = {}

        y_prob = self.model.predict(self.val_data, verbose=0)
        if self.num_classes == 1:
            y_pred = (np.asarray(y_prob).reshape(-1) >= 0.5).astype(int)
        else:
            y_pred = np.argmax(np.asarray(y_prob), axis=-1)

        _, _, _, val_f1, _, _ = compute_metrics(self.y_true, y_pred, self.num_classes)
        logs["val_f1"] = val_f1

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_epoch = epoch + 1

        print(f" - val_f1: {val_f1:.4f}")


def make_training_callbacks(
    results_dir: str,
    val_data: tf.data.Dataset,
    num_classes: int,
) -> tuple[list[callbacks.Callback], ValidationF1Callback, str]:
    """Create callbacks that monitor validation F1."""
    f1_callback = ValidationF1Callback(val_data, num_classes)
    best_weights_path = os.path.join(results_dir, "best_weights.weights.h5")

    callback_list: list[callbacks.Callback] = [
        f1_callback,
        callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_f1",
            mode="max",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_delta=MIN_DELTA,
            min_lr=LR_REDUCE_MIN_LR,
            verbose=1,
        ),
    ]
    return callback_list, f1_callback, best_weights_path


def generate_dashboard(
    all_results: List[Dict],
    save_dir: str = RESULTS_ROOT,
) -> None:
    """
    Generate CSV/HTML dashboard and best-model summary across experiments.
    """
    if not all_results:
        print("No results to summarize.")
        return

    df = pd.DataFrame(all_results)

    os.makedirs(save_dir, exist_ok=True)
    progress_csv = os.path.join(save_dir, "progress_dashboard.csv")
    progress_html = os.path.join(save_dir, "progress_dashboard.html")
    df.to_csv(progress_csv, index=False)
    df.to_html(progress_html, index=False)

    # Best models per (scenario, dataset, model) based on MCC
    best_df = (
        df.sort_values("mcc", ascending=False)
        .drop_duplicates(subset=["scenario", "dataset", "model", "dsm_mode_tag"])
        .reset_index(drop=True)
    )
    best_csv = os.path.join(save_dir, "best_models.csv")
    best_html = os.path.join(save_dir, "best_models.html")
    best_df.to_csv(best_csv, index=False)
    best_df.to_html(best_html, index=False)

    print("\n===== PROGRESS DASHBOARD =====")
    print(f"All runs summary:        {progress_csv}")
    print(f"Best models per setting: {best_csv}")


# -------------------------
# Checkpoint / resume utils
# -------------------------
CHECKPOINT_PATH = os.path.join(RESULTS_ROOT, "progress_checkpoint.json")


def load_checkpoint() -> Set[str]:
    """Load set of completed run IDs from checkpoint file."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                data = json.load(f)
            return set(data.get("completed", []))
        except Exception:
            print("Warning: failed to read checkpoint file, starting fresh.")
            return set()
    return set()


def save_checkpoint(completed: Set[str]) -> None:
    """Persist completed run IDs to disk."""
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": sorted(list(completed))}, f, indent=2)



# ============================================================
# DATASET PIPELINES (SNN)
# ============================================================


def make_snn_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    include_density: bool = True,
    include_unc: bool = True,
) -> tf.data.Dataset:
    """
    Build tf.data.Dataset for SNN.

    Uses DSM channels selected by flags:
        - mandatory: pre/post nDSM
        - optional: pre/post density and/or uncertainty
    """
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

    # Limit prefetch to 1 and reduce private threadpool to avoid excessive
    # pthread creation on some platforms. The options are best-effort — if the
    # TF build doesn't support an option it will be skipped.
    try:
        opts = tf.data.Options()
        # Restrict the private threadpool used by the tf.data runtime
        opts.experimental_threading.private_threadpool_size = 1
        opts.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
    except Exception:
        pass

    ds = ds.prefetch(1)
    return ds


def make_snn_test_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    include_density: bool = True,
    include_unc: bool = True,
) -> tf.data.Dataset:
    """
    Convenience wrapper for SNN test dataset pipeline.
    """
    return make_snn_dataset(
        X,
        Y,
        batch_size=batch_size,
        shuffle=False,
        include_density=include_density,
        include_unc=include_unc,
    )


# ============================================================
# TRAIN + EVAL: SNN
# ============================================================


def run_snn_experiment(
    scenario: int,
    dataset_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_classes: int,
    optimizer_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    results_dir: str,
    all_results_list: List[Dict],
    include_density: bool,
    include_unc: bool,
    residual: bool,
    fusion: str = "concat",
) -> None:
    """
    Train + evaluate one SNN configuration under a given scenario.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"\n===== [SNN] Scenario {scenario} | {dataset_name} | "
        f"OPT={optimizer_name} | LR={lr} | BS={batch_size} ====="
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
    dsm_channels = len(
        resolve_dsm_channel_indices(
            include_density=include_density,
            include_unc=include_unc,
        )[0]
    )
    dsm_mode_tag = get_dsm_mode_tag(include_density, include_unc)
    input_shape = (H, W, dsm_channels)
    set_seed(1234)
    snn = SNN(
        num_of_class=num_classes,
        residual=residual,
        dropout=True,
        dense=True,
        num_of_layer=3,
        input_shape=input_shape,
        substraction=True,
        shared=True,
        fusion=fusion,
    )
    model = snn.get_model()

    if num_classes == 1:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=get_optimizer(optimizer_name, lr),
        loss=loss,
        metrics=["accuracy"],
    )

    training_callbacks, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=results_dir,
        val_data=val_ds,
        num_classes=num_classes,
    )

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=training_callbacks,
    )

    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)

    train_time = time.time() - t0

    # Evaluate
    y_prob = model.predict(test_ds)
    if num_classes == 1:
        y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    else:
        y_pred = np.argmax(y_prob, axis=-1)

    y_true = np.concatenate([y for (_, y) in test_ds], axis=0)

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame(
        [
            {
                "scenario": scenario,
                "dataset": dataset_name,
                "model": "SNN",
                "residual": residual,
                "fusion": fusion,
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "num_classes": num_classes,
                # "dsm_include_density": include_density,
                # "dsm_include_unc": include_unc,
                "dsm_mode_tag": dsm_mode_tag,
                "dsm_post_concat_with_rgb": False,
                "use_four_stream": False,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "macro_f1": f1,
                "best_val_f1": f1_callback.best_val_f1,
                "best_val_epoch": f1_callback.best_epoch,
                "monitor_metric": "val_f1",
                "mcc": mcc,
                "cm00": cm[0][0],
                "cm01": cm[0][1],
                "cm10": cm[1][0],
                "cm11": cm[1][1],
                "train_time_sec": train_time,
                "result_dir": results_dir,
            }
        ]
    ).to_csv(metrics_path, index=False)

    # Save classification report
    cr_txt = os.path.join(results_dir, "classification_report.txt")
    cr_csv = os.path.join(results_dir, "classification_report.csv")
    save_classification_report(y_true, y_pred, cr_txt, cr_csv)

    # Save model
    model.save(os.path.join(results_dir, "model.keras"))
    model.save_weights(os.path.join(results_dir, "weights.h5"))

    all_results_list.append(
        {
            "scenario": scenario,
            "dataset": dataset_name,
            "model": "SNN",
            "residual": residual,
            "fusion": fusion,
            "optimizer": optimizer_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_classes": num_classes,
            # "dsm_include_density": include_density,
            # "dsm_include_unc": include_unc,
            "dsm_mode_tag": dsm_mode_tag,
            "dsm_post_concat_with_rgb": False,
            "use_four_stream": False,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": f1,
            "best_val_f1": f1_callback.best_val_f1,
            "best_val_epoch": f1_callback.best_epoch,
            "monitor_metric": "val_f1",
            "mcc": mcc,
            "cm00": cm[0][0],
            "cm01": cm[0][1],
            "cm10": cm[1][0],
            "cm11": cm[1][1],
            "train_time_sec": train_time,
            "result_dir": results_dir,
        }
    )

    print(f"[SNN] Scenario {scenario} | ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")


# ============================================================
# TRAIN + EVAL: MMF-EMSNet
# ============================================================


def run_mmf_experiment(
    scenario: int,
    dataset_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_classes: int,
    optimizer_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    results_dir: str,
    all_results_list: List[Dict],
    include_density: bool,
    include_unc: bool,
    concat_post_dsm: bool,
    four_stream: bool = False,
    residual: bool = False,
    fusion: str = "mcmaf",
) -> None:
    """
    Train + evaluate one MMF-EMSNet configuration under a given scenario.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"\n===== [MMF] Scenario {scenario} | {dataset_name} | "
        f"OPT={optimizer_name} | LR={lr} | BS={batch_size} ====="
    )

    train_ds = make_mmf_dataset(
        X_train,
        Y_train,
        batch_size,
        shuffle=True,
        include_density=include_density,
        include_unc=include_unc,
        four_stream=four_stream,
    )
    val_ds = make_mmf_dataset(
        X_val,
        Y_val,
        batch_size,
        shuffle=False,
        include_density=include_density,
        include_unc=include_unc,
        four_stream=four_stream,
    )
    test_ds = make_mmf_dataset(
        X_test,
        Y_test,
        batch_size,
        shuffle=False,
        include_density=include_density,
        include_unc=include_unc,
        four_stream=four_stream,
    )

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(
        resolve_dsm_channel_indices(
            include_density=include_density,
            include_unc=include_unc,
        )[0]
    )
    dsm_mode_tag = get_dsm_mode_tag(include_density, include_unc)
    set_seed(1234)
    model = build_mmf_emsnet_conv(
        input_shape_dsm=(H, W, dsm_channels),
        input_shape_rgb=(H, W, 3),
        num_classes=num_classes,
        token_dim=128,
        concat_post_dsm=concat_post_dsm,
        four_stream=four_stream,
        residual=residual,
        fusion=fusion,
    )

    if num_classes == 1:
        loss = "binary_crossentropy"
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=get_optimizer(optimizer_name, lr),
        loss=loss,
        metrics=["accuracy"],
    )

    training_callbacks, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=results_dir,
        val_data=val_ds,
        num_classes=num_classes,
    )

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=training_callbacks,
    )

    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)

    train_time = time.time() - t0

    # Evaluate
    y_prob = model.predict(test_ds)
    if num_classes == 1:
        y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    else:
        y_pred = np.argmax(y_prob, axis=-1)

    y_true = np.concatenate([y for (_, y) in test_ds], axis=0)

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.csv")
    pd.DataFrame(
        [
            {
                "scenario": scenario,
                "dataset": dataset_name,
                "model": "MMF",
                "residual": residual,
                "fusion": fusion,
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "num_classes": num_classes,
                # "dsm_include_density": include_density,
                # "dsm_include_unc": include_unc,
                "dsm_mode_tag": dsm_mode_tag,
                "dsm_post_concat_with_rgb": concat_post_dsm,
                "use_four_stream": four_stream,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "macro_f1": f1,
                "best_val_f1": f1_callback.best_val_f1,
                "best_val_epoch": f1_callback.best_epoch,
                "monitor_metric": "val_f1",
                "mcc": mcc,
                "cm00": cm[0][0],
                "cm01": cm[0][1],
                "cm10": cm[1][0],
                "cm11": cm[1][1],
                "train_time_sec": train_time,
                "result_dir": results_dir,
            }
        ]
    ).to_csv(metrics_path, index=False)

    # Save classification report
    cr_txt = os.path.join(results_dir, "classification_report.txt")
    cr_csv = os.path.join(results_dir, "classification_report.csv")
    save_classification_report(y_true, y_pred, cr_txt, cr_csv)

    # Save model
    model.save(os.path.join(results_dir, "model.keras"))
    model.save_weights(os.path.join(results_dir, "weights.h5"))

    all_results_list.append(
        {
            "scenario": scenario,
            "dataset": dataset_name,
            "model": "MMF",
            "residual": residual,
            "fusion": fusion,
            "optimizer": optimizer_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "num_classes": num_classes,
            # "dsm_include_density": include_density,
            # "dsm_include_unc": include_unc,
            "dsm_mode_tag": dsm_mode_tag,
            "dsm_post_concat_with_rgb": concat_post_dsm,
            "use_four_stream": four_stream,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": f1,
            "best_val_f1": f1_callback.best_val_f1,
            "best_val_epoch": f1_callback.best_epoch,
            "monitor_metric": "val_f1",
            "mcc": mcc,
            "cm00": cm[0][0],
            "cm01": cm[0][1],
            "cm10": cm[1][0],
            "cm11": cm[1][1],
            "train_time_sec": train_time,
            "result_dir": results_dir,
        }
    )

    print(f"[MMF] Scenario {scenario} | ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")


# ============================================================
# MAIN
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="1, 2, 3 or 'all' (for running all scenarios).",
    )
    parser.add_argument(
        "--dsm-mode",
        type=str,
        default="all",
        choices=["all"] + [mode["key"] for mode in DSM_MODES],
        help="DSM feature mode to run. Default runs all four modes.",
    )
    parser.add_argument(
        "--use-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include density_norm channels in DSM inputs (default: enabled).",
    )
    parser.add_argument(
        "--use-unc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include unc_norm channels in DSM inputs (default: enabled).",
    )
    parser.add_argument(
        "--mmf-variants",
        type=str,
        choices=["3stream", "4stream", "both"],
        default="both",
        help=(
            "MMF-EMSNet variants to run: '3stream' (only 3-stream), '4stream' (only 4-stream), "
            "or 'both' (default: both). "
            "For 3-stream, use --concat to control post-DSM + post-RGB concatenation."
        ),
    )
    parser.add_argument(
        "--concat",
        type=str,
        choices=["yes", "no", "both"],
        default="both",
        help=(
            "For 3-stream variant: whether to concatenate post-DSM with post-RGB. "
            "'yes' (concat enabled), 'no' (concat disabled), or 'both' (default: test both). "
            "Ignored when using 4-stream variant."
        ),
    )
    parser.add_argument(
        "--residual",
        type=str,
        choices=["true", "false", "both"],
        default="both",
        help=(
            "Whether to use residual connections: 'true', 'false', or 'both' (default: both)."
        ),
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "mcmaf", "both"],
        default="both",
        help=(
            "Fusion method to use before the classifier: 'concat' (simple concatenation), "
            "'mcmaf' (attention-style MCMAF), or 'both' (default: try both)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dsm_mode == "all":
        selected_dsm_modes = DSM_MODES
    else:
        selected_dsm_modes = [mode for mode in DSM_MODES if mode["key"] == args.dsm_mode]

    dsm_descriptions = ", ".join([str(mode["label"]) for mode in selected_dsm_modes])
    print(f"DSM input modes: {dsm_descriptions}")

    if args.scenario == "all":
        scenarios = [1, 2, 3]
    else:
        scenarios = [int(args.scenario)]
        if scenarios[0] not in (1, 2, 3):
            raise ValueError("Scenario must be 1, 2, 3, or 'all'.")

    # Parse MMF variant choices
    if args.mmf_variants == "both":
        mmf_variants_to_run = [False, True]  # False=3stream, True=4stream
        print("Running MMF with both 3-stream and 4-stream variants")
    elif args.mmf_variants == "3stream":
        mmf_variants_to_run = [False]
        print("Running MMF with 3-stream variant only")
    else:  # "4stream"
        mmf_variants_to_run = [True]
        print("Running MMF with 4-stream variant only")

    # Parse concat choices (only relevant for 3-stream)
    if args.concat == "both":
        concat_choices = [True, False]
        print("For 3-stream: testing both concat and no-concat options")
    elif args.concat == "yes":
        concat_choices = [True]
        print("For 3-stream: concat enabled")
    else:  # "no"
        concat_choices = [False]
        print("For 3-stream: concat disabled")

    all_results: List[Dict] = []

    # Parse fusion choices: whether to try concat, mcmaf, or both
    if args.fusion == "both":
        fusion_choices = ["concat", "mcmaf"]
        print("Testing both fusion methods: concat and mcmaf")
    else:
        fusion_choices = [args.fusion]
        print(f"Testing fusion method: {args.fusion}")

    for mode in selected_dsm_modes:
        include_density = bool(mode["include_density"])
        include_unc = bool(mode["include_unc"])
        dsm_mode_tag = str(mode["tag"])

        # Determine residual choices: if user left default 'both', test both
        # True/False. Otherwise use the explicit choice.
        if args.residual == "both":
            residual_choices = [True, False]
        else:
            residual_choices = [args.residual == "true"]

        dsm_pre_idx, _ = resolve_dsm_channel_indices(
            include_density=include_density,
            include_unc=include_unc,
        )
        print(
            f"\n>>> Running DSM mode '{mode['label']}' "
            f"({len(dsm_pre_idx)} channel(s) per pre/post tensor)"
        )

        for scenario in scenarios:
            for ds_name, npz_path in DATASETS.items():
                print("\n" + "=" * 70)
                print(
                    f"Scenario {scenario} | DSM mode: {mode['label']} | "
                    f"Loading dataset: {ds_name} from {npz_path}"
                )
                print("=" * 70)

                # Load raw splits
                (
                    X_train_raw,
                    Y_train_raw,
                    X_val_raw,
                    Y_val_raw,
                    X_test_raw,
                    Y_test_raw,
                ) = load_dataset(npz_path)

                # Process labels per scenario
                X_train_proc, Y_train_proc, num_classes, _ = prepare_split_for_scenario(
                    X_train_raw, Y_train_raw, scenario
                )
                X_val_proc, Y_val_proc, _, _ = prepare_split_for_scenario(
                    X_val_raw, Y_val_raw, scenario
                )
                X_test_proc, Y_test_proc, _, _ = prepare_split_for_scenario(
                    X_test_raw, Y_test_raw, scenario
                )

                # ============= SNN HPO =============
                completed_runs = load_checkpoint()
                for opt_name in OPTIMIZERS:
                    for lr in LEARNING_RATES:
                        for bs in BATCH_SIZES:
                            for residual_choice in residual_choices:
                                for fusion_choice in fusion_choices:
                                    result_dir = os.path.join(
                                        RESULTS_ROOT,
                                        f"scenario_{scenario}",
                                        "SNN",
                                        ds_name,
                                        dsm_mode_tag,
                                        f"residual_{residual_choice}",
                                        f"fusion_{fusion_choice}",
                                        f"{opt_name}_lr{lr}_bs{bs}",
                                    )
                                    run_id = (
                                        f"scenario={scenario}|dataset={ds_name}|model=SNN|"
                                        f"dsm={dsm_mode_tag}|residual={residual_choice}|fusion={fusion_choice}|opt={opt_name}|lr={lr}|bs={bs}"
                                    )

                                    if run_id in completed_runs:
                                        print(f"Skipping completed run: {run_id}")
                                        # Try to recover existing metrics to keep dashboard up-to-date
                                        metrics_path = os.path.join(result_dir, "metrics.csv")
                                        if os.path.exists(metrics_path):
                                            try:
                                                df = pd.read_csv(metrics_path)
                                                all_results.extend(df.to_dict(orient="records"))
                                            except Exception:
                                                pass
                                        continue

                                    try:
                                        run_snn_experiment(
                                            scenario=scenario,
                                            dataset_name=ds_name,
                                            X_train=X_train_proc,
                                            Y_train=Y_train_proc,
                                            X_val=X_val_proc,
                                            Y_val=Y_val_proc,
                                            X_test=X_test_proc,
                                            Y_test=Y_test_proc,
                                            num_classes=num_classes,
                                            optimizer_name=opt_name,
                                            lr=lr,
                                            batch_size=bs,
                                            epochs=EPOCHS,
                                            results_dir=result_dir,
                                            all_results_list=all_results,
                                            include_density=include_density,
                                            include_unc=include_unc,
                                            residual=residual_choice,
                                            fusion=fusion_choice,
                                        )

                                        # mark completed and persist checkpoint
                                        completed_runs.add(run_id)
                                        save_checkpoint(completed_runs)
                                    except Exception as e:
                                        # log exception to the result directory and continue
                                        os.makedirs(result_dir, exist_ok=True)
                                        err_path = os.path.join(result_dir, "error.log")
                                        with open(err_path, "a") as f:
                                            f.write("\nRun failed with exception:\n")
                                            f.write(traceback.format_exc())
                                        print(f"Run {run_id} failed; logged to {err_path}")
                                        # do not add to completed_runs so it will be retried next run
                                        continue

                # ============= MMF HPO =============
                # Run MMF with selected variants
                completed_runs = load_checkpoint()
                concat_tag_map = {True: "3stream_concat", False: "3stream_no_concat"}

                for four_stream_choice in mmf_variants_to_run:
                    if four_stream_choice:
                        # 4-stream: concat doesn't apply
                        concat_variants = [True]  # dummy value, not used
                    else:
                        # 3-stream: test selected concat options
                        concat_variants = concat_choices

                    for concat_choice in concat_variants:
                        if four_stream_choice:
                            variant_tag = "4stream"
                        else:
                            variant_tag = concat_tag_map[concat_choice]

                        for opt_name in OPTIMIZERS:
                            for lr in LEARNING_RATES:
                                for bs in BATCH_SIZES:
                                    for residual_choice in residual_choices:
                                        for fusion_choice in fusion_choices:
                                            result_dir = os.path.join(
                                                RESULTS_ROOT,
                                                f"scenario_{scenario}",
                                                "MMF",
                                                ds_name,
                                                dsm_mode_tag,
                                                variant_tag,
                                                f"residual_{residual_choice}",
                                                f"fusion_{fusion_choice}",
                                                f"{opt_name}_lr{lr}_bs{bs}",
                                            )
                                            run_id = (
                                                f"scenario={scenario}|dataset={ds_name}|model=MMF|"
                                                f"dsm={dsm_mode_tag}|variant={variant_tag}|residual={residual_choice}|fusion={fusion_choice}|opt={opt_name}|lr={lr}|bs={bs}"
                                            )

                                            if run_id in completed_runs:
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
                                                run_mmf_experiment(
                                                    scenario=scenario,
                                                    dataset_name=ds_name,
                                                    X_train=X_train_proc,
                                                    Y_train=Y_train_proc,
                                                    X_val=X_val_proc,
                                                    Y_val=Y_val_proc,
                                                    X_test=X_test_proc,
                                                    Y_test=Y_test_proc,
                                                    num_classes=num_classes,
                                                    optimizer_name=opt_name,
                                                    lr=lr,
                                                    batch_size=bs,
                                                    epochs=EPOCHS,
                                                    results_dir=result_dir,
                                                    all_results_list=all_results,
                                                    include_density=include_density,
                                                    include_unc=include_unc,
                                                    concat_post_dsm=concat_choice,
                                                    four_stream=four_stream_choice,
                                                    residual=residual_choice,
                                                    fusion=fusion_choice,
                                                )

                                                completed_runs.add(run_id)
                                                save_checkpoint(completed_runs)
                                            except Exception:
                                                os.makedirs(result_dir, exist_ok=True)
                                                err_path = os.path.join(result_dir, "error.log")
                                                with open(err_path, "a") as f:
                                                    f.write("\nRun failed with exception:\n")
                                                    f.write(traceback.format_exc())
                                                print(f"Run {run_id} failed; logged to {err_path}")
                                                continue

    # After all experiments
    generate_dashboard(all_results, save_dir=RESULTS_ROOT)


if __name__ == "__main__":
    main()
