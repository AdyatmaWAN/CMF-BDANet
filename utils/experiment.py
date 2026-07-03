"""
Shared training-pipeline utilities used by every `train_*.py` grid-search
script: determinism setup, DSM-mode metadata, optimizer/callback factories,
prediction/report saving, and the generic grid-search driver.

IMPORTANT: `set_global_determinism()` only touches env vars/RNGs and must be
called before `import tensorflow`. Nothing at module import time here pulls
in TensorFlow — it's imported lazily inside the functions that need it — so
importing this module is always safe to do first.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
import traceback
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score

from utils.metrics import compute_metrics

SEED = 1234


# ============================================================
# Determinism
# ============================================================


def set_global_determinism(seed: int = SEED) -> None:
    """Set env vars + seed stdlib/numpy RNGs. Call before `import tensorflow`."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)


def set_tf_determinism(seed: int = SEED) -> None:
    """Seed/pin TensorFlow. Call once, right after `import tensorflow`."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# DSM mode metadata (shared ablation axis across all models)
# ============================================================

DSM_MODES: List[Dict[str, object]] = [
    {
        "key": "dsm_density_uncertainty",
        "label": "dsm+density+uncertainty",
        "include_density": True,
        "include_unc": True,
    },
    {
        "key": "dsm_density",
        "label": "dsm+density",
        "include_density": True,
        "include_unc": False,
    },
    {
        "key": "dsm_uncertainty",
        "label": "dsm+uncertainty",
        "include_density": False,
        "include_unc": True,
    },
    {
        "key": "dsm_only",
        "label": "dsm only",
        "include_density": False,
        "include_unc": False,
    },
]
DSM_MODE_BY_KEY: Dict[str, Dict[str, object]] = {m["key"]: m for m in DSM_MODES}


def dsm_mode_channels(dsm_mode: str) -> tuple[bool, bool]:
    """Return (include_density, include_unc) for a dsm_mode key."""
    mode = DSM_MODE_BY_KEY[dsm_mode]
    return bool(mode["include_density"]), bool(mode["include_unc"])


# ============================================================
# Optimizer factory (lazy TF import)
# ============================================================


def make_siamese_dsm_dataset(
    X, Y, batch_size: int, shuffle: bool, include_density: bool, include_unc: bool, seed: int = SEED,
    ordinal: bool = False, num_classes: int | None = None,
):
    """tf.data pipeline shared by FCSNN and FUJITA: both take a (pre-DSM, post-DSM)
    input pair selected via the DSM-mode channel flags, with no RGB stream.

    ordinal: If True, encode Y into CORAL's (N, num_classes-1) binary target
        matrix (see models/ordinal.py::encode_coral_labels) instead of
        leaving it as plain integer labels. Requires num_classes.
    """
    import tensorflow as tf

    from models.mmfemsnet import resolve_dsm_channel_indices

    if ordinal:
        if num_classes is None:
            raise ValueError("num_classes is required when ordinal=True")
        from models.ordinal import encode_coral_labels

        Y = encode_coral_labels(Y, num_classes)

    pre_idx, post_idx = resolve_dsm_channel_indices(include_density, include_unc)
    ds = tf.data.Dataset.from_tensor_slices(((X[..., pre_idx], X[..., post_idx]), Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed)
    ds = ds.batch(batch_size)
    try:
        opts = tf.data.Options()
        opts.experimental_threading.private_threadpool_size = 1
        opts.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
    except Exception:
        pass
    return ds.prefetch(1)


def get_optimizer(name: str, lr: float):
    from tensorflow.keras import optimizers

    opt_dict = {
        "Adam": optimizers.Adam(lr),
        "SGD": optimizers.SGD(lr),
        "RMSprop": optimizers.RMSprop(lr),
        "Nadam": optimizers.Nadam(lr),
        "Adamax": optimizers.Adamax(lr),
    }
    if name not in opt_dict:
        raise ValueError(f"Unknown optimizer: {name}")
    return opt_dict[name]


# ============================================================
# Metrics / reporting helpers
# ============================================================


def compute_f2(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if num_classes == 1:
        return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    return float(fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0))


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path_txt: str,
    path_csv: str,
) -> None:
    """Save classification report in TXT (pretty) and CSV (tabular) formats."""
    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    with open(path_txt, "w") as f:
        f.write(str(classification_report(y_true, y_pred)))
    pd.DataFrame(cr_dict).transpose().to_csv(path_csv, index=True)


def build_predictions_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    source_indices: np.ndarray,
) -> pd.DataFrame:
    """Per-sample prediction table: true/pred label, correctness, and
    per-class probabilities, keyed back to the original dataset row via
    `source_index`.
    """
    df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true), dtype=np.int64),
            "source_index": np.asarray(source_indices).astype(np.int64),
            "y_true": np.asarray(y_true).astype(np.int64),
            "y_pred": np.asarray(y_pred).astype(np.int64),
            "is_correct": (np.asarray(y_true) == np.asarray(y_pred)),
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


# ============================================================
# Training callbacks (lazy TF import)
# ============================================================


def make_training_callbacks(
    results_dir: str,
    val_data,
    num_classes: int,
    *,
    early_stopping_patience: int = 12,
    lr_reduce_patience: int = 6,
    lr_reduce_factor: float = 0.5,
    lr_reduce_min_lr: float = 1e-6,
    min_delta: float = 1e-4,
    decode_pred_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    decode_true_fn: Callable[[np.ndarray], np.ndarray] | None = None,
):
    """Build the callback list every NN training run uses: a val-F1 tracker
    driving checkpointing, early stopping, and LR reduction (all monitor
    `val_f1` rather than val_loss).

    `decode_pred_fn`/`decode_true_fn` let a caller override how raw model
    output / raw dataset labels get turned into integer class labels before
    computing val_f1 — needed for CORAL ordinal training, where both the
    model's output and the dataset's `Y` are not plain integer labels.
    Defaults reproduce the original threshold/argmax behavior.
    """
    import tensorflow as tf
    from tensorflow.keras import callbacks

    decode_pred = decode_pred_fn or (
        lambda p: (np.asarray(p).reshape(-1) >= 0.5).astype(int)
        if num_classes == 1
        else np.argmax(np.asarray(p), axis=-1)
    )
    decode_true = decode_true_fn or (lambda y: np.asarray(y))

    def _collect_dataset_labels(dataset) -> np.ndarray:
        labels: List[np.ndarray] = []
        for _, y in dataset:
            labels.append(y.numpy() if tf.is_tensor(y) else np.asarray(y))
        return np.concatenate(labels, axis=0)

    class ValidationF1Callback(callbacks.Callback):
        """Compute validation F1 at the end of each epoch and inject it into logs."""

        def __init__(self):
            super().__init__()
            self.y_true = decode_true(_collect_dataset_labels(val_data))
            self.best_val_f1 = float("-inf")
            self.best_epoch = 0

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            if logs is None:
                logs = {}
            y_prob = self.model.predict(val_data, verbose=0)
            y_pred = decode_pred(y_prob)

            _, _, _, val_f1, _, _ = compute_metrics(self.y_true, y_pred, num_classes)
            logs["val_f1"] = val_f1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch + 1
            print(f" - val_f1: {val_f1:.4f}")

    f1_callback = ValidationF1Callback()
    best_weights_path = os.path.join(results_dir, "best_weights.weights.h5")

    callback_list = [
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
            patience=early_stopping_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_f1",
            mode="max",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
            min_delta=min_delta,
            min_lr=lr_reduce_min_lr,
            verbose=1,
        ),
    ]
    return callback_list, f1_callback, best_weights_path


# ============================================================
# Shared NN compile -> train -> evaluate -> save loop
# ============================================================


def train_and_evaluate_nn(
    *,
    result_dir: str,
    model,
    model_label: str,
    num_classes: int,
    optimizer_name: str,
    learning_rate: float,
    train_ds,
    val_ds,
    test_ds,
    test_indices,
    epochs: int,
    callback_cfg: Dict,
    extra_summary: Dict,
    loss: str | None = None,
    decode_pred_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    decode_true_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    class_probs_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    extra_metrics_fn: Callable[[np.ndarray, np.ndarray], Dict] | None = None,
) -> Dict:
    """Compile, train, evaluate, and save everything for one already-built
    (uncompiled) Keras model. This is the block every NN `train_*.py` script
    (FCSNN, MMF-EMSNet, FUJITA) needs identically; only how the model and
    datasets are built differs between them, which stays in each script.

    `extra_summary` supplies whatever identity/ablation columns the caller
    wants in metrics.csv (scenario, dataset, dsm_mode_tag, batch_size, and
    any model-specific ablation fields like residual/fusion/variant_tag) —
    this function only knows about what it directly computes.

    The five optional hooks below all default to None, reproducing the
    original nominal-classification behavior exactly. They exist for CORAL
    ordinal training (see train_mmfemsnet.py), where the loss, the
    model-output -> label decoding, and the model-output -> per-class
    probability conversion are all different from plain softmax/sigmoid:
        loss             - override the auto-selected loss.
        decode_pred_fn   - model output probs -> integer y_pred.
        decode_true_fn   - raw dataset Y -> integer y_true.
        class_probs_fn   - model output probs -> per-class probability
                           matrix for predictions.csv (identity by default).
        extra_metrics_fn - (y_true_int, y_pred_int) -> extra summary fields,
                           e.g. {"mae": ..., "qwk": ...}.
    """
    os.makedirs(result_dir, exist_ok=True)

    if loss is None:
        loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer=get_optimizer(optimizer_name, learning_rate), loss=loss, metrics=["accuracy"])

    decode_pred = decode_pred_fn or (
        lambda p: (np.asarray(p).reshape(-1) > 0.5).astype(int)
        if num_classes == 1
        else np.argmax(np.asarray(p), axis=-1)
    )
    decode_true = decode_true_fn or (lambda y: np.asarray(y))
    to_class_probs = class_probs_fn or (lambda p: p)

    training_callbacks, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=result_dir, val_data=val_ds, num_classes=num_classes,
        decode_pred_fn=decode_pred_fn, decode_true_fn=decode_true_fn,
        **callback_cfg,
    )

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1, callbacks=training_callbacks)
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)
    train_time = time.time() - t0

    y_prob = model.predict(test_ds)
    y_pred = decode_pred(y_prob)
    y_true = decode_true(np.concatenate([y for (_, y) in test_ds], axis=0))

    acc, prec, rec, f1, mcc, cm = compute_metrics(y_true, y_pred, num_classes)
    macro_f2 = compute_f2(y_true, y_pred, num_classes)

    pred_df = build_predictions_frame(y_true, y_pred, to_class_probs(y_prob), num_classes, test_indices)
    predictions_csv = os.path.join(result_dir, "predictions.csv")
    pred_df.to_csv(predictions_csv, index=False)
    save_classification_report(
        y_true, y_pred,
        os.path.join(result_dir, "classification_report.txt"),
        os.path.join(result_dir, "classification_report.csv"),
    )

    model.save(os.path.join(result_dir, "model.keras"))
    model.save_weights(os.path.join(result_dir, "weights.h5"))

    summary = {
        **extra_summary,
        "model": model_label,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "num_classes": num_classes,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "macro_f2": macro_f2,
        "mcc": mcc,
        **(extra_metrics_fn(y_true, y_pred) if extra_metrics_fn else {}),
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
        "predictions_csv": predictions_csv,
        "result_dir": result_dir,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(result_dir, "metrics.csv"), index=False)

    print(f"[{model_label}] ACC={acc:.4f} F1={f1:.4f} MCC={mcc:.4f}")
    return summary


# ============================================================
# Resume / skip-if-already-trained + grid-search driver
# ============================================================


def is_already_trained(result_dir: str) -> bool:
    """A run is considered done once its metrics.csv exists — the result
    directory path already encodes every grid parameter, so this is a
    complete resume mechanism without needing a separate checkpoint file.
    """
    return os.path.exists(os.path.join(result_dir, "metrics.csv"))


def write_run_error(result_dir: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "error.log"), "a") as f:
        f.write("\nRun failed with exception:\n")
        f.write(traceback.format_exc())


def _combos(grid: Dict[str, Sequence]) -> List[Dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def grid_search(
    ablation_grid: Dict[str, Sequence],
    hpo_grid: Dict[str, Sequence],
    result_dir_fn: Callable[..., str],
    train_one_fn: Callable[..., Dict],
) -> List[Dict]:
    """Iterate the cartesian product of `ablation_grid` x `hpo_grid`.

    For each combination of parameters (all ablation + hpo keys merged into
    one kwargs dict):
      - build `result_dir = result_dir_fn(**params)`
      - skip training (but recover its metrics row) if already trained
      - otherwise call `train_one_fn(result_dir=result_dir, **params)`,
        which must train, evaluate, save metrics.csv/predictions.csv itself,
        and return the summary dict
      - on exception, log to `result_dir/error.log` and continue to the next
        combination rather than aborting the whole grid search

    Returns the summary dicts for every completed (or already-completed) run.
    """
    results: List[Dict] = []
    for ablation_params in _combos(ablation_grid):
        for hpo_params in _combos(hpo_grid):
            params = {**ablation_params, **hpo_params}
            result_dir = result_dir_fn(**params)

            if is_already_trained(result_dir):
                print(f"Skipping already-trained run: {result_dir}")
                try:
                    df = pd.read_csv(os.path.join(result_dir, "metrics.csv"))
                    results.extend(df.to_dict(orient="records"))
                except Exception:
                    pass
                continue

            try:
                summary = train_one_fn(result_dir=result_dir, **params)
                results.append(summary)
            except Exception:
                write_run_error(result_dir)
                print(f"Run failed; logged to {os.path.join(result_dir, 'error.log')}")
                continue
    return results


# ============================================================
# Post-run aggregation
# ============================================================


def aggregate_scenarios_after_run(results_root: str, scenarios: Sequence[int], best_metric: str = "f1") -> None:
    """Call once at the end of a train_*.py script's main(): re-aggregate
    every scenario that was part of this run, across every model that has
    results there (not just the one that just trained), so
    `results/scenario_N/aggregated_metrics.csv` and `best_overall.csv` stay
    current without a separate manual `tools/aggregate_metrics.py` step.

    Scoping the scan to `results/scenario_N/` rather than all of `results/`
    is what makes this "models in the same scenario" rather than
    all-scenarios-mixed-together — every model's results already live nested
    under that per-scenario directory (see each train_*.py's build_result_dir).
    """
    from pathlib import Path

    from tools.aggregate_metrics import aggregate_and_write

    for scenario in scenarios:
        scenario_root = Path(results_root) / f"scenario_{scenario}"
        if not scenario_root.exists():
            continue
        per_ablation_best, best_overall = aggregate_and_write(scenario_root, best_metric=best_metric)
        if per_ablation_best.empty:
            continue
        print(
            f"[aggregate] scenario {scenario}: {len(per_ablation_best)} ablation-config rows, "
            f"{len(best_overall)} best-overall rows -> {scenario_root / 'aggregated_metrics.csv'}"
        )
