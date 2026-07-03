"""
Custom training and inference runner script for MMF-EMSNet.
Allows training the model with a custom parameter set, extracting MCMAFFusion weights,
and saving visualization images and attention weight stats for each class.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from models.mmfemsnet import (
    build_mmf_emsnet_conv,
    make_dataset as make_mmf_dataset,
    resolve_dsm_channel_indices,
)
from utils.label_processing import prepare_split_for_scenario
from utils.metrics import compute_metrics
from models.snn import load_dataset

# Global determinism settings matching the train_*.py grid-search scripts
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
tf.random.set_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Training/Optimization constants matching the train_*.py grid-search scripts
EARLY_STOPPING_PATIENCE: int = 12
LR_REDUCE_PATIENCE: int = 6
LR_REDUCE_FACTOR: float = 0.5
LR_REDUCE_MIN_LR: float = 1e-6
MIN_DELTA: float = 1e-4


def _collect_dataset_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect labels from a batched tf.data.Dataset into a flat numpy array."""
    labels: List[np.ndarray] = []
    for _, y in dataset:
        if tf.is_tensor(y):
            labels.append(y.numpy())
        else:
            labels.append(np.asarray(y))
    return np.concatenate(labels, axis=0)


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


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom Train & Inference for MMF-EMSNet")
    
    # Dataset and model args
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset/NPZ/dataset_16.npz",
        help="Path to the NPZ dataset file."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Experimental scenario (1: 0 vs 4, 2: 0-3 vs 4, 3: multiclass all 5 classes)."
    )
    
    # Training hyperparams
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "SGD", "RMSprop", "Nadam", "Adamax"],
        help="Optimizer to use."
    )
    
    # Architecture configuration
    parser.add_argument(
        "--residual",
        type=str2bool,
        default=True,
        help="Whether to use multi-level features (residual=True means 3 levels of features, otherwise 1 level)."
    )
    parser.add_argument(
        "--dsm-mode",
        type=str,
        default="dsm_density_uncertainty",
        choices=["dsm_density_uncertainty", "dsm_density", "dsm_uncertainty", "dsm_only"],
        help="DSM channels to include."
    )
    parser.add_argument(
        "--concat-post-dsm",
        type=str2bool,
        default=True,
        help="Concat post DSM with post RGB in three-stream model."
    )
    parser.add_argument(
        "--four-stream",
        type=str2bool,
        default=False,
        help="Use 4-stream model instead of 3-stream."
    )
    
    # Output and visualization args
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=5,
        help="Number of test samples of each class to save visualizations for."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="custom_results",
        help="Output directory to save model, predictions, and visualizations."
    )
    parser.add_argument(
        "--building-ids",
        type=str,
        nargs="*",
        default=[
            "10500", "10814", "11226", "11663", "12926",
            "15905", "16755", "17199", "3331", "4760",
            "6356", "648", "7276", "7959", "9533",
        ],
        help="Specific building IDs to always visualize (in addition to per-class samples)."
    )
    
    return parser.parse_args()


# load_npz_data removed to match the shared NPZ loading system (models/snn.py::load_dataset)


def get_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    opts = {
        "Adam": tf.keras.optimizers.Adam(lr),
        "SGD": tf.keras.optimizers.SGD(lr),
        "RMSprop": tf.keras.optimizers.RMSprop(lr),
        "Nadam": tf.keras.optimizers.Nadam(lr),
        "Adamax": tf.keras.optimizers.Adamax(lr),
    }
    return opts[name]


def save_visualization(
    building_id: str,
    true_class: int,
    pred_class: int,
    dsm_pre: np.ndarray,
    dsm_post: np.ndarray,
    rgb_post: np.ndarray,
    output_path: str
) -> None:
    """
    Saves a combined visualization panel showing:
    Pre DSM, Post DSM, Absolute Difference, and Post RGB.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. Pre DSM (nDSM is the first channel)
    im0 = axes[0].imshow(dsm_pre, cmap="viridis")
    axes[0].set_title("Pre DSM")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Post DSM
    im1 = axes[1].imshow(dsm_post, cmap="viridis")
    axes[1].set_title("Post DSM")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Absolute DSM difference
    diff = np.abs(dsm_post - dsm_pre)
    im2 = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("Abs DSM Diff")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Post RGB (Ensure range is [0, 1])
    rgb_clipped = np.clip(rgb_post, 0.0, 1.0)
    axes[3].imshow(rgb_clipped)
    axes[3].set_title("Post RGB")
    axes[3].axis("off")
    
    plt.suptitle(f"Building ID: {building_id} | True Class: {true_class} | Pred Class: {pred_class}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    
    # Set seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Ensure output directories exist
    output_path = Path(args.output_dir)
    vis_dir = output_path / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve DSM channel configurations
    include_density = "density" in args.dsm_mode
    include_unc = "uncertainty" in args.dsm_mode
    
    pre_idx, post_idx = resolve_dsm_channel_indices(
        include_density=include_density,
        include_unc=include_unc,
    )
    dsm_channels = len(pre_idx)
    
    # 1. Load data using the shared NPZ loading system
    print(f"Loading dataset from: {args.dataset}")
    (
        X_train_raw,
        Y_train_raw,
        X_val_raw,
        Y_val_raw,
        X_test_raw,
        Y_test_raw,
    ) = load_dataset(args.dataset)
    
    # 2. Process split for the selected scenario (filtering X, Y, and tracking building IDs)
    X_train, Y_train, num_classes, mapping = prepare_split_for_scenario(X_train_raw, Y_train_raw, args.scenario)
    X_val, Y_val, _, _ = prepare_split_for_scenario(X_val_raw, Y_val_raw, args.scenario)

    # Load building IDs from the NPZ.  After running convert_npz_compat.py the
    # bid_test array is stored as fixed-width Unicode (e.g. <U32) so it can be
    # loaded without allow_pickle, which NumPy 1.x cannot handle for object arrays
    # created by NumPy 2.x.
    try:
        _bid_data = np.load(args.dataset, allow_pickle=False)
        bid_test_raw = _bid_data["bid_test"]
        del _bid_data
        print(f"Loaded bid_test: dtype={bid_test_raw.dtype}, shape={bid_test_raw.shape}")
    except Exception as e:
        print(f"Warning: could not load bid_test ({e}). "
              "Run convert_npz_compat.py first, or use --dataset pointing to the "
              "converted file. Falling back to sequential integer indices.")
        bid_test_raw = np.arange(len(Y_test_raw), dtype=np.int64).astype(str)

    # Construct scenario-filtered test set while keeping bid_test aligned
    if args.scenario == 1:
        test_mask = np.isin(Y_test_raw, [0, 4])
        X_test = X_test_raw[test_mask]
        Y_test_raw_filtered = Y_test_raw[test_mask]
        bid_test = bid_test_raw[test_mask]
        mapping = {0: 0, 4: 1}
        Y_test = np.array([mapping[int(y)] for y in Y_test_raw_filtered], dtype=np.int64)
        num_classes = 1
    elif args.scenario == 2:
        X_test = X_test_raw
        Y_test_raw_filtered = Y_test_raw
        bid_test = bid_test_raw
        mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
        Y_test = np.array([mapping[int(y)] for y in Y_test_raw_filtered], dtype=np.int64)
        num_classes = 1
    elif args.scenario == 3:
        X_test = X_test_raw
        Y_test_raw_filtered = Y_test_raw
        bid_test = bid_test_raw
        mapping = {int(c): int(c) for c in np.unique(Y_test_raw)}
        Y_test = Y_test_raw.astype(np.int64)
        num_classes = len(mapping)

    print(f"Data shapes after processing Scenario {args.scenario}:")
    print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test:  X={X_test.shape}, Y={Y_test.shape}, Bids={bid_test.shape}")
    
    # 3. Create datasets
    train_ds = make_mmf_dataset(
        X_train, Y_train, batch=args.batch_size, shuffle=True,
        include_density=include_density, include_unc=include_unc, four_stream=args.four_stream
    )
    val_ds = make_mmf_dataset(
        X_val, Y_val, batch=args.batch_size, shuffle=False,
        include_density=include_density, include_unc=include_unc, four_stream=args.four_stream
    )
    test_ds = make_mmf_dataset(
        X_test, Y_test, batch=args.batch_size, shuffle=False,
        include_density=include_density, include_unc=include_unc, four_stream=args.four_stream
    )
    
    # 4. Build Model
    H, W = X_train.shape[1], X_train.shape[2]
    model = build_mmf_emsnet_conv(
        input_shape_dsm=(H, W, dsm_channels),
        input_shape_rgb=(H, W, 3),
        num_classes=num_classes,
        token_dim=128,
        concat_post_dsm=args.concat_post_dsm,
        four_stream=args.four_stream,
        residual=args.residual,
        fusion="mcmaf",
    )
    
    loss_fn = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=get_optimizer(args.optimizer, args.lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    # Callbacks
    callback_list, f1_callback, best_weights_path = make_training_callbacks(
        results_dir=str(output_path),
        val_data=val_ds,
        num_classes=num_classes,
    )
    
    # 5. Train
    print("Training model...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callback_list
    )
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Load best weights
    if os.path.exists(best_weights_path):
        print("Loading best weights for inference...")
        model.load_weights(best_weights_path)
    
    # Save the model
    model.save(str(output_path / "model.keras"))
    
    # 6. Inference and Weight Extraction
    print("Running inference and extracting MCMAFFusion weights...")
    # Wrap model to output both predictions and mcmaf_gates output
    gates_tensor = model.get_layer("MCMAF").output[1]
    inference_model = tf.keras.Model(inputs=model.input, outputs=[model.output, gates_tensor])
    
    # Predict on test set
    y_prob, gates_val = inference_model.predict(test_ds, verbose=0)
    
    # Post-process predictions
    if num_classes == 1:
        y_prob_arr = np.asarray(y_prob).reshape(-1)
        y_pred = (y_prob_arr >= 0.5).astype(np.int64)
        confidences = np.maximum(1.0 - y_prob_arr, y_prob_arr)
    else:
        y_prob_arr = np.asarray(y_prob)
        y_pred = np.argmax(y_prob_arr, axis=-1).astype(np.int64)
        confidences = np.max(y_prob_arr, axis=-1)
        
    # Standard metrics
    acc, prec, rec, macro_f1, mcc, cm = compute_metrics(Y_test, y_pred, num_classes)
    print(f"Test Set Performance: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, MCC={mcc:.4f}")
    
    # 7. Collect per-sample details and generate visualizations
    # We group by the original test labels (Y_test_raw_filtered) so that we visualize per original class
    unique_classes = np.unique(Y_test_raw_filtered)
    print(f"Unique classes in test set: {unique_classes}")
    
    all_sample_details = []
    target_attention_records = []  # attention weights for the 15 target buildings only
    visualized_count_by_class = {c: 0 for c in unique_classes}
    target_bids: set = set(args.building_ids) if args.building_ids else set()
    
    num_levels = gates_val.shape[1]  # 3 if residual=True, else 1
    
    for i in range(len(X_test)):
        building_id = str(bid_test[i])
        orig_class = int(Y_test_raw_filtered[i])
        scenario_class = int(Y_test[i])
        pred_class = int(y_pred[i])
        prob = float(confidences[i])

        # Extract weightings
        # gates_val[i] has shape (num_levels, 2)
        # Column 0: DSM weight, Column 1: RGB weight
        # Levels: 0=Low, 1=Medium, 2=High (if residual=True)
        low_dsm = float(gates_val[i, 0, 0])
        low_rgb = float(gates_val[i, 0, 1])

        if num_levels >= 3:
            mid_dsm = float(gates_val[i, 1, 0])
            mid_rgb = float(gates_val[i, 1, 1])
            high_dsm = float(gates_val[i, 2, 0])
            high_rgb = float(gates_val[i, 2, 1])
        else:
            mid_dsm, mid_rgb = np.nan, np.nan
            high_dsm, high_rgb = np.nan, np.nan

        sample_info = {
            "building_id": building_id,
            "original_class": orig_class,
            "scenario_target_class": scenario_class,
            "predicted_class": pred_class,
            "confidence": prob,
            "correct": (scenario_class == pred_class),
            "low_level_dsm_weight": low_dsm,
            "low_level_rgb_weight": low_rgb,
            "mid_level_dsm_weight": mid_dsm,
            "mid_level_rgb_weight": mid_rgb,
            "high_level_dsm_weight": high_dsm,
            "high_level_rgb_weight": high_rgb,
        }
        all_sample_details.append(sample_info)

        # Determine whether to save a visualization for this sample.
        # Always visualize if the building ID is in the target list;
        # otherwise fall back to the per-class sample limit.
        should_visualize = (
            building_id in target_bids
            or visualized_count_by_class[orig_class] < args.num_samples_per_class
        )

        if should_visualize:
            # Visualize nDSM (channels 6 and 10)
            dsm_pre_img = X_test[i, ..., 6]
            dsm_post_img = X_test[i, ..., 10]
            rgb_post_img = X_test[i, ..., 3:6]

            vis_filename = f"class_{orig_class}_bid_{building_id}.png"
            vis_path = vis_dir / vis_filename

            save_visualization(
                building_id=building_id,
                true_class=orig_class,
                pred_class=pred_class,
                dsm_pre=dsm_pre_img,
                dsm_post=dsm_post_img,
                rgb_post=rgb_post_img,
                output_path=str(vis_path)
            )
            if building_id not in target_bids:
                # Only count against the per-class limit for non-targeted buildings
                visualized_count_by_class[orig_class] += 1

        # Record attention weights for target buildings
        if building_id in target_bids:
            target_attention_records.append({
                "building_id": building_id,
                "low_level_dsm_weight": low_dsm,
                "low_level_rgb_weight": low_rgb,
                "mid_level_dsm_weight": mid_dsm,
                "mid_level_rgb_weight": mid_rgb,
                "high_level_dsm_weight": high_dsm,
                "high_level_rgb_weight": high_rgb,
            })
            
    # Save CSV details
    df_details = pd.DataFrame(all_sample_details)
    csv_path = output_path / "custom_inference_details.csv"
    df_details.to_csv(csv_path, index=False)
    print(f"Saved inference details CSV to: {csv_path}")
    print(f"Generated visualizations saved in: {vis_dir}")
    
    # 8. Print and save statistical summary of MCMAFFusion weights per class
    print("\n===== MCMAFFusion WEIGHTING STATISTICAL SUMMARY PER CLASS =====")
    summary_rows = []
    for c in unique_classes:
        class_df = df_details[df_details["original_class"] == c]
        if class_df.empty:
            continue
            
        mean_low_dsm = class_df["low_level_dsm_weight"].mean()
        mean_low_rgb = class_df["low_level_rgb_weight"].mean()
        
        row = {
            "original_class": c,
            "sample_count": len(class_df),
            "mean_low_dsm_weight": mean_low_dsm,
            "mean_low_rgb_weight": mean_low_rgb,
        }
        
        if num_levels >= 3:
            mean_mid_dsm = class_df["mid_level_dsm_weight"].mean()
            mean_mid_rgb = class_df["mid_level_rgb_weight"].mean()
            mean_high_dsm = class_df["high_level_dsm_weight"].mean()
            mean_high_rgb = class_df["high_level_rgb_weight"].mean()
            
            row.update({
                "mean_mid_dsm_weight": mean_mid_dsm,
                "mean_mid_rgb_weight": mean_mid_rgb,
                "mean_high_dsm_weight": mean_high_dsm,
                "mean_high_rgb_weight": mean_high_rgb,
            })
            
            print(f"Original Class {c} (N={len(class_df)}):")
            print(f"  Low Level:  DSM Weight = {mean_low_dsm:.4f} | RGB Weight = {mean_low_rgb:.4f}")
            print(f"  Mid Level:  DSM Weight = {mean_mid_dsm:.4f} | RGB Weight = {mean_mid_rgb:.4f}")
            print(f"  High Level: DSM Weight = {mean_high_dsm:.4f} | RGB Weight = {mean_high_rgb:.4f}")
        else:
            print(f"Original Class {c} (N={len(class_df)}):")
            print(f"  Low Level:  DSM Weight = {mean_low_dsm:.4f} | RGB Weight = {mean_low_rgb:.4f}")
            
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_path / "mcmaf_weights_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved weight summary CSV to: {summary_path}")
    df_target = pd.DataFrame(target_attention_records)
    target_path = output_path / "mcmaf_weights_target.csv"
    df_target.to_csv(target_path, index=False)
    print(f"Saved target weights CSV to: {target_path}")

if __name__ == "__main__":
    main()
