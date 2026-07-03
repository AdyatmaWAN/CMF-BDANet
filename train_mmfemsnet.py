"""
MMF-EMSNet grid-search training entrypoint.

Same shape as train_fcsnn.py: reads its grid from a JSON config (default
configs/mmfemsnet.json), trains+tests every combination, writes
metrics.csv + predictions.csv + classification report + model artifacts,
and skips runs whose metrics.csv already exists.

The "variant" ablation axis encodes the 3-stream-with-concat /
3-stream-without-concat / 4-stream choice as a single categorical value
(rather than two independent booleans) so the grid doesn't produce the
redundant 4-stream/no-concat duplicate of 4-stream/concat.

Usage:
    python train_mmfemsnet.py
    python train_mmfemsnet.py --config configs/mmfemsnet.json --scenario 1
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from typing import Dict, Tuple

from utils.experiment import set_global_determinism

set_global_determinism()

import pandas as pd  # noqa: E402

from utils.experiment import (  # noqa: E402
    DSM_MODES,
    dsm_mode_channels,
    grid_search,
    set_tf_determinism,
    train_and_evaluate_nn,
)
from models.fcsnn import load_dataset  # noqa: E402
from models.mmfemsnet import (  # noqa: E402
    build_mmf_emsnet_conv,
    make_dataset as make_mmf_dataset,
    resolve_dsm_channel_indices,
)
from utils.label_processing import prepare_split_with_indices  # noqa: E402

set_tf_determinism()

VARIANTS = ("3stream_concat", "3stream_no_concat", "4stream")


def decode_variant(variant: str) -> Tuple[bool, bool]:
    """Return (four_stream, concat_post_dsm)."""
    if variant == "4stream":
        return True, True  # concat_post_dsm is unused in 4-stream mode
    if variant == "3stream_concat":
        return False, True
    if variant == "3stream_no_concat":
        return False, False
    raise ValueError(f"Unknown MMF variant: {variant}")


def build_result_dir(
    results_root: str,
    scenario: int,
    dataset_name: str,
    *,
    dsm_mode: str,
    variant: str,
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
        "MMF",
        dataset_name,
        dsm_mode,
        variant,
        f"residual_{residual}",
        f"fusion_{fusion}",
        f"{optimizer}_lr{learning_rate}_bs{batch_size}",
    )


def train_one(
    *,
    result_dir: str,
    dsm_mode: str,
    variant: str,
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
    include_density, include_unc = dsm_mode_channels(dsm_mode)
    four_stream, concat_post_dsm = decode_variant(variant)

    print(
        f"\n===== [MMF] scenario={scenario} dataset={dataset_name} dsm={dsm_mode} variant={variant} "
        f"residual={residual} fusion={fusion} opt={optimizer} lr={learning_rate} bs={batch_size} ====="
    )

    train_ds = make_mmf_dataset(X_train, Y_train, batch_size, True, include_density, include_unc, four_stream)
    val_ds = make_mmf_dataset(X_val, Y_val, batch_size, False, include_density, include_unc, four_stream)
    test_ds = make_mmf_dataset(X_test, Y_test, batch_size, False, include_density, include_unc, four_stream)

    H, W = X_train.shape[1], X_train.shape[2]
    dsm_channels = len(resolve_dsm_channel_indices(include_density, include_unc)[0])

    set_tf_determinism(seed)
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

    return train_and_evaluate_nn(
        result_dir=result_dir,
        model=model,
        model_label="MMF",
        num_classes=num_classes,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        test_indices=test_indices,
        epochs=epochs,
        callback_cfg=callback_cfg,
        extra_summary={
            "scenario": scenario,
            "dataset": dataset_name,
            "dsm_mode_tag": dsm_mode,
            "variant_tag": variant,
            "dsm_post_concat_with_rgb": concat_post_dsm,
            "use_four_stream": four_stream,
            "residual": residual,
            "fusion": fusion,
            "batch_size": batch_size,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mmfemsnet.json")
    parser.add_argument("--scenario", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--dsm-mode", type=str, default=None, choices=[m["key"] for m in DSM_MODES])
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS)
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
    if args.variant is not None:
        ablation_grid["variant"] = [args.variant]

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
        pd.DataFrame(all_results).to_csv(os.path.join(results_root, "MMF_grid_summary.csv"), index=False)


if __name__ == "__main__":
    main()
