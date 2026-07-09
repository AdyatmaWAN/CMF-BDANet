"""
BDD-Net model definition and dataset utilities.

Seydi, S.T., Rastiveis, H., Kalantar, B., Halin, A.A. & Ueda, N. (2022).
"BDD-Net: An End-to-End Multiscale Residual CNN for Earthquake-Induced
Building Damage Detection." Remote Sensing, 14(9), 2214.
https://doi.org/10.3390/rs14092214

Architecture (paper Figure 5): three parallel streams —
- Optical stream: 5 multi-scale/residual conv stages + 3 max-pools (Figure 6).
- Lidar stream: identical stage layout, separate weights.
- Fusion stream: 8 fusion nodes, matching every block in the optic/lidar rows
  (5 conv-stage nodes + 3 post-pool nodes) — at each, it concatenates the
  optic/lidar taps at that point with its own running output and applies a
  convolution layer ("repeated for other convolution layers", Section 2.3.4)
  — fusion happens at every level, not just first/last, per the paper's main
  claimed novelty.
Each stream ends in a depth-wise convolution (Section 2.3.3 point 4); the
three resulting deep-feature maps are flattened, concatenated, and fed to a
1500-unit fully-connected layer (paper Section 3.1) before the classifier.

Stage 1 uses the Multi-scale Shallow Block (Figure 6a: parallel 5x5/7x7/3x3/
dilated-3x3 convs, concatenated). Stages 2-5 use the Multi-scale Residual
Dilated Convolution Block (Figure 6b: parallel 5x5/3x3/dilated-3x3(rate2)/
dilated-3x3(rate4) convs concatenated, added to a depth-wise-separable(9x9)
skip path per the residual mechanism in Figure 7).

Dataset adaptation: the paper fuses a single post-event Optical (50x50x4:
RGB+NIR) patch with a single post-event Lidar (50x50x1) patch — it is a
damage *classification* network, not a change-detection one. This repo's
NPZ has no Lidar band and no NIR band, so the two streams are mapped onto
the nearest available modalities: post-event RGB (channels 3-5) as
"optical", and post-event nDSM (+ optional density/uncertainty, channels
10-12, see models/mmfemsnet.py) as "lidar" (both are post-event 3D surface
proxies). Pre-event channels are unused, matching the paper (no pre/post
pair).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from models.mmfemsnet import POST_NDSM_IDX, POST_DENSITY_IDX, POST_UNC_IDX
from models.ordinal import add_coral_head, encode_coral_labels

SEED = 1234
tf.random.set_seed(SEED)

# Per-stage (base_filters multiplier, pool-after-this-stage) — 5 stages, 3 pools,
# per "five multi-scale/residual convolution layers with three pooling layers"
# (Section 2.3.3).
_STAGE_MULT = (1, 2, 4, 4, 4)
_STAGE_POOL = (True, True, True, False, False)


# ============================================================
# 1. BUILDING BLOCKS
# ============================================================


def _init(seed: int = SEED):
    return tf.keras.initializers.GlorotUniform(seed=seed)


def conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: int, name_prefix: str, dilation_rate: int = 1) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        name=f"{name_prefix}_conv",
        kernel_initializer=_init(),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    return layers.ReLU(name=f"{name_prefix}_relu")(x)


def depthwise_separable(x: tf.Tensor, filters: int, kernel_size: int, name_prefix: str) -> tf.Tensor:
    """Depth-wise conv (Section 2.3.3 point 4 / Figure 9) followed by a 1x1
    pointwise projection to `filters` channels (MobileNets [39])."""
    x = layers.DepthwiseConv2D(
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_dwconv",
        depthwise_initializer=_init(),
    )(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name_prefix}_pwconv", kernel_initializer=_init())(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    return layers.ReLU(name=f"{name_prefix}_relu")(x)


def multiscale_shallow_block(x: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
    """Figure 6a: parallel Conv5x5/Conv7x7/Conv3x3/Dilated-Conv3x3(rate2), concatenated."""
    branches = [
        conv_bn_relu(x, filters, 5, f"{name_prefix}_c5"),
        conv_bn_relu(x, filters, 7, f"{name_prefix}_c7"),
        conv_bn_relu(x, filters, 3, f"{name_prefix}_c3"),
        conv_bn_relu(x, filters, 3, f"{name_prefix}_d2", dilation_rate=2),
    ]
    return layers.Concatenate(name=f"{name_prefix}_concat")(branches)


def multiscale_residual_block(x: tf.Tensor, filters: int, name_prefix: str) -> tf.Tensor:
    """Figure 6b: concat(Conv5x5, Conv3x3, D-Conv3x3 rate2, D-Conv3x3 rate4)
    added (Figure 7-style residual) to a depth-wise-separable(9x9) skip path."""
    branches = [
        conv_bn_relu(x, filters, 5, f"{name_prefix}_c5"),
        conv_bn_relu(x, filters, 3, f"{name_prefix}_c3"),
        conv_bn_relu(x, filters, 3, f"{name_prefix}_d2", dilation_rate=2),
        conv_bn_relu(x, filters, 3, f"{name_prefix}_d4", dilation_rate=4),
    ]
    main = layers.Concatenate(name=f"{name_prefix}_concat")(branches)
    skip = depthwise_separable(x, filters * 4, 9, f"{name_prefix}_skip")
    out = layers.Add(name=f"{name_prefix}_add")([main, skip])
    return layers.ReLU(name=f"{name_prefix}_out_relu")(out)


def stream_backbone(
    x: tf.Tensor, base_filters: int, name_prefix: str
) -> Tuple[List[tf.Tensor], List[tf.Tensor], tf.Tensor]:
    """Five multi-scale/residual stages + 3 interleaved max-pools + a final
    depth-wise conv (Figure 5, one Optical/Lidar channel). Returns the
    per-stage taps (pre-pool) and per-pool taps (post-pool) used by the
    fusion stream's 8 nodes, plus the final deep features.
    """
    stage_outputs = []
    pooled_outputs = []
    for i, (mult, pool) in enumerate(zip(_STAGE_MULT, _STAGE_POOL), start=1):
        filters = base_filters * mult
        block = multiscale_shallow_block if i == 1 else multiscale_residual_block
        x = block(x, filters, f"{name_prefix}_s{i}")
        stage_outputs.append(x)
        if pool:
            x = layers.MaxPooling2D(2, name=f"{name_prefix}_pool{i}")(x)
            pooled_outputs.append(x)
    deep_features = depthwise_separable(x, base_filters * 4, 9, f"{name_prefix}_deep")
    return stage_outputs, pooled_outputs, deep_features


def fusion_stream(
    optic_stages: List[tf.Tensor],
    optic_pooled: List[tf.Tensor],
    lidar_stages: List[tf.Tensor],
    lidar_pooled: List[tf.Tensor],
    base_filters: int,
) -> tf.Tensor:
    """Fusion channel (Section 2.3.4): 8 nodes matching every block in the
    optic/lidar rows of Figure 5 — one after each of the 5 conv stages (same
    block type used at that level) and one after each of the 3 pooling steps
    (a lighter 3x3 conv, since a pool doesn't introduce new features, just
    re-syncs resolution). Each node concatenates the optic/lidar taps at that
    point with the fusion stream's own running output, then convolves.
    """
    fused = None
    pool_idx = 0
    for i, (mult, pool) in enumerate(zip(_STAGE_MULT, _STAGE_POOL), start=1):
        filters = base_filters * mult
        parts = [optic_stages[i - 1], lidar_stages[i - 1]]
        if fused is not None:
            parts.append(fused)
        merged = layers.Concatenate(name=f"fusion_s{i}_in")(parts)
        block = multiscale_shallow_block if i == 1 else multiscale_residual_block
        fused = block(merged, filters, f"fusion_s{i}")
        if pool:
            fused = layers.MaxPooling2D(2, name=f"fusion_pool{i}")(fused)
            pool_idx += 1
            merged = layers.Concatenate(name=f"fusion_pool{i}_in")(
                [optic_pooled[pool_idx - 1], lidar_pooled[pool_idx - 1], fused]
            )
            fused = conv_bn_relu(merged, filters, 3, f"fusion_pool{i}_conv")
    return depthwise_separable(fused, base_filters * 4, 9, "fusion_deep")


# ============================================================
# 2. MODEL DEFINITION (BDD-Net)
# ============================================================


def build_bddnet_model(
    input_shape_optical: Tuple[int, int, int],
    input_shape_lidar: Tuple[int, int, int],
    num_classes: int,
    base_filters: int = 16,
    fc_units: int = 1500,
    ordinal: bool = False,
) -> tf.keras.Model:
    """
    Build the BDD-Net model (Figure 5).

    Args:
        input_shape_optical: Shape of the optical input, e.g. (H, W, 3).
        input_shape_lidar: Shape of the lidar/DSM input, e.g. (H, W, 1).
        num_classes: 1 or 2 -> binary (sigmoid); >2 -> multi-class (softmax),
            or ordinal (CORAL) when ordinal=True.
        base_filters: Filter count for stage 1 (doubles at stages 2-3 then
            holds, matching Section 2.3.3's five-stage layout). Not specified
            numerically in the paper.
        fc_units: Fully-connected layer width — 1500, per Section 3.1.
        ordinal: If True, replace the sigmoid/softmax head with a CORAL
            ordinal head (see models/ordinal.py). Requires num_classes >= 3.

    Returns:
        Keras Model named "BDD_Net" with inputs [optical_input, lidar_input].
    """
    optic_in = layers.Input(input_shape_optical, name="optical_input")
    lidar_in = layers.Input(input_shape_lidar, name="lidar_input")

    optic_stages, optic_pooled, optic_deep = stream_backbone(optic_in, base_filters, "optic")
    lidar_stages, lidar_pooled, lidar_deep = stream_backbone(lidar_in, base_filters, "lidar")
    fusion_deep = fusion_stream(optic_stages, optic_pooled, lidar_stages, lidar_pooled, base_filters)

    merged = layers.Concatenate(name="deep_feature_concat")(
        [
            layers.Flatten(name="optic_flatten")(optic_deep),
            layers.Flatten(name="lidar_flatten")(lidar_deep),
            layers.Flatten(name="fusion_flatten")(fusion_deep),
        ]
    )
    x = layers.Dense(fc_units, activation="relu", name="fc", kernel_initializer=_init())(merged)
    x = layers.Dropout(0.5, name="fc_dropout", seed=SEED)(x)

    if ordinal:
        out = add_coral_head(x, num_classes, seed=SEED)
    elif num_classes in (1, 2):
        out = layers.Dense(1, activation="sigmoid", name="output", kernel_initializer=_init())(x)
    else:
        out = layers.Dense(num_classes, activation="softmax", name="output", kernel_initializer=_init())(x)

    return models.Model(inputs=[optic_in, lidar_in], outputs=out, name="BDD_Net")


# ============================================================
# 3. DATA PIPELINE
# ============================================================


def resolve_lidar_channel_indices(include_density: bool = True, include_unc: bool = True) -> List[int]:
    indices = [POST_NDSM_IDX]
    if include_density:
        indices.append(POST_DENSITY_IDX)
    if include_unc:
        indices.append(POST_UNC_IDX)
    return indices


def extract_inputs(X: np.ndarray, include_density: bool = True, include_unc: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Optical stream = post-event RGB (channels 3-5); Lidar stream =
    post-event nDSM (+ optional density/uncertainty, channels 10-12) — see
    module docstring for why these stand in for the paper's Optical/Lidar."""
    lidar_idx = resolve_lidar_channel_indices(include_density, include_unc)
    if max(lidar_idx) >= X.shape[-1]:
        raise ValueError(f"Input has {X.shape[-1]} channels, but lidar indices require up to {max(lidar_idx)}.")
    optical = X[..., 3:6]
    lidar = X[..., lidar_idx]
    return optical, lidar


def make_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch: int,
    shuffle: bool = True,
    include_density: bool = True,
    include_unc: bool = True,
    ordinal: bool = False,
    num_classes: int | None = None,
) -> tf.data.Dataset:
    """Build tf.data.Dataset for BDD-Net training/evaluation, yielding
    ((optical, lidar), Y)."""
    if ordinal:
        if num_classes is None:
            raise ValueError("num_classes is required when ordinal=True")
        Y = encode_coral_labels(Y, num_classes)

    optical, lidar = extract_inputs(X, include_density=include_density, include_unc=include_unc)
    ds = tf.data.Dataset.from_tensor_slices(((optical, lidar), Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(batch)

    try:
        opts = tf.data.Options()
        opts.experimental_threading.private_threadpool_size = 1
        opts.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
    except Exception:
        pass

    return ds.prefetch(1)


if __name__ == "__main__":
    opt_shape, lidar_shape = (16, 16, 3), (16, 16, 1)

    for num_classes in (1, 2, 5):
        model = build_bddnet_model(opt_shape, lidar_shape, num_classes)
        assert model.input_shape == [(None, *opt_shape), (None, *lidar_shape)]
        units = model.output_shape[-1]
        actv = model.layers[-1].activation.__name__
        expected_units, expected_actv = (1, "sigmoid") if num_classes in (1, 2) else (num_classes, "softmax")
        assert (units, actv) == (expected_units, expected_actv), (num_classes, units, actv)
    print("bddnet.build_bddnet_model self-check OK: num_classes in {1,2} -> sigmoid(1), else -> softmax(n)")

    ordinal_model = build_bddnet_model(opt_shape, lidar_shape, 5, ordinal=True)
    assert ordinal_model.output_shape[-1] == 4, ordinal_model.output_shape
    assert ordinal_model.layers[-1].__class__.__name__ == "CoralBiases", ordinal_model.layers[-1]
    print("bddnet.build_bddnet_model ordinal self-check OK: num_classes=5, ordinal=True -> CoralBiases(4 thresholds)")

    X_dummy = np.random.rand(4, 16, 16, 14).astype(np.float32)
    optical, lidar = extract_inputs(X_dummy)
    assert optical.shape == (4, 16, 16, 3) and lidar.shape == (4, 16, 16, 3)
    preds = build_bddnet_model((16, 16, 3), (16, 16, 3), 5).predict([optical, lidar], verbose=0)
    assert preds.shape == (4, 5) and np.allclose(preds.sum(axis=1), 1.0, atol=1e-4)
    print("bddnet.extract_inputs + forward pass self-check OK: (4,16,16,14) -> optical(4,16,16,3), lidar(4,16,16,3) -> preds(4,5)")

    build_bddnet_model((16, 16, 3), (16, 16, 3), 5).summary()