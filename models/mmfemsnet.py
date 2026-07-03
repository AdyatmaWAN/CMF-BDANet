"""
MMF-EMSNet model definition and dataset utilities.

Inputs:
- DSM pre-event (channel 6-8, dynamically chosen channel)
- DSM post-event (channel 10-12, dynamically chosen channel)
- RGB post-event (channels 3–5)

Supports:
- Binary classification (num_classes=1) via sigmoid.
- Multi-class classification (num_classes>1) via softmax.
- Ordinal regression (ordinal=True) via a CORAL head (see models/ordinal.py).

Channel convention:
    0–2 : pre RGB  (unused here)
    3–5 : post RGB
    6   : pre DSM
    7   : pre density
    8   : pre uncertainty
    9   : pre dtm fill
    10  : post DSM
    11  : post density
    12  : post uncertainty
    13  : post dtm fill
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from models.ordinal import add_coral_head, encode_coral_labels

SEED = 1234
tf.random.set_seed(SEED)

# Fused NPZ channel map (see dataset builder/pipeline)
PRE_NDSM_IDX = 6
PRE_DENSITY_IDX = 7
PRE_UNC_IDX = 8
POST_NDSM_IDX = 10
POST_DENSITY_IDX = 11
POST_UNC_IDX = 12


# ============================================================
# 1. MODEL DEFINITION (MMF-EMSNet)
# ============================================================


def bdd_conv_block(
    x: tf.Tensor,
    filters: int,
    name_prefix: str,
) -> tf.Tensor:
    """
    Basic convolutional block used in BDDNet-style backbones.

    Structure:
        Conv(3x3) → BN → ReLU
        Conv(3x3) → BN → ReLU
        Conv(3x3) → BN → ReLU
        MaxPool(2x2)
        Dropout(0.5)
    """
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv1",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv2",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name_prefix}_conv3",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu3")(x)

    x = layers.MaxPooling2D(2, name=f"{name_prefix}_pool")(x)
    x = layers.Dropout(0.5, name=f"{name_prefix}_dropout", seed=SEED)(x)
    return x


def encoder_backbone(
    x: tf.Tensor,
    base_filters: int = 32,
    name_prefix: str = "enc",
) -> List[tf.Tensor]:
    """
    Simple 3-level encoder returning multi-scale feature maps.

    Levels:
        l1: base_filters
        l2: base_filters * 2
        l3: base_filters * 4

    Returns:
        List [f1, f2, f3] of feature maps.
    """
    feats: List[tf.Tensor] = []

    x = bdd_conv_block(x, base_filters, f"{name_prefix}_l1")
    feats.append(x)

    x = bdd_conv_block(x, base_filters * 2, f"{name_prefix}_l2")
    feats.append(x)

    x = bdd_conv_block(x, base_filters * 4, f"{name_prefix}_l3")
    feats.append(x)

    return feats


class MCMAFFusion(layers.Layer):
    """
    Multi-level Cross-Modal Attention Fusion (MCMAF) layer.

    At each level:
        - Project DSM and RGB feats to token_dim.
        - Global average pool → tokens.
        - Stack DSM/RGB tokens, compute softmax attention.
        - Weighted sum → fused token.

    Output is concatenation of fused tokens across all levels.
    """

    def __init__(self, token_dim: int = 128, num_levels: int = 3, name: str = "MCMAF"):
        super().__init__(name=name)
        self.num_levels = num_levels
        self.token_dim = token_dim

        self.conv_dsm: List[layers.Conv2D] = [
            layers.Conv2D(
                token_dim,
                1,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            )
            for _ in range(num_levels)
        ]
        self.conv_rgb: List[layers.Conv2D] = [
            layers.Conv2D(
                token_dim,
                1,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            )
            for _ in range(num_levels)
        ]

        self.gap = layers.GlobalAveragePooling2D()
        self.score_dense = layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        )

    def call(
        self,
        inputs: Tuple[Iterable[tf.Tensor], Iterable[tf.Tensor]],
        *args,
        **kwargs,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            inputs: (dsm_feats, rgb_feats)
                Each is a list/tuple of tensors with shape (B, H_i, W_i, C_i)

        Returns:
            Fused representation: (B, num_levels * token_dim)
            Gates: (B, num_levels, 2)
        """
        dsm_feats, rgb_feats = inputs
        fused_tokens: List[tf.Tensor] = []
        gates_list: List[tf.Tensor] = []

        for i in range(self.num_levels):
            td = self.gap(self.conv_dsm[i](dsm_feats[i]))
            tr = self.gap(self.conv_rgb[i](rgb_feats[i]))

            tokens = tf.stack([td, tr], axis=1)  # (B, 2, token_dim)
            scores = self.score_dense(tokens)    # (B, 2, 1)
            gates = tf.nn.softmax(scores, axis=1)

            fused_i = tf.reduce_sum(tokens * gates, axis=1)  # (B, token_dim)
            fused_tokens.append(fused_i)
            gates_list.append(tf.squeeze(gates, axis=-1))

        gates_all = tf.stack(gates_list, axis=1)
        return tf.concat(fused_tokens, axis=-1), gates_all


def build_mmf_emsnet_conv(
    input_shape_dsm: Tuple[int, int, int],
    input_shape_rgb: Tuple[int, int, int],
    num_classes: int,
    token_dim: int = 128,
    concat_post_dsm: bool = True,
    four_stream: bool = False,
    residual: bool = True,
    fusion: str = "mcmaf",
    ordinal: bool = False,
) -> tf.keras.Model:
    """
    Build MMF-EMSNet model using convolutional encoders.

    Inputs:
        - DSM pre-event (H, W, 1)
        - DSM post-event (H, W, 1)
        - RGB post-event (H, W, 3)

    Args:
        input_shape_dsm: Shape of DSM inputs, e.g. (H, W, 1).
        input_shape_rgb: Shape of RGB inputs, e.g. (H, W, 3).
        num_classes: 1 for binary (sigmoid), >1 for multi-class (softmax)
            or ordinal (CORAL) when ordinal=True.
        token_dim: Dimension of token embeddings in fusion.
        ordinal: If True, replace the softmax head with a CORAL ordinal
            head (see models/ordinal.py). Requires num_classes >= 3.

    Returns:
        Keras Model named "MMF_EMSNet".
    """
    # Inputs
    if four_stream:
        inp_pre = layers.Input(input_shape_dsm, name="dsm_pre")
        inp_post = layers.Input(input_shape_dsm, name="dsm_post")
        inp_rgb_pre = layers.Input(input_shape_rgb, name="rgb_pre")
        inp_rgb_post = layers.Input(input_shape_rgb, name="rgb_post")
    else:
        inp_pre = layers.Input(input_shape_dsm, name="dsm_pre")
        inp_post = layers.Input(input_shape_dsm, name="dsm_post")
        inp_rgb = layers.Input(input_shape_rgb, name="rgb_post")

    # DSM Siamese encoder
    dsm_in = layers.Input(input_shape_dsm, name="dsm_encoder_input")
    dsm_feats = encoder_backbone(dsm_in, base_filters=32, name_prefix="dsm")
    dsm_encoder = models.Model(dsm_in, dsm_feats, name="dsm_encoder")

    f_pre = dsm_encoder(inp_pre)
    f_post = dsm_encoder(inp_post)

    dsm_change: List[tf.Tensor] = []
    for i in range(3):
        diff = layers.Subtract(name=f"dsm_diff_l{i+1}")([f_post[i], f_pre[i]])
        absd = layers.Lambda(lambda x: tf.abs(x), name=f"dsm_abs_l{i+1}")(diff)
        dsm_change.append(layers.Concatenate(name=f"dsm_cat_l{i+1}")([diff, absd]))

    # RGB encoder. Two modes:
    # - three-stream (default): a single post-RGB input (optionally concatenated
    #   with post-DSM) is encoded by an RGB encoder.
    # - four-stream: a siamese RGB encoder processes pre/post RGB separately
    #   (weights shared between the two RGB streams but different from DSM encoder
    #   weights). The rgb_change features (diff+abs) are computed similarly to DSM.
    if four_stream:
        # Build RGB siamese encoder (shared weights between pre/post RGB)
        rgb_in = layers.Input(input_shape_rgb, name="rgb_encoder_input")
        rgb_feats = encoder_backbone(rgb_in, base_filters=32, name_prefix="rgb")
        rgb_encoder = models.Model(rgb_in, rgb_feats, name="rgb_encoder")

        f_rgb_pre = rgb_encoder(inp_rgb_pre)
        f_rgb_post = rgb_encoder(inp_rgb_post)

        rgb_change: List[tf.Tensor] = []
        for i in range(3):
            diff = layers.Subtract(name=f"rgb_diff_l{i+1}")([f_rgb_post[i], f_rgb_pre[i]])
            absd = layers.Lambda(lambda x: tf.abs(x), name=f"rgb_abs_l{i+1}")(diff)
            rgb_change.append(layers.Concatenate(name=f"rgb_cat_l{i+1}")([diff, absd]))

    else:
        # three-stream behavior (keep previous semantics)
        if concat_post_dsm:
            rgbd = layers.Concatenate(name="rgbd_concat")([inp_rgb, inp_post])
        else:
            rgbd = inp_rgb
        rgb_in = layers.Input(rgbd.shape[1:], name="rgb_encoder_input")
        rgb_feats = encoder_backbone(rgb_in, base_filters=32, name_prefix="rgb")
        rgb_encoder = models.Model(rgb_in, rgb_feats, name="rgb_encoder")
        f_rgb = rgb_encoder(rgbd)


    # Fusion — support 'mcmaf' (attention-style fusion) or 'concat' (simple
    # global-pooled concatenation of tokens). Default preserves previous
    # behaviour ('mcmaf'). The 'residual' flag controls how many levels are
    # considered for mcmaf (3) vs 1 when not residual.
    if fusion not in ("mcmaf", "concat"):
        raise ValueError("fusion must be 'mcmaf' or 'concat'")

    if fusion == "mcmaf":
        num_levels = 1 if not residual else 3
        fusion_layer = MCMAFFusion(token_dim=token_dim, num_levels=num_levels)
        if four_stream:
            if not residual:
                # When not residual=True, rgb_change is a single tensor, wrap it in a list
                z, gates_all = fusion_layer(([dsm_change[-1]], [rgb_change[-1]]))
            else:
                z, gates_all = fusion_layer((dsm_change, rgb_change))
        else:
            if not residual:
                # When not residual=True, f_rgb is a single tensor, wrap it in a list
                z, gates_all = fusion_layer(([dsm_change[-1]], [f_rgb[-1]]))
            else:
                z, gates_all = fusion_layer((dsm_change, f_rgb))
    else:  # concat
        # Convert per-level feature maps to tokens via global average pool and
        # then concatenate DSM and RGB tokens across levels/modalities.
        gap = layers.GlobalAveragePooling2D()
        d_tokens = [gap(t) for t in dsm_change]
        if four_stream:
            r_tokens = [gap(t) for t in rgb_change]
        else:
            # f_rgb might be a list of level tensors or a single tensor
            if isinstance(f_rgb, list):
                r_tokens = [gap(t) for t in f_rgb]
            else:
                r_tokens = [gap(f_rgb)]

        z = layers.Concatenate(name="fusion_concat_tokens")(d_tokens + r_tokens)

    # Classifier
    x = layers.Dense(
        256,
        activation="relu",
        name="cls_fc",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
    )(z)
    x = layers.Dropout(0.5, name="cls_dropout", seed=SEED)(x)

    if ordinal:
        out = add_coral_head(x, num_classes, seed=SEED)
    elif num_classes == 1:
        out = layers.Dense(
            1,
            activation="sigmoid",
            name="output",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        )(x)
    else:
        out = layers.Dense(
            num_classes,
            activation="softmax",
            name="output",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
        )(x)

    if four_stream:
        model = models.Model(
            inputs=[inp_pre, inp_post, inp_rgb_pre, inp_rgb_post],
            outputs=out,
            name="MMF_EMSNet_4stream",
        )
    else:
        model = models.Model(
            inputs=[inp_pre, inp_post, inp_rgb],
            outputs=out,
            name="MMF_EMSNet",
        )
    return model


# ============================================================
# 2. DATA PIPELINES
# ============================================================


def resolve_dsm_channel_indices(
    include_density: bool = True,
    include_unc: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Resolve DSM channel indices for pre/post tensors.

    nDSM is always included. Density and uncertainty channels are optional.
    """
    pre_indices = [PRE_NDSM_IDX]
    post_indices = [POST_NDSM_IDX]

    if include_density:
        pre_indices.append(PRE_DENSITY_IDX)
        post_indices.append(POST_DENSITY_IDX)
    if include_unc:
        pre_indices.append(PRE_UNC_IDX)
        post_indices.append(POST_UNC_IDX)

    return pre_indices, post_indices


def extract_inputs(
    X: np.ndarray,
    include_density: bool = True,
    include_unc: bool = True,
    return_rgb_pre: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Extract DSM pre, DSM post, and RGB post from fused X.
    """
    pre_idx, post_idx = resolve_dsm_channel_indices(
        include_density=include_density,
        include_unc=include_unc,
    )
    if max(post_idx) >= X.shape[-1]:
        raise ValueError(
            f"Input has {X.shape[-1]} channels, but DSM indices require up to {max(post_idx)}."
        )

    dsm_pre = X[..., pre_idx]
    dsm_post = X[..., post_idx]
    rgb_post = X[..., 3:6]
    if return_rgb_pre:
        rgb_pre = X[..., 0:3]
        return dsm_pre, dsm_post, rgb_pre, rgb_post
    return dsm_pre, dsm_post, rgb_post


def make_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch: int,
    shuffle: bool = True,
    include_density: bool = True,
    include_unc: bool = True,
    four_stream: bool = False,
    ordinal: bool = False,
    num_classes: int | None = None,
) -> tf.data.Dataset:
    """
    Build tf.data.Dataset for MMF-EMSNet training/evaluation.

    Args:
        X: Data array (N, H, W, C).
        Y: Labels array (N,), already processed for scenario.
        batch: Batch size.
        shuffle: Whether to shuffle.
        ordinal: If True, encode Y into CORAL's (N, num_classes-1) binary
            target matrix (see encode_coral_labels) instead of leaving it as
            plain integer labels. Requires num_classes.

    Returns:
        Prefetched batched tf.data.Dataset with ((dsm_pre, dsm_post, rgb_post), Y).
    """
    if ordinal:
        if num_classes is None:
            raise ValueError("num_classes is required when ordinal=True")
        Y = encode_coral_labels(Y, num_classes)

    if four_stream:
        dsm_pre, dsm_post, rgb_pre, rgb = extract_inputs(
            X,
            include_density=include_density,
            include_unc=include_unc,
            return_rgb_pre=True,
        )
        ds = tf.data.Dataset.from_tensor_slices(((dsm_pre, dsm_post, rgb_pre, rgb), Y))
    else:
        dsm_pre, dsm_post, rgb = extract_inputs(
            X,
            include_density=include_density,
            include_unc=include_unc,
        )
        ds = tf.data.Dataset.from_tensor_slices(((dsm_pre, dsm_post, rgb), Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    ds = ds.batch(batch)

    # Limit prefetch and private threadpool size to avoid too many threads
    # being created by tf.data (helps avoid pthread_create failures).
    try:
        opts = tf.data.Options()
        opts.experimental_threading.private_threadpool_size = 1
        opts.experimental_threading.max_intra_op_parallelism = 1
        ds = ds.with_options(opts)
    except Exception:
        pass

    ds = ds.prefetch(1)
    return ds
