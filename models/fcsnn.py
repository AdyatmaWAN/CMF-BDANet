"""
FCSNN (Siamese Neural Network) model definition and dataset utilities.

Architecture (see docs/EXPERIMENT_PLAN.md and the architecture diagrams):

    Convolution Block(N):
        (Conv2D(N) -> BatchNorm -> ReLU) x num_of_layer
        -> MaxPooling2D(2, 2)
        -> Dropout(0.5)          [only when dropout=True]

    Tower (per arm):
        Conv Block(bf) -> Conv Block(2*bf) -> Conv Block(4*bf)
        Flatten(last block) -> Dense Block(256)        [only when dense=True]
        Concat([Flatten(block1), Flatten(block2), Flatten(block3), dense])
                                                       [only when residual=True]

    Head:
        fuse(tower(pre), tower(post)) -> Dense(n, sigmoid/softmax) or CORAL

`fusion` is a single parameter with one conditional sequence:

    "sub"    -> abs(a - b)                  (the historical FCSNN behaviour)
    "concat" -> Concatenate([a, b])         (pseudo-siamese concatenation)
    "mcmaf"  -> softmax-gated weighted sum of a and b treated as two tokens

`shared=True` reuses one tower for both inputs (true Siamese); `shared=False`
builds two independent towers (pseudo-Siamese), each with its own seed so they
do not start from identical weights.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

from models.ordinal import add_coral_head

# Default seed, used only when a caller does not pass one explicitly. Every
# random op in this module reads the instance's `seed`, never a module global,
# so N different seeds really do produce N different initialisations.
SEED = 1234

FUSIONS = ("sub", "concat", "mcmaf")


# ============================================================
# 1. FCSNN MODEL DEFINITION
# ============================================================


class FCSNN:
    """
    FCSNN (Siamese Neural Network) for building damage change detection
    using pre- and post-DSM tensors.

    Args:
        num_of_class: Number of classes.
                      - 1 -> binary (sigmoid)
                      - >1 -> multi-class (softmax), or ordinal (CORAL) when
                        ordinal=True
        residual: If True, concatenates the flattened output of every conv
                  block with the dense output inside each tower.
        dropout: If True, adds Dropout(0.5) after each block's MaxPooling.
        dense: If True, adds one Dense(256)+ReLU block after the final
               flatten, before the residual concatenation.
        num_of_layer: Number of Conv->BN->ReLU repeats per conv block.
        input_shape: Input tensor shape (H, W, C).
        fusion: How the two tower embeddings are combined — one of
                "sub", "concat", "mcmaf" (see module docstring).
        shared: If True, share one Siamese tower; else two independent towers.
        ordinal: If True, use a CORAL ordinal head instead of sigmoid/softmax.
                 Requires num_of_class >= 3.
        base_filters: Conv width; blocks are (bf, 2*bf, 4*bf).
        seed: Seed for every initializer and dropout mask in this model.
        substraction: Deprecated. `substraction=True` is an alias for
                      `fusion="sub"`. Kept so older callers keep working.
    """

    def __init__(
        self,
        num_of_class: int,
        residual: bool,
        dropout: bool,
        dense: bool,
        num_of_layer: int,
        input_shape: Tuple[int, int, int],
        shared: bool,
        fusion: str = "sub",
        ordinal: bool = False,
        base_filters: int = 32,
        seed: int = SEED,
        substraction: bool | None = None,
    ) -> None:
        if substraction is not None:
            warnings.warn(
                "FCSNN(substraction=...) is deprecated; pass fusion='sub' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if substraction:
                fusion = "sub"

        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")

        self.n_class = num_of_class
        self.is_residual = residual
        self.is_dropout = dropout
        self.is_dense = dense
        self.num_of_layer = num_of_layer
        self.input_shape = input_shape
        self.shared = shared
        self.fusion = fusion
        self.ordinal = ordinal
        self.base_filters = base_filters
        self.seed = seed

    def _init(self, seed: int):
        return tf.keras.initializers.GlorotUniform(seed=seed)

    def __build_siamese_model(self, seed: int, name: str) -> Model:
        """Build one tower. `seed` drives every initializer and dropout mask in
        it, so two unshared towers built with different seeds do not start from
        identical weights.
        """
        inputs = layers.Input(self.input_shape, name=f"{name}_input")
        x = inputs
        feats = []
        bf = self.base_filters

        for conv in (bf, bf * 2, bf * 4):
            y = x
            for _ in range(self.num_of_layer):
                y = layers.Conv2D(
                    conv,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(1e-4),
                    kernel_initializer=self._init(seed),
                )(y)
                y = layers.BatchNormalization()(y)
                y = layers.Activation("relu")(y)

            y = layers.MaxPooling2D(pool_size=(2, 2))(y)
            if self.is_dropout:
                y = layers.Dropout(0.5, seed=seed)(y)
            if self.is_residual:
                feats.append(layers.Flatten()(y))
            x = y

        x = layers.Flatten()(x)

        # One Dense Block(256) after the flatten, before the concatenation —
        # NOT num_of_layer copies of it (that reused the conv-depth knob for
        # dense depth, which the architecture diagram does not call for).
        if self.is_dense:
            x = layers.Dense(
                256,
                kernel_regularizer=l2(1e-4),
                kernel_initializer=self._init(seed),
            )(x)
            x = layers.Activation("relu")(x)

        if self.is_residual:
            output = layers.Concatenate(name=f"{name}_concat_feats")(feats + [x])
        else:
            output = x

        return Model(inputs, output, name=name)

    def _fuse(self, feat_a: tf.Tensor, feat_b: tf.Tensor) -> tf.Tensor:
        """Single conditional sequence over the three fusion modes."""
        if self.fusion == "sub":
            d = layers.Subtract(name="fcsnn_subtract")([feat_a, feat_b])
            # Lambda wrapper: raw tf ops on symbolic tensors are rejected by Keras 3.
            return layers.Lambda(lambda t: tf.abs(t), name="fcsnn_abs")(d)

        if self.fusion == "concat":
            return layers.Concatenate(name="fcsnn_concat")([feat_a, feat_b])

        # mcmaf: treat the two embeddings as tokens and softmax-gate them.
        tokens = layers.Lambda(
            lambda t: tf.stack(t, axis=1), name="fcsnn_stack"
        )([feat_a, feat_b])  # (B, 2, C)
        scores = layers.Dense(1, kernel_initializer=self._init(self.seed))(tokens)
        gates = layers.Softmax(axis=1, name="fcsnn_gates")(scores)
        return layers.Lambda(
            lambda t: tf.reduce_sum(t[0] * t[1], axis=1), name="fcsnn_gate_sum"
        )([tokens, gates])

    def get_model(self) -> Model:
        """Build the full FCSNN model with two inputs (preDSM, postDSM)."""
        img_a = layers.Input(self.input_shape, name="input_pre_dsm")
        img_b = layers.Input(self.input_shape, name="input_post_dsm")

        if self.shared:
            tower = self.__build_siamese_model(self.seed, "fcsnn_tower")
            feat_a = tower(img_a)
            feat_b = tower(img_b)
        else:
            # Different seeds per tower: GlorotUniform(seed=s) is deterministic,
            # so two towers built with the same seed would start out identical,
            # which is not what "unshared weights" is supposed to mean.
            feat_a = self.__build_siamese_model(self.seed, "fcsnn_tower_a")(img_a)
            feat_b = self.__build_siamese_model(self.seed + 1, "fcsnn_tower_b")(img_b)

        distance = self._fuse(feat_a, feat_b)

        if self.ordinal:
            out = add_coral_head(distance, self.n_class, seed=self.seed)
        else:
            units, actv = (1, "sigmoid") if self.n_class == 1 else (self.n_class, "softmax")
            out = layers.Dense(
                units,
                activation=actv,
                kernel_regularizer=l2(1e-4),
                kernel_initializer=self._init(self.seed),
                name="output",
            )(distance)

        return Model(inputs=[img_a, img_b], outputs=out, name="FCSNN")


# ============================================================
# 2. DATA LOADING
# ============================================================


def load_dataset(
    npz_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from NPZ file."""
    data = np.load(npz_path)
    return (
        data["X_train"],
        data["Y_train"],
        data["X_val"],
        data["Y_val"],
        data["X_test"],
        data["Y_test"],
    )
