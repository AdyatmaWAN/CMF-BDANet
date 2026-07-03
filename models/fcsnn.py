"""
FCSNN (Siamese Neural Network) model definition and dataset utilities.

Architecture:
- Shared Siamese tower with 3 conv blocks.
- Optional residual-style concatenation of intermediate features.
- Optional dense layers.
- Final head supports binary (num_classes=1) or multi-class (num_classes>1).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

# Global seed for deterministic initializers inside this module
SEED = 1234
tf.random.set_seed(SEED)


# ============================================================
# 1. FCSNN MODEL DEFINITION
# ============================================================


class FCSNN:
    """
    FCSNN (Siamese Neural Network) for building damage change detection
    using pre- and post-DSM tensors.

    Args:
        num_of_class: Number of classes for classification.
                      - 1 → binary (sigmoid)
                      - >1 → multi-class (softmax)
        residual: If True, concatenates flattened intermediate features.
        dropout: If True, enables dropout after pooling/dense.
        dense: If True, adds dense layers before final head.
        num_of_layer: Number of conv (and dense) layers per block.
        input_shape: Input tensor shape (H, W, C), where C can be
                     nDSM only or nDSM+density and/or nDSM+unc.
        substraction: If True, use absolute difference of embeddings;
                      otherwise, concatenate embeddings.
        shared: If True, share Siamese tower; else use separate towers.
    """

    def __init__(
        self,
        num_of_class: int,
        residual: bool,
        dropout: bool,
        dense: bool,
        num_of_layer: int,
        input_shape: Tuple[int, int, int],
        substraction: bool,
        shared: bool,
        fusion: str = "concat",
    ) -> None:
        self.n_class = num_of_class
        self.is_residual = residual
        self.is_dropout = dropout
        self.is_dense = dense
        self.num_of_layer = num_of_layer
        self.input_shape = input_shape
        self.substraction = substraction
        self.shared = shared
        self.fusion = fusion

    def __build_siamese_model(self) -> Model:
        """
        Build the base Siamese tower.

        Conv channels: [32, 64, 128]
        Each block:
            (Conv → BN → ReLU) × num_of_layer → MaxPool(2x2) → optional Dropout
        Optionally collects intermediate flattened tensors for residual concat.
        """
        inputs = layers.Input(self.input_shape, name="fcsnn_input")
        x = inputs
        feats = []
        convs = [32, 64, 128]

        for conv in convs:
            y = x
            for _ in range(self.num_of_layer):
                y = layers.Conv2D(
                    conv,
                    (3, 3),
                    padding="same",
                    kernel_regularizer=l2(1e-4),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=SEED
                    ),
                )(y)
                y = layers.BatchNormalization()(y)
                y = layers.Activation("relu")(y)

            y = layers.MaxPooling2D(pool_size=(2, 2))(y)
            if self.is_dropout:
                y = layers.Dropout(0.5, seed=SEED)(y)
            if self.is_residual:
                feats.append(layers.Flatten()(y))
            x = y

        x = layers.Flatten()(x)

        if self.is_dense:
            for _ in range(self.num_of_layer):
                x = layers.Dense(
                    256,
                    kernel_regularizer=l2(1e-4),
                    kernel_initializer=tf.keras.initializers.GlorotUniform(
                        seed=SEED
                    ),
                )(x)
                x = layers.Activation("relu")(x)
            if self.is_dropout:
                x = layers.Dropout(0.5, seed=SEED)(x)

        if self.is_residual:
            output = layers.Concatenate(name="fcsnn_concat_feats")(feats + [x])
        else:
            output = x

        return Model(inputs, output, name="fcsnn_tower")

    def get_model(self) -> Model:
        """
        Build the full FCSNN model with two inputs (preDSM, postDSM) and a head.

        Behavior:
            - If n_class == 1 → Dense(1, sigmoid)  (binary)
            - If n_class > 1 → Dense(n_class, softmax) (multi-class)
        """
        img_a = layers.Input(self.input_shape, name="input_pre_dsm")
        img_b = layers.Input(self.input_shape, name="input_post_dsm")

        if self.shared:
            tower = self.__build_siamese_model()
            feat_a = tower(img_a)
            feat_b = tower(img_b)
        else:
            feat_a = self.__build_siamese_model()(img_a)
            feat_b = self.__build_siamese_model()(img_b)

        if self.substraction:
            distance = layers.Subtract(name="fcsnn_subtract")([feat_a, feat_b])
            distance = tf.abs(distance)
        else:
            distance = layers.Concatenate(name="fcsnn_concat")([feat_a, feat_b])

        # Optional MCMAF-style attention fusion over the two embedding vectors
        # (treats feat_a and feat_b as tokens and computes softmax gates).
        if self.fusion == "mcmaf":
            # stack tokens: (B, 2, C)
            tokens = tf.stack([feat_a, feat_b], axis=1)
            score_dense = layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            )
            scores = score_dense(tokens)  # (B, 2, 1)
            gates = tf.nn.softmax(scores, axis=1)
            distance = tf.reduce_sum(tokens * gates, axis=1)

        if self.n_class == 1:
            actv = "sigmoid"
            units = 1
        else:
            actv = "softmax"
            units = self.n_class

        out = layers.Dense(
            units,
            activation=actv,
            kernel_regularizer=l2(1e-4),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
            name="output",
        )(distance)

        model = Model(inputs=[img_a, img_b], outputs=out, name="FCSNN")
        return model


# ============================================================
# 2. DATA LOADING
# ============================================================


def load_dataset(
    npz_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from NPZ file.
    """
    data = np.load(npz_path)
    return (
        data["X_train"],
        data["Y_train"],
        data["X_val"],
        data["Y_val"],
        data["X_test"],
        data["Y_test"],
    )
    # return (
    #     data["train_X"],
    #     data["train_Y"],
    #     data["val_X"],
    #     data["val_Y"],
    #     data["test_X"],
    #     data["test_Y"],
    # )
