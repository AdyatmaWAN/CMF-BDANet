"""
CORAL ordinal-regression head shared by FCSNN, MMF-EMSNet, and FUJITA.

Source: Cao, W., Mirjalili, V., & Raschka, S. (2020). "Rank Consistent
Ordinal Regression for Neural Networks with Application to Age Estimation."
Pattern Recognition Letters, 140, 325-331. https://arxiv.org/abs/1901.07884

See the README's "Ordinal regression (scenario 5)" section for the full
explanation of the mechanics and loss function. In short: a K-class ordinal
problem is decomposed into K-1 binary "is y > threshold k?" questions, all
sharing one weight vector and differing only by a bias term constrained to
be non-increasing (via CoralBiases) so the K-1 answers can never
contradict each other. Any model can add this head to its penultimate
feature tensor via `add_coral_head`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from utils.metrics import compute_ordinal_metrics

SEED = 1234


class CoralBiases(layers.Layer):
    """
    Turns a single shared logit `w^T g(x)` (computed upstream by a
    `Dense(1, use_bias=False)` layer, so every threshold uses the *same*
    weight vector) into K-1 sigmoid outputs P(y>0), ..., P(y>K-2) by adding
    a different bias per threshold. Rank consistency — the guarantee that
    P(y>0) >= P(y>1) >= ... >= P(y>K-2), so the thresholds never contradict
    each other — comes from constraining those biases to be non-increasing.
    This is enforced structurally: bias_k = bias_0 - cumsum(softplus(gaps)),
    where softplus keeps each gap >= 0, so the cumulative sum can only grow,
    which makes bias_k shrink as k increases. Gradient descent can move
    bias_0 and the gaps freely; it can never produce an increasing sequence.
    """

    def __init__(self, num_thresholds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_thresholds = num_thresholds

    def build(self, input_shape):
        self.bias0 = self.add_weight(name="bias0", shape=(), initializer="zeros", trainable=True)
        self.gaps = self.add_weight(
            name="gaps", shape=(self.num_thresholds - 1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, shared_logit: tf.Tensor) -> tf.Tensor:
        cumulative_gaps = tf.cumsum(tf.nn.softplus(self.gaps))
        leading_zero = tf.zeros((1,), dtype=cumulative_gaps.dtype)
        step_drops = tf.concat([leading_zero, cumulative_gaps], axis=0)  # (T,), non-decreasing
        biases = self.bias0 - step_drops  # (T,), non-increasing by construction
        logits = shared_logit + biases[tf.newaxis, :]  # (B,1) + (1,T) -> (B,T)
        return tf.nn.sigmoid(logits)

    def get_config(self):
        config = super().get_config()
        config.update({"num_thresholds": self.num_thresholds})
        return config


def add_coral_head(x: tf.Tensor, num_classes: int, seed: int = SEED, name_prefix: str = "coral") -> tf.Tensor:
    """Attach a CORAL ordinal head to a penultimate feature tensor `x`.
    Requires num_classes >= 3 (at least 2 thresholds).
    """
    if num_classes < 3:
        raise ValueError("ordinal=True requires num_classes >= 3 (at least 2 thresholds).")
    shared_logit = layers.Dense(
        1,
        use_bias=False,
        name=f"{name_prefix}_shared_weight",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    )(x)
    return CoralBiases(num_classes - 1, name=f"{name_prefix}_biases")(shared_logit)


def encode_coral_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Encode integer class labels for CORAL training. Each label y becomes a
    length (num_classes-1) binary vector where entry k is 1 if y > k else 0
    — i.e. "is the true rank past this threshold?" for every threshold.
    Example (num_classes=5): label 2 -> [1, 1, 0, 0].
    """
    y = np.asarray(y).astype(np.int64)
    thresholds = np.arange(num_classes - 1)
    return (y[:, None] > thresholds[None, :]).astype(np.float32)


def decode_coral_predictions(probs: np.ndarray) -> np.ndarray:
    """
    Decode CORAL sigmoid outputs (N, num_classes-1) back to integer class
    labels by counting how many thresholds are exceeded (probability > 0.5).
    A prediction of rank r means "further than r thresholds were exceeded".
    """
    return np.sum(np.asarray(probs) > 0.5, axis=-1).astype(np.int64)


def decode_coral_true_labels(y_encoded: np.ndarray) -> np.ndarray:
    """True CORAL-encoded labels are exact (no thresholding uncertainty):
    exactly `rank` of the `num_classes - 1` entries are 1, so summing
    recovers the original integer rank directly.
    """
    return np.sum(np.asarray(y_encoded), axis=-1).astype(np.int64)


def coral_probs_to_class_probs(probs: np.ndarray) -> np.ndarray:
    """
    Convert CORAL's threshold probabilities P(y>0)..P(y>K-2) into a proper
    per-class probability distribution P(y=0)..P(y=K-1), via the identity
    P(y=k) = P(y>k-1) - P(y>k), using the conventions P(y>-1)=1 and
    P(y>K-1)=0. This lets predictions.csv keep the same prob_class_0..K-1
    schema for ordinal runs as every other (nominal) run.
    """
    probs = np.asarray(probs)
    n = probs.shape[0]
    extended = np.concatenate(
        [np.ones((n, 1), dtype=probs.dtype), probs, np.zeros((n, 1), dtype=probs.dtype)],
        axis=1,
    )  # (N, K+1): [P(y>-1)=1, P(y>0), ..., P(y>K-2), P(y>K-1)=0]
    class_probs = -np.diff(extended, axis=1)  # (N, K)

    # Guard against tiny negative values from floating-point rounding.
    class_probs = np.clip(class_probs, 0.0, None)
    row_sums = class_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return class_probs / row_sums


def ordinal_extra_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """extra_metrics_fn for train_and_evaluate_nn: adds MAE + QWK to metrics.csv."""
    mae, qwk = compute_ordinal_metrics(y_true, y_pred)
    return {"mae": mae, "qwk": qwk}
