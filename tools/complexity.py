"""
Memory and computation complexity of every model, at the native config.

Per model: parameters (weight memory), activation footprint (inference peak
and training-time total, since backprop keeps every activation), and
forward-pass FLOPs. The SVM baselines have no activations -- their cost is
reported analytically from the feature dimension and support-vector count.

Usage:
    python tools/complexity.py                # markdown + CSV into results/
    python tools/complexity.py --size 16 --width 32
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tools.flops_curve import build, count_flops, MODELS, DSM_CH, NATIVE_SIZE

BYTES = 4  # float32
MB = 1024 ** 2


def activations(model):
    """(peak inference, training total) activation elements, batch size 1.

    Inference peak is approximated by the largest single layer output; training
    total is the sum over all layers, since backprop retains every one.
    """
    sizes = []
    for layer in model.layers:
        outs = layer.output if isinstance(layer.output, list) else [layer.output]
        for o in outs:
            n = 1
            for d in o.shape[1:]:
                n *= int(d)
            sizes.append(n)
        if isinstance(layer, tf.keras.Model):  # sub-encoder, called once per stream
            p, t = activations(layer)
            sizes.append(t)
    return max(sizes), sum(sizes)


def nn_rows(size, width, batch):
    # build() halves the width for BDD-Net (its own default is 16), so it is
    # also reported at the matched width, where its widest stage equals the
    # other backbones' 4*C.
    todo = [(n, width) for n in MODELS] + [("BDD-Net", width * 2)]
    rows = []
    for name, width in todo:
        m = build(name, size, width)
        peak, total = activations(m)
        params = m.count_params()
        rows.append(dict(
            model=f"{name} (bf={width // 2 if name == 'BDD-Net' else width})",
            params=params,
            weight_mb=params * BYTES / MB,
            # Adam keeps 2 moments + a gradient copy alongside the weights.
            optimizer_mb=params * 4 * BYTES / MB,
            act_peak_mb=peak * BYTES / MB,
            act_train_mb=total * batch * BYTES / MB,
            train_mb=(params * 4 + total * batch) * BYTES / MB,
            flops=count_flops(m),
            gflops=count_flops(m) / 1e9,
        ))
        tf.keras.backend.clear_session()
    return rows


def svm_rows(size):
    """Analytic cost: an RBF SVM stores n_sv*d floats and costs O(n_sv*d) per
    prediction. Feature dim d is fixed by the extractor, not by the classifier.
    """
    d_moya = 3                        # mean, std, correlation of the DSM diff
    d_hajeb = size * size * DSM_CH    # flattened per-pixel DSM difference
    rows = []
    for name, d in (("Moya (SVM)", d_moya), ("Hajeb (SVM)", d_hajeb)):
        rows.append(dict(model=name, params=d, weight_mb=float("nan"),
                         optimizer_mb=0.0, act_peak_mb=0.0, act_train_mb=0.0,
                         train_mb=float("nan"), flops=0, gflops=float("nan")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=NATIVE_SIZE)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1, help="batch size for training memory")
    ap.add_argument("--out", default="results/complexity")
    args = ap.parse_args()

    rows = nn_rows(args.size, args.width, args.batch) + svm_rows(args.size)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = f"{"model":20s} {'params':>10s} {'weights MB':>11s} {'act peak MB':>12s} " \
          f"{'train MB':>10s} {'MFLOPs':>10s}"
    print(f"\ninput {args.size}x{args.size}, base width {args.width}, batch {args.batch}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r["model"]:20s} {r['params']:10,d} {r['weight_mb']:11.3f} "
              f"{r['act_peak_mb']:12.3f} {r['train_mb']:10.2f} {r['flops']/1e6:10.2f}")
    print(f"\nwrote {args.out}.csv")


def _selfcheck():
    m = tf.keras.Sequential([tf.keras.layers.Input((16, 16, 3)),
                             tf.keras.layers.Conv2D(8, 3, padding="same")])
    peak, total = activations(m)
    assert peak == 16 * 16 * 8 and total == 16 * 16 * 8, (peak, total)
    rows = nn_rows(16, 8, 1)
    assert all(r["params"] > 0 and r["act_peak_mb"] > 0 for r in rows), rows
    assert svm_rows(16)[1]["params"] == 16 * 16 * DSM_CH
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
