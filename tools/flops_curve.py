"""
Complexity curves: FLOPs vs input size (HW) and FLOPs vs network width.

Builds each model at synthetic input sizes / widths and counts forward-pass
FLOPs with the TF1 profiler. No data, no weights, no training involved.

Usage:
    python tools/flops_curve.py                  # CSV + PDF/PNG into results/
    python tools/flops_curve.py --sizes 8 16 32  # custom HW sweep
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
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

tf.get_logger().setLevel("ERROR")

from models.bddnet import build_bddnet_model
from models.fcsnn import FCSNN
from models.fujita import snn as FujitaSNN
from models.cmf_bdanet import build_cmf_bdanet_conv

# Channel counts matching the default dsm_mode="dsm_density_uncertainty" runs:
# 3 DSM-side channels (nDSM + density + uncertainty), 3 RGB channels.
DSM_CH, RGB_CH = 3, 3
NUM_CLASSES = 4
# 3 max-pools in most backbones -> sizes must stay divisible by 8.
DEFAULT_SIZES = list(range(8, 129, 8))
DEFAULT_WIDTHS = [8, 16, 32, 64, 128]
# Real dataset patch size (dataset_16.npz); used as the fixed HW for the
# width sweep.
NATIVE_SIZE = 16


def count_flops(model) -> int:
    """Forward-pass FLOPs for batch size 1 (2 flops per MAC)."""
    specs = [tf.TensorSpec((1, *i.shape[1:]), tf.float32) for i in model.inputs]
    fn = tf.function(lambda x: model(x)).get_concrete_function(specs)
    frozen = convert_variables_to_constants_v2(fn)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    prof = tf.compat.v1.profiler.profile(
        graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(), cmd="scope", options=opts
    )
    return prof.total_float_ops


def build(name: str, size: int, width: int):
    """Build one model at input size (size, size) and the given base width."""
    dsm, rgb = (size, size, DSM_CH), (size, size, RGB_CH)
    if name == "CMF-BDANet":
        return build_cmf_bdanet_conv(dsm, rgb, NUM_CLASSES, base_filters=width)
    if name == "BDD-Net":
        return build_bddnet_model(rgb, dsm, NUM_CLASSES, base_filters=width // 2)
    if name == "FC-SNN":
        return FCSNN(
            num_of_class=NUM_CLASSES, residual=True, dropout=True, dense=True,
            num_of_layer=3, input_shape=dsm, substraction=True, shared=True,
            base_filters=width,
        ).get_model()
    if name == "Fujita":
        return FujitaSNN(NUM_CLASSES, dsm, base_filters=width).get_model()
    raise ValueError(name)


MODELS = ["CMF-BDANet", "BDD-Net", "FC-SNN", "Fujita"]


def sweep(sizes, widths):
    rows = []
    for name in MODELS:
        for s in sizes:
            m = build(name, s, 32)
            rows.append(dict(panel="hw", model=name, x=s * s, size=s, width=32,
                             flops=count_flops(m), params=m.count_params()))
            tf.keras.backend.clear_session()
            print(f"  hw    {name:12s} S={s:4d} -> {rows[-1]['flops']/1e9:.4f} GFLOPs")
        for w in widths:
            m = build(name, NATIVE_SIZE, w)
            rows.append(dict(panel="width", model=name, x=w, size=NATIVE_SIZE, width=w,
                             flops=count_flops(m), params=m.count_params()))
            tf.keras.backend.clear_session()
            print(f"  width {name:12s} C={w:4d} -> {rows[-1]['flops']/1e9:.4f} GFLOPs")
    return rows


def plot(rows, out_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("hw", "Input size ($HW$, pixels)", "hw"),
        ("width", "Base filter count ($C$)", "width"),
    ]
    for panel, xlabel, suffix in panels:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for name in MODELS:
            pts = sorted([r for r in rows if r["panel"] == panel and r["model"] == name],
                         key=lambda r: r["x"])
            ax.plot([p["x"] for p in pts], [p["flops"] / 1e9 for p in pts],
                    marker="o", ms=3, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Floating point operations (GFLOPs)")
        ax.margins(x=0.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{out_stem}_{suffix}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    ap.add_argument("--out", default="results/flops_curve")
    ap.add_argument("--replot", action="store_true",
                    help="re-render figures from an existing CSV, skipping the sweep")
    args = ap.parse_args()

    bad = [s for s in args.sizes if s % 8]
    if bad:
        ap.error(f"sizes must be divisible by 8 (3 max-pool stages): {bad}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.replot:
        with open(f"{args.out}.csv", newline="", encoding="utf-8") as f:
            rows = [{**r, "x": float(r["x"]), "flops": int(r["flops"])}
                    for r in csv.DictReader(f)]
        plot(rows, args.out)
        print(f"wrote {args.out}_hw / {args.out}_width .pdf/.png")
        return

    rows = sweep(args.sizes, args.widths)

    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.out)
    print(f"\nwrote {args.out}.csv and {args.out}_hw / {args.out}_width .pdf/.png")


def _selfcheck():
    """FLOP count of a hand-computable graph, and monotonicity of the sweep."""
    m = tf.keras.Sequential([tf.keras.layers.Input((16, 16, 3)),
                             tf.keras.layers.Conv2D(8, 3, padding="same", use_bias=False)])
    assert count_flops(m) == 2 * 16 * 16 * 8 * 27, count_flops(m)
    rows = sweep([8, 16], [8, 16])
    for name in MODELS:
        for panel in ("hw", "width"):
            f = [r["flops"] for r in rows if r["model"] == name and r["panel"] == panel]
            assert f[1] > f[0], (name, panel, f)
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
