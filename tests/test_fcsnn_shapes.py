"""Shape/capacity self-check for models/fcsnn.py.

Run directly:  python tests/test_fcsnn_shapes.py

Checks the things the experiment protocol depends on:
  - all three fusion modes build, and produce the expected fused width
  - unshared has ~2x the parameters of shared at the same base_filters
  - the capacity-control width makes unshared ~= shared in parameters
  - sub and concat differ only in the head (identical tower parameters)
  - dropout/dense flags actually change the graph
  - different seeds produce different weights; the same seed reproduces
  - unshared towers do not start out identical to each other

It also prints the parameter count per arm — that is the `params` column in
the paper tables, and what re-pins CONTROL_BF if the arithmetic was wrong.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from models.fcsnn import FCSNN  # noqa: E402

INPUT_SHAPE = (16, 16, 1)
REPORT_BF = 32
CONTROL_BF = 22


def build(**kwargs):
    defaults = dict(
        num_of_class=1,
        residual=True,
        dropout=True,
        dense=True,
        num_of_layer=3,
        input_shape=INPUT_SHAPE,
        shared=True,
        fusion="sub",
        base_filters=REPORT_BF,
        seed=12345,
    )
    defaults.update(kwargs)
    return FCSNN(**defaults).get_model()


def n_params(model) -> int:
    return int(model.count_params())


def test_all_fusions_build():
    for fusion in ("sub", "concat", "mcmaf"):
        m = build(fusion=fusion)
        assert m.output_shape == (None, 1), (fusion, m.output_shape)
    print("OK  all three fusion modes build")


def test_fusion_widths():
    """sub and mcmaf collapse to one embedding; concat doubles it."""
    width = {}
    for fusion in ("sub", "concat", "mcmaf"):
        m = build(fusion=fusion)
        width[fusion] = m.get_layer("output").input.shape[-1]
    assert width["sub"] == width["mcmaf"], width
    assert width["concat"] == 2 * width["sub"], width
    print(f"OK  fused widths sub={width['sub']} mcmaf={width['mcmaf']} concat={width['concat']}")


def test_deprecated_substraction_alias():
    m = build(substraction=True, fusion="concat")
    names = [l.name for l in m.layers]
    assert "fcsnn_subtract" in names, names
    assert "fcsnn_concat" not in names, names
    print("OK  substraction=True overrides fusion to 'sub'")


def test_invalid_fusion_rejected():
    try:
        build(fusion="nonsense")
    except ValueError:
        print("OK  invalid fusion rejected")
        return
    raise AssertionError("invalid fusion was accepted")


def test_unshared_is_about_double():
    for fusion in ("sub", "concat"):
        shared = n_params(build(shared=True, fusion=fusion))
        unshared = n_params(build(shared=False, fusion=fusion))
        ratio = unshared / shared
        assert 1.9 < ratio < 2.1, (fusion, shared, unshared, ratio)
    print("OK  unshared ~= 2x shared parameters")


def test_capacity_control_matches():
    """The whole point of the bf=CONTROL_BF arm: same budget as shared bf=32."""
    for fusion in ("sub", "concat"):
        shared = n_params(build(shared=True, fusion=fusion, base_filters=REPORT_BF))
        control = n_params(build(shared=False, fusion=fusion, base_filters=CONTROL_BF))
        ratio = control / shared
        assert 0.9 < ratio < 1.1, (
            f"capacity control bf={CONTROL_BF} is {ratio:.3f}x shared bf={REPORT_BF} "
            f"({control} vs {shared}) - re-pin CONTROL_BF"
        )
    print(f"OK  unshared bf={CONTROL_BF} within 10% of shared bf={REPORT_BF}")


def test_sub_and_concat_share_tower_size():
    """sub vs concat must differ only in the head, so the comparison is not
    confounded by capacity."""
    for shared in (True, False):
        sub = n_params(build(fusion="sub", shared=shared))
        cat = n_params(build(fusion="concat", shared=shared))
        head = sub  # head is Dense(units) on the fused vector; towers identical
        assert cat > sub, (sub, cat)
        # the only extra parameters are the widened output Dense
        fused = build(fusion="sub", shared=shared).get_layer("output").input.shape[-1]
        assert cat - sub == fused, (cat - sub, fused)
        del head
    print("OK  sub vs concat differ only by the output layer width")


def test_flags_change_graph():
    base = build(dropout=True, dense=True)
    no_drop = build(dropout=False, dense=True)
    no_dense = build(dropout=True, dense=False)

    assert len(no_drop.layers) < len(base.layers) or n_params(no_drop) == n_params(base)
    drop_layers = sum(
        1 for layer in base.get_layer("fcsnn_tower").layers if "dropout" in layer.name
    )
    assert drop_layers == 3, drop_layers
    drop_layers_off = sum(
        1 for layer in no_drop.get_layer("fcsnn_tower").layers if "dropout" in layer.name
    )
    assert drop_layers_off == 0, drop_layers_off
    assert n_params(no_dense) != n_params(base)
    print("OK  dropout/dense flags change the graph")


def test_seed_controls_weights():
    a1 = build(seed=12345).get_weights()
    a2 = build(seed=12345).get_weights()
    b = build(seed=23451).get_weights()

    assert all(np.array_equal(x, y) for x, y in zip(a1, a2)), "same seed did not reproduce"
    assert any(not np.array_equal(x, y) for x, y in zip(a1, b)), "different seeds gave same weights"
    print("OK  seed controls initialisation (reproducible, and distinct across seeds)")


def test_unshared_towers_differ():
    m = build(shared=False, seed=12345)
    wa = m.get_layer("fcsnn_tower_a").get_weights()
    wb = m.get_layer("fcsnn_tower_b").get_weights()
    assert any(not np.array_equal(x, y) for x, y in zip(wa, wb)), (
        "the two unshared towers started from identical weights"
    )
    print("OK  unshared towers start from different weights")


def report_params():
    print("\nparameter counts (input 16x16x1, num_of_layer=3, residual+dense on):")
    rows = []
    for fusion in ("sub", "concat"):
        for shared in (True, False):
            for bf in (REPORT_BF, CONTROL_BF):
                if shared and bf != REPORT_BF:
                    continue
                p = n_params(build(fusion=fusion, shared=shared, base_filters=bf))
                rows.append((fusion, "shared" if shared else "unshared", bf, p))
    ref = {(f, "shared", REPORT_BF): p for f, s, b, p in rows if s == "shared" for _ in [0]}
    for fusion, sharing, bf, p in rows:
        base = ref.get((fusion, "shared", REPORT_BF), p)
        print(f"  {fusion:6s} {sharing:8s} bf={bf:<3d} params={p:>9,d}  ({p / base:.3f}x shared)")


def main() -> None:
    test_all_fusions_build()
    test_fusion_widths()
    test_deprecated_substraction_alias()
    test_invalid_fusion_rejected()
    test_unshared_is_about_double()
    test_capacity_control_matches()
    test_sub_and_concat_share_tower_size()
    test_flags_change_graph()
    test_seed_controls_weights()
    test_unshared_towers_differ()
    report_params()
    print("\nall FCSNN shape checks passed")


if __name__ == "__main__":
    main()
