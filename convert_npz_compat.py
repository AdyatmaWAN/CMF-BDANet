"""
convert_npz_compat.py
=====================
Converts an NPZ file created with NumPy 2.x into one that is fully compatible
with NumPy 1.x (tested against 1.24/1.25/1.26).

WHY THIS IS NEEDED
------------------
NumPy 2.x may use pickle protocol 5 when saving object-dtype arrays (e.g. the
``bid_test`` string array).  NumPy 1.x can only read protocols up to 4, so
``np.load(..., allow_pickle=True)`` raises an error on those arrays.

STRATEGY
--------
For every array in the source NPZ:
  - Numeric arrays  → kept as-is (already compatible).
  - Object arrays   → converted to a fixed-width Unicode dtype (e.g. ``<U32``).
    Fixed-width Unicode is stored inside the .npy file without any pickling, so
    NumPy 1.x can load it with ``allow_pickle=False``.

REQUIREMENTS
------------
Run this script in a Python environment where NumPy >= 2.0 is installed (i.e.
the same environment in which the original NPZ was created), so that the object
arrays can be loaded successfully.

USAGE
-----
    python convert_npz_compat.py --input Dataset/NPZ/dataset_16.npz \\
                                  --output Dataset/NPZ/dataset_16_compat.npz

After conversion, point run_custom_train_infer.py at the new file:
    --dataset Dataset/NPZ/dataset_16_compat.npz
"""

from __future__ import annotations

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npy_bytes(data: bytes) -> np.ndarray:
    """Load a single .npy blob, trying with pickle first, then without."""
    try:
        return np.load(BytesIO(data), allow_pickle=True)
    except Exception:
        return np.load(BytesIO(data), allow_pickle=False)


def convert_object_array(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Convert an object-dtype array to a fixed-width Unicode array so that it
    can be saved and reloaded by NumPy 1.x without pickling.
    """
    flat = arr.flatten()
    try:
        str_values = [str(v) for v in flat]
        max_len = max((len(s) for s in str_values), default=1)
        # Round up to a nice boundary and add a small buffer
        width = max(max_len + 4, 8)
        converted = arr.astype(f"<U{width}")
        print(f"    [{key}] object → <U{width}  (max_str_len={max_len})")
        return converted
    except Exception as exc:
        print(f"    [{key}] WARNING: could not convert object array: {exc}")
        print(f"    [{key}] Falling back to bytes representation.")
        byte_values = [str(v).encode() for v in flat]
        max_blen = max((len(b) for b in byte_values), default=1)
        return np.array([str(v).encode() for v in arr.flatten()]).reshape(arr.shape).astype(f"|S{max_blen}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path) -> None:
    print(f"\nSource : {input_path}  ({input_path.stat().st_size / 1e6:.1f} MB)")

    converted: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(input_path, "r") as zf:
        names = zf.namelist()
        print(f"Keys   : {[n.removesuffix('.npy') for n in names]}\n")

        for name in names:
            key = name.removesuffix(".npy")
            raw = zf.read(name)
            arr = load_npy_bytes(raw)

            if arr.dtype == object:
                print(f"  Converting '{key}'  shape={arr.shape}  dtype=object")
                arr = convert_object_array(arr, key)
            else:
                print(f"  Keeping   '{key}'  shape={arr.shape}  dtype={arr.dtype}")

            converted[key] = arr

    print(f"\nSaving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **converted)
    print(f"Saved  ({output_path.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Verify the output can be loaded without allow_pickle
    # ------------------------------------------------------------------
    print("\nVerifying (allow_pickle=False) ...")
    try:
        check = np.load(output_path, allow_pickle=False)
        for k in check.files:
            print(f"  OK  {k:20s}  dtype={check[k].dtype}  shape={check[k].shape}")
        print("\nConversion successful — the output NPZ is NumPy 1.x compatible.")
    except Exception as exc:
        print(f"\nERROR during verification: {exc}")
        print("Some arrays may still require allow_pickle=True.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a NumPy 2.x NPZ file to NumPy 1.x compatible format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Dataset/NPZ/dataset_16.npz",
        help="Path to the source NPZ file (created with NumPy 2.x).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Dataset/NPZ/dataset_16_compat.npz",
        help="Path for the converted output NPZ file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(Path(args.input), Path(args.output))
