#!/usr/bin/env python3
"""Compares tensor_numpy_validate's output against NumPy.

  ./tensor_numpy_validate > tensor_cases.txt && python3 tensor_numpy_validate.py

NumPy is the reference implementation of the flat-buffer-plus-strides layout
Tensor.hpp uses, so agreeing with it on permute / reshape / tensordot is the
strongest available statement that the stride arithmetic is right.
"""
import sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "tensor_cases.txt"
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"


def load(path):
    """Read the two-line-per-case dump into {name: ndarray}."""
    out, lines, i = {}, open(path).read().split("\n"), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        parts = lines[i].split()
        name, rank = parts[0], int(parts[1])
        shape = tuple(int(x) for x in parts[2:2 + rank])
        vals = np.array([float(x) for x in lines[i + 1].split()])
        out[name] = vals.reshape(shape)
        i += 2
    return out


def main():
    d = load(PATH)
    A, B, C = d["A"], d["B"], d["C"]
    expected = {
        "transposeT":   A.T,
        "permute201":   np.transpose(A, (2, 0, 1)),
        "swap02":       np.swapaxes(A, 0, 2),
        "slice_ax1_i2": A[:, 2, :],
        "reshape_6_4":  A.reshape(6, 4),
        "sum_ax0":      A.sum(axis=0),
        "sum_ax1":      A.sum(axis=1),
        "sum_ax2":      A.sum(axis=2),
        "tensordot1":   np.tensordot(A, B, 1),
        "tensordot2":   np.tensordot(A, C, 2),
        "named_2_0":    np.tensordot(A, B, axes=([2], [0])),
        "elem_add":     A + A,
        "elem_mul":     A * A,
        "perm_add":     np.transpose(A, (2, 1, 0)) + np.transpose(A, (2, 1, 0)),
    }

    bad = 0
    for name, want in expected.items():
        got = d[name]
        if got.shape != want.shape:
            print(f"  {RED}FAIL{OFF}  {name}: shape {got.shape}, NumPy says {want.shape}")
            bad += 1
        elif not np.allclose(got, want, rtol=1e-13, atol=1e-15):
            print(f"  {RED}FAIL{OFF}  {name}: max difference {np.abs(got - want).max():.3e}")
            bad += 1
        else:
            print(f"  {GREEN}PASS{OFF}  {name}  {want.shape}")

    total = len(expected)
    print("\n" + "=" * 62)
    if bad:
        print(f"  {RED}{bad} of {total} tensor operations disagree with NumPy{OFF}")
    else:
        print(f"  {total}/{total} tensor operations agree with NumPy")
    print("=" * 62)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
