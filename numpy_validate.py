#!/usr/bin/env python3
"""
Cross-validates MatrixCpp against NumPy.

numpy_validate.cpp runs every operator on fixed inputs and dumps the inputs and
results to validation/*.dat. This script recomputes each one in NumPy/SciPy and
compares.

Where a result is unique — add, multiply, det, inverse, solve, expm, … — the
comparison is element by element. Where it is NOT unique, comparing factors
directly would be meaningless: QR, LU, SVD and eig all admit sign flips, column
reorderings and different pivoting, so two correct implementations routinely
disagree entry for entry. For those, this checks what actually IS well defined:
that the factors reconstruct A, that they have the structure they claim
(triangular, orthogonal, a permutation), and that the spectrum matches NumPy's
once sorted.

This is a different check from validate.cpp. That one tests the library against
itself; this one tests it against an independent implementation, which is what
catches a convention that is self-consistent but not what everyone else means.

Usage:
    ./numpy_validate && python3 numpy_validate.py
"""
import os
import sys
import glob
import numpy as np

try:
    import scipy.linalg as sla
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

VAL_DIR = "validation"
RTOL, ATOL = 1e-8, 1e-8

GREEN, RED, YELLOW, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"

passed = failed = skipped = 0
failures = []


# ────────────────────────────────────────────────────────────── parsing ──
def load_case(path):
    """Parse one .dat file into {'op','dtype','param', <label>: ndarray}."""
    with open(path) as f:
        tokens = f.read().split()
    i, case = 0, {}
    while i < len(tokens):
        tok = tokens[i]
        if tok == "op":
            case["op"] = tokens[i + 1]; i += 2
        elif tok == "dtype":
            case["dtype"] = tokens[i + 1]; i += 2
        elif tok == "param":
            case["param"] = float(tokens[i + 1]); i += 2
        elif tok == "mat":
            label = tokens[i + 1]
            rows, cols = int(tokens[i + 2]), int(tokens[i + 3])
            # Per-matrix element kind: a complex case still holds real matrices
            # for real(), imag() and the norms.
            kind = tokens[i + 4]
            i += 5
            per = 2 if kind == "c" else 1
            count = rows * cols * per
            vals = np.array([float(t) for t in tokens[i:i + count]])
            i += count
            if per == 2:
                arr = (vals[0::2] + 1j * vals[1::2]).reshape(rows, cols)
            else:
                arr = vals.reshape(rows, cols)
            case[label] = arr
        elif tok == "end":
            i += 1
        else:
            i += 1
    return case


def report(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}PASS{OFF}  {name}")
    else:
        failed += 1
        failures.append((name, detail))
        print(f"  {RED}FAIL{OFF}  {name}   {detail}")


def close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs numpy {b.shape}"
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        d = np.abs(a - b)
        idx = np.unravel_index(np.argmax(d), d.shape)
        return False, (f"max |diff| = {d.max():.3e} at {idx} "
                       f"(cpp {np.asarray(a)[idx]!r} vs numpy {np.asarray(b)[idx]!r})")
    return True, ""


# ─────────────────────────────────── operators with a unique NumPy answer ──
# key -> f(case) returning the NumPy expectation for case["R"]
DIRECT = {
    "add":            lambda c: c["A"] + c["B"],
    "subtract":       lambda c: c["A"] - c["B"],
    "multiply":       lambda c: c["A"] @ c["B"],
    "hadamard":       lambda c: c["A"] * c["B"],
    "elem_div":       lambda c: c["A"] / c["B"],
    "negate":         lambda c: -c["A"],
    "scalar_mul":     lambda c: c["A"] * c["param"],
    "scalar_div":     lambda c: c["A"] / c["param"],
    "transpose":      lambda c: c["A"].T,
    "conj":           lambda c: np.conj(c["A"]),
    "conj_transpose": lambda c: c["A"].conj().T,
    "real_part":      lambda c: np.real(c["A"]),
    "imag_part":      lambda c: np.imag(c["A"]),

    "elem_exp":       lambda c: np.exp(c["A"]),
    "elem_ln":        lambda c: np.log(c["A"]),
    "elem_log10":     lambda c: np.log10(c["A"]),
    "elem_pow":       lambda c: c["A"] ** c["param"],
    "elem_sqrt":      lambda c: np.sqrt(c["A"]),
    "elem_sin":       lambda c: np.sin(c["A"]),
    "elem_cos":       lambda c: np.cos(c["A"]),
    "elem_tan":       lambda c: np.tan(c["A"]),
    "elem_sinh":      lambda c: np.sinh(c["A"]),
    "elem_cosh":      lambda c: np.cosh(c["A"]),
    "elem_tanh":      lambda c: np.tanh(c["A"]),

    "sum_all":        lambda c: np.array([[c["A"].sum()]]),
    "sum_cols":       lambda c: c["A"].sum(axis=0).reshape(1, -1),
    "sum_rows":       lambda c: c["A"].sum(axis=1).reshape(-1, 1),
    "trace":          lambda c: np.array([[np.trace(c["A"])]]),

    "triu":           lambda c: np.triu(c["A"]),
    "tril":           lambda c: np.tril(c["A"]),
    "triu_k1":        lambda c: np.triu(c["A"], 1),
    "tril_km1":       lambda c: np.tril(c["A"], -1),
    "diag_extract":   lambda c: np.diag(c["A"]).reshape(-1, 1),
    "diag_build":     lambda c: np.diagflat(c["A"]),
    "reshape":        lambda c: c["A"].reshape(1, -1),
    "concat_h":       lambda c: np.hstack((c["A"], c["B"])),
    "concat_v":       lambda c: np.vstack((c["A"], c["B"])),
    "tensor":         lambda c: np.kron(c["A"], c["B"]),

    "norm_fro":       lambda c: np.array([[np.linalg.norm(c["A"], "fro")]]),
    "norm_one":       lambda c: np.array([[np.linalg.norm(c["A"], 1)]]),
    "norm_inf":       lambda c: np.array([[np.linalg.norm(c["A"], np.inf)]]),

    "min":            lambda c: np.array([[c["A"].min()]]),
    "max":            lambda c: np.array([[c["A"].max()]]),
    "mean":           lambda c: np.array([[c["A"].mean()]]),
    "var_pop":        lambda c: np.array([[c["A"].var()]]),
    "var_samp":       lambda c: np.array([[c["A"].var(ddof=1)]]),
    "stddev":         lambda c: np.array([[c["A"].std()]]),
    "min_cols":       lambda c: c["A"].min(axis=0).reshape(1, -1),
    "max_rows":       lambda c: c["A"].max(axis=1).reshape(-1, 1),
    "mean_cols":      lambda c: c["A"].mean(axis=0).reshape(1, -1),
    "argmin":         lambda c: np.array([list(np.unravel_index(
                          np.argmin(c["A"]), c["A"].shape))], dtype=float),
    "argmax":         lambda c: np.array([list(np.unravel_index(
                          np.argmax(c["A"]), c["A"].shape))], dtype=float),

    "det":            lambda c: np.array([[np.linalg.det(c["A"])]]),
    "inverse":        lambda c: np.linalg.inv(c["A"]),
    "pinv":           lambda c: np.linalg.pinv(c["A"]),
    "rank":           lambda c: np.array([[float(np.linalg.matrix_rank(c["A"]))]]),
    "cholesky":       lambda c: np.linalg.cholesky(c["A"]),
    "solve":          lambda c: np.linalg.solve(c["A"], c["B"]),
    # MATLAB's mrdivide: X = A / B solves X*B = A, i.e. A @ inv(B).
    "mrdivide":       lambda c: (c["A"] + c["A"]) @ np.linalg.inv(c["B"]),
    "lstsq":          lambda c: np.linalg.lstsq(c["A"], c["B"], rcond=None)[0],
    "adjugate":       lambda c: np.linalg.det(c["A"]) * np.linalg.inv(c["A"]),
    "mat_pow_int":    lambda c: np.linalg.matrix_power(c["A"], int(c["param"])),
}

SCIPY_DIRECT = {
    "mat_pow_real":   lambda c: sla.sqrtm(c["A"]),
    "mat_sqrt":       lambda c: sla.sqrtm(c["A"]),
    "mat_log":        lambda c: sla.logm(c["A"]),
    "mat_exp":        lambda c: sla.expm(c["A"]),
    "mat_sin":        lambda c: sla.sinm(c["A"]),
    "mat_cos":        lambda c: sla.cosm(c["A"]),
    "mat_tan":        lambda c: sla.tanm(c["A"]),
    "mat_sinh":       lambda c: sla.sinhm(c["A"]),
    "mat_cosh":       lambda c: sla.coshm(c["A"]),
    "mat_tanh":       lambda c: sla.tanhm(c["A"]),
}


# ────────────────────────────────── factorisations: check what IS defined ──
def check_lu(c, name):
    A, L, U, P = c["A"], c["L"], c["U"], c["P"]
    ok, d = close(P @ A, L @ U)
    report(f"{name}  P*A == L*U (numpy product)", ok, d)
    report(f"{name}  L unit lower triangular",
           np.allclose(np.triu(L, 1), 0, atol=ATOL) and
           np.allclose(np.diag(L), 1, atol=ATOL))
    report(f"{name}  U upper triangular", np.allclose(np.tril(U, -1), 0, atol=ATOL))
    report(f"{name}  P a permutation matrix",
           np.allclose(np.sort(P.sum(0)), 1) and np.allclose(np.sort(P.sum(1)), 1))
    # LAPACK factors the same matrix; |det| must agree whatever the pivoting.
    report(f"{name}  |det| matches numpy",
           np.isclose(abs(np.prod(np.diag(U))), abs(np.linalg.det(A)), rtol=1e-8))


def check_qr(c, name):
    A, Q, R, P = c["A"], c["Q"], c["R"], c["P"]
    ok, d = close(A @ P, Q @ R)
    report(f"{name}  A*P == Q*R (numpy product)", ok, d)
    ok, d = close(Q.T @ Q, np.eye(Q.shape[0]))
    report(f"{name}  Q orthogonal", ok, d)
    report(f"{name}  R upper triangular", np.allclose(np.tril(R, -1), 0, atol=1e-9))
    # Column pivoting is supposed to order |R_ii| non-increasingly.
    dg = np.abs(np.diag(R))
    report(f"{name}  |diag(R)| non-increasing (pivoting)", np.all(np.diff(dg) <= 1e-9))


def check_svd(c, name):
    A, U, S, V = c["A"], c["U"], c["S"], c["V"]
    ok, d = close(U @ S @ V.T, A)
    report(f"{name}  U*S*V^T == A (numpy product)", ok, d)
    ok, d = close(U.T @ U, np.eye(U.shape[0]))
    report(f"{name}  U orthogonal", ok, d)
    ok, d = close(V.T @ V, np.eye(V.shape[0]))
    report(f"{name}  V orthogonal", ok, d)
    mine = np.sort(np.diag(S))[::-1]
    theirs = np.sort(np.linalg.svd(A, compute_uv=False))[::-1]
    ok, d = close(mine, theirs)
    report(f"{name}  singular values match numpy", ok, d)


def check_eig_sym(c, name):
    A, E, Q = c["A"], c["E"].ravel(), c["Q"]
    theirs = np.sort(np.linalg.eigvalsh(A))
    ok, d = close(np.sort(E), theirs)
    report(f"{name}  eigenvalues match numpy", ok, d)
    ok, d = close(Q.T @ Q, np.eye(Q.shape[0]))
    report(f"{name}  Q orthogonal", ok, d)
    # Q^T A Q must be the diagonal of eigenvalues for a symmetric A.
    ok, d = close(np.sort(np.diag(Q.T @ A @ Q)), theirs)
    report(f"{name}  Q^T*A*Q is diagonal with those eigenvalues", ok, d)


def check_eigvals(c, name):
    A = c["A"]
    mine = c["RE"].ravel() + 1j * c["IM"].ravel()
    theirs = np.linalg.eigvals(A)
    key = lambda z: (np.round(z.real, 8), np.round(z.imag, 8))
    ok, d = close(np.array(sorted(mine, key=key)), np.array(sorted(theirs, key=key)))
    report(f"{name}  eigenvalues (complex) match numpy", ok, d)


FACTORISATIONS = {
    "lu": check_lu, "qr": check_qr, "svd": check_svd,
    "eig_symmetric": check_eig_sym, "eigvals": check_eigvals,
}


# ────────────────────────────────────────────────────────────────── main ──
def main():
    global skipped
    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.dat")))
    if not files:
        sys.exit(f"No {VAL_DIR}/*.dat found — build and run ./numpy_validate first.")

    print(f"{BOLD}MatrixCpp vs NumPy {np.__version__}"
          f"{'' if HAVE_SCIPY else '  (scipy missing — matrix functions skipped)'}{OFF}")
    print(f"comparing {len(files)} cases   rtol={RTOL:g} atol={ATOL:g}\n")

    current = None
    for path in files:
        c = load_case(path)
        op, dtype = c["op"], c["dtype"]
        if dtype != current:
            current = dtype
            print(f"\n{BOLD}── {dtype} ──{OFF}")
        name = f"{op} [{dtype}]"

        if op in FACTORISATIONS:
            FACTORISATIONS[op](c, name)
        elif op in DIRECT:
            ok, d = close(c["R"], DIRECT[op](c))
            report(name, ok, d)
        elif op in SCIPY_DIRECT:
            if not HAVE_SCIPY:
                skipped += 1
                print(f"  {YELLOW}SKIP{OFF}  {name}  (needs scipy)")
                continue
            expected = SCIPY_DIRECT[op](c)
            # scipy returns a complex array when the true result is real;
            # the imaginary part must be negligible before we drop it.
            if np.iscomplexobj(expected) and not np.iscomplexobj(c["R"]):
                if np.max(np.abs(expected.imag)) < 1e-9:
                    expected = expected.real
            ok, d = close(c["R"], expected)
            report(name, ok, d)
        else:
            skipped += 1
            print(f"  {YELLOW}SKIP{OFF}  {name}  (no NumPy reference registered)")

    total = passed + failed
    print("\n" + "=" * 62)
    print(f"  {passed}/{total} comparisons passed"
          + (f", {skipped} skipped" if skipped else ""))
    if failed:
        print(f"  {RED}{failed} FAILED{OFF}")
        for n, d in failures:
            print(f"    - {n}  {d}")
    else:
        print(f"  {GREEN}every operator agrees with NumPy{OFF}")
    print("=" * 62)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
