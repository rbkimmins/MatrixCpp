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
    import scipy.signal as ssig
    import scipy.integrate as sint
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

def _rref(A, tol=None):
    """Reduced row echelon form — NumPy has no rref, so here is the reference."""
    R = A.astype(float).copy()
    rows, cols = R.shape
    if tol is None:
        tol = np.finfo(float).eps * max(rows, cols) * max(1.0, np.abs(R).sum(axis=1).max())
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        piv = row + int(np.argmax(np.abs(R[row:, col])))
        if abs(R[piv, col]) <= tol:
            R[row:, col] = 0.0
            continue
        if piv != row:
            R[[row, piv]] = R[[piv, row]]
        R[row] = R[row] / R[row, col]
        for i in range(rows):
            if i != row and R[i, col] != 0.0:
                R[i] = R[i] - R[i, col] * R[row]
        row += 1
    return R


# np.trapz was renamed np.trapezoid in NumPy 2.0. Bind whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz

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
    "kron":         lambda c: np.kron(c["A"], c["B"]),

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
    # --- tier 6: convolution, polynomials, calculus, interpolation ---
    "conv":           lambda c: np.convolve(c["A"].ravel(), c["B"].ravel()).reshape(1, -1),
    "conv_fft":       lambda c: np.convolve(c["A"].ravel(), c["B"].ravel()).reshape(1, -1),
    # NumPy's polydiv returns (quotient, remainder) with the same descending
    # coefficient convention this library uses.
    "deconv_q":       lambda c: np.polydiv(c["A"].ravel(), c["B"].ravel())[0].reshape(1, -1),
    "poly_roots":     lambda c: np.poly(c["A"].ravel()).reshape(1, -1),
    # np.poly of a square matrix is its characteristic polynomial, same rule.
    "poly_charpoly":  lambda c: np.poly(c["A"]).reshape(1, -1),
    # np.trapezoid is NumPy 2.0's name for what 1.x calls np.trapz; accept either
    # so this file does not pin a NumPy version for one reference.
    "trapz_unit":     lambda c: np.array([[_trapz(c["A"].ravel())]]),
    "trapz_xy":       lambda c: np.array([[_trapz(c["B"].ravel(), c["A"].ravel())]]),
    # SciPy's cumulative_trapezoid with initial=0 matches MATLAB's cumtrapz.
    "cumtrapz":       lambda c: sint.cumulative_trapezoid(
                          c["A"].ravel(), initial=0).reshape(1, -1),
    "gradient":       lambda c: np.gradient(c["A"].ravel()).reshape(1, -1),
    "gradient_h":     lambda c: np.gradient(c["A"].ravel(), 0.5).reshape(1, -1),
    # np.interp CLAMPS outside the range where ours returns NaN, so the query
    # points here are all interior and the two agree.
    "interp1":        lambda c: np.interp(c["B"].ravel(), c["A"].ravel(),
                          np.sin(c["A"].ravel())).reshape(1, -1),
    "filter":         lambda c: ssig.lfilter([0.2, 0.5, 0.3], [1.0, -0.4, 0.1],
                          c["A"].ravel()).reshape(1, -1),

    # --- tier 6: the FFT ---
    # NumPy puts the whole 1/n on ifft and none on fft, same as MATLAB, which is
    # why these references are one-liners rather than rescalings.
    "fft_re":         lambda c: np.fft.fft(c["A"].ravel()).real.reshape(1, -1),
    "fft_im":         lambda c: np.fft.fft(c["A"].ravel()).imag.reshape(1, -1),
    "fft_prime_re":   lambda c: np.fft.fft(c["A"].ravel()).real.reshape(1, -1),
    "fft_prime_im":   lambda c: np.fft.fft(c["A"].ravel()).imag.reshape(1, -1),
    "fft_pad_re":     lambda c: np.fft.fft(c["A"].ravel(), 100).real.reshape(1, -1),
    "fft_trunc_re":   lambda c: np.fft.fft(c["A"].ravel(), 20).real.reshape(1, -1),
    # NumPy's axis=0 walks down columns, which is MATLAB's default for fft.
    "fft_cols_re":    lambda c: np.fft.fft(c["A"], axis=0).real,
    "fft_cols_im":    lambda c: np.fft.fft(c["A"], axis=0).imag,
    "fft_rows_re":    lambda c: np.fft.fft(c["A"], axis=1).real,
    "ifft_re":        lambda c: np.fft.ifft(c["A"].ravel()).real.reshape(1, -1),
    "ifft_im":        lambda c: np.fft.ifft(c["A"].ravel()).imag.reshape(1, -1),
    "fftshift":       lambda c: np.fft.fftshift(c["A"].ravel()).reshape(1, -1),

    # --- sequences, shape and constructors (tier 5) ---
    # These take no meaningful input, so "A" is a placeholder the dumper needs.
    "linspace":       lambda c: np.linspace(-2.0, 3.0, 11).reshape(1, -1),
    "logspace":       lambda c: np.logspace(-1.0, 2.0, 7).reshape(1, -1),
    "range":          lambda c: np.arange(0.0, 9.0 + 1e-12, 2.0).reshape(1, -1),
    "hilb5":          lambda c: sla.hilbert(5),
    "pascal4":        lambda c: sla.pascal(4).astype(float),
    "fliplr":         lambda c: np.fliplr(c["A"]),
    "flipud":         lambda c: np.flipud(c["A"]),
    # NumPy's rot90 is counterclockwise, which is what MATLAB's is too.
    "rot90_1":        lambda c: np.rot90(c["A"], 1),
    "rot90_m1":       lambda c: np.rot90(c["A"], -1),
    "repmat":         lambda c: np.tile(c["A"], (2, 3)),
    "circ_row":       lambda c: np.roll(c["A"], 1, axis=0),
    "circ_col":       lambda c: np.roll(c["A"], -2, axis=1),
    "blkdiag":        lambda c: sla.block_diag(c["A"], c["B"]),
    "toeplitz":       lambda c: sla.toeplitz(c["A"].ravel()),
    # NumPy's vander is ascending by default; ours is descending, like polyval.
    "vander":         lambda c: np.vander(c["A"].ravel(), increasing=False),

    # --- least squares with Q left implicit ---
    "lstsq_tall":     lambda c: np.linalg.lstsq(c["A"], c["B"], rcond=None)[0],
    "rank_tall":      lambda c: np.array([[float(np.linalg.matrix_rank(c["A"]))]]),

    # --- complex linear algebra (work_t) ---
    "cx_inverse":     lambda c: np.linalg.inv(c["A"]),
    "cx_det":         lambda c: np.array([[np.linalg.det(c["A"])]]),
    "cx_solve":       lambda c: np.linalg.solve(c["A"], c["B"]),
    "cx_pinv":        lambda c: np.linalg.pinv(c["A"]),
    # Singular values are unique where U and V are not, so they compare directly.
    "cx_svdvals":     lambda c: np.linalg.svd(c["A"], compute_uv=False).reshape(-1, 1),
    "cx_norm2":       lambda c: np.array([[np.linalg.norm(c["A"], 2)]]),
    # NumPy's cholesky returns the LOWER factor with A == L @ L.conj().T, same
    # convention as ours.
    "cx_chol":        lambda c: np.linalg.cholesky(c["A"]),
    "cx_expm":        lambda c: sla.expm(c["A"]),

    # --- det / LU of a singular matrix ---
    # NumPy returns 0.0 for a singular determinant rather than raising, which is
    # the behaviour these cases pin down.
    "det_singular":    lambda c: np.array([[np.linalg.det(c["A"])]]),
    "det_zeros":       lambda c: np.array([[np.linalg.det(c["A"])]]),
    "det_nonsingular": lambda c: np.array([[np.linalg.det(c["A"])]]),

    # --- funm and generalized eigenvalues (tier 4) ---
    "funm_exp":       lambda c: sla.expm(c["A"]),
    # SciPy's eigh solves exactly this symmetric-definite problem and returns the
    # eigenvalues ascending, same as ours.
    "geneig_sym":     lambda c: sla.eigh(c["A"], c["B"])[0].reshape(-1, 1),
    "geneig_orth":    lambda c: np.eye(c["A"].shape[0]),

    # --- decomposition and condition estimates (tier 4) ---
    # A factorisation is only useful if it solves the same system, so the
    # reference is NumPy's own solve, not a re-implementation of ours.
    "decomp_solve":   lambda c: np.linalg.solve(c["A"], c["B"]),
    "decomp_det":     lambda c: np.array([[np.linalg.det(c["A"])]]),
    # The Hager-Higham estimate is a lower bound in general; on a 5x5 Hilbert
    # matrix it lands exactly on the true 1-norm condition number.
    "condest_hilbert": lambda c: np.array([[np.linalg.cond(c["A"], 1)]]),
    # NumPy's lstsq returns the minimum-norm solution for an under-determined
    # system, which is exactly what lsqminnorm promises.
    "lsqminnorm":     lambda c: np.linalg.lstsq(c["A"], c["B"], rcond=None)[0],

    # --- element-wise maths (tier 3) ---
    "sign":           lambda c: np.sign(c["A"]),
    "floor":          lambda c: np.floor(c["A"]),
    "ceil":           lambda c: np.ceil(c["A"]),
    # NumPy's round is banker's rounding; MATLAB's and C's send halves away from
    # zero. The test data avoids exact halves so the two agree.
    "round":          lambda c: np.round(c["A"]),
    "fix":            lambda c: np.fix(c["A"]),
    "mod3":           lambda c: np.mod(c["A"], 3.0),
    "rem3":           lambda c: np.fmod(c["A"], 3.0),
    "expm1":          lambda c: np.expm1(c["A"]),
    "asinh":          lambda c: np.arcsinh(c["A"]),
    "atanh":          lambda c: np.arctanh(c["A"]),
    "angle_real":     lambda c: np.angle(c["A"]),
    "atan2":          lambda c: np.arctan2(c["A"], c["B"]),
    "hypot":          lambda c: np.hypot(c["A"], c["B"]),

    # --- structure, subspaces and polynomials (tier 4) ---
    "normest":        lambda c: np.array([[np.linalg.norm(c["A"], 2)]]),
    "bandwidth_lo":   lambda c: np.array([[float(max(
                          (i - j for i in range(c["A"].shape[0])
                                 for j in range(c["A"].shape[1])
                                 if i > j and abs(c["A"][i, j]) > 0), default=0))]]),
    "rref":           lambda c: _rref(c["A"]),
    "null_dim":       lambda c: np.array([[float(
                          c["A"].shape[1] - np.linalg.matrix_rank(c["A"]))]]),
    "orth_dim":       lambda c: np.array([[float(np.linalg.matrix_rank(c["A"]))]]),
    # Bases are not unique; the projectors onto those subspaces are.
    "orth_proj":      lambda c: (lambda o: o @ o.T)(sla.orth(c["A"])),
    "null_proj":      lambda c: (lambda n: n @ n.T)(sla.null_space(c["A"])),
    "cross":          lambda c: np.cross(c["A"].ravel(), c["B"].ravel()).reshape(-1, 1),
    "dot":            lambda c: np.array([[float(np.vdot(c["A"], c["B"]))]]),
    "polyval":        lambda c: np.polyval(c["A"].ravel(), c["B"].ravel()).reshape(
                          c["B"].shape),
    "roots_moduli":   lambda c: np.sort(np.abs(np.roots(c["A"].ravel()))).reshape(-1, 1),
    "polyfit":        lambda c: np.polyfit(c["A"].ravel(), c["B"].ravel(), 3).reshape(-1, 1),

    # --- scans, orderings and multiset reductions (tier 2) ---
    # NumPy's axis=0 walks DOWN columns, which is this library's addcol=false.
    "prod_all":       lambda c: np.array([[c["A"].prod()]]),
    "prod_col":       lambda c: c["A"].prod(axis=0).reshape(1, -1),
    "prod_row":       lambda c: c["A"].prod(axis=1).reshape(-1, 1),
    "cumsum_col":     lambda c: np.cumsum(c["A"], axis=0),
    "cumsum_row":     lambda c: np.cumsum(c["A"], axis=1),
    "cumprod_col":    lambda c: np.cumprod(c["A"], axis=0),
    "diff_col":       lambda c: np.diff(c["A"], axis=0),
    "diff_row":       lambda c: np.diff(c["A"], axis=1),
    "sort_col":       lambda c: np.sort(c["A"], axis=0),
    "sort_row":       lambda c: np.sort(c["A"], axis=1),
    "sort_desc":      lambda c: -np.sort(-c["A"], axis=0),
    "median_all":     lambda c: np.array([[np.median(c["A"])]]),
    "median_col":     lambda c: np.median(c["A"], axis=0).reshape(1, -1),
    "median_row":     lambda c: np.median(c["A"], axis=1).reshape(-1, 1),
    "sortrows_k0":    lambda c: c["A"][np.argsort(c["A"][:, 0], kind="stable")],
    "unique":         lambda c: np.unique(c["A"]).reshape(-1, 1),
    # NumPy has no mode; ties go to the smallest, which is what this library does.
    "mode":           lambda c: np.array([[float(
                          min(np.unique(c["A"]),
                              key=lambda v: (-np.count_nonzero(c["A"] == v), v)))]]),

    # --- logical masks, dumped as 0/1 so they compare directly ---
    "mask_gt":        lambda c: (c["A"] > 0.0).astype(float),
    "mask_le":        lambda c: (c["A"] <= 0.0).astype(float),
    "mask_band":      lambda c: ((c["A"] > -0.5) & (c["A"] < 0.5)).astype(float),
    "mask_bor":       lambda c: ((c["A"] > 0.5) | (c["A"] < -0.5)).astype(float),
    "mask_bxor":      lambda c: ((c["A"] > 0.0) ^ (c["A"] > 0.5)).astype(float),
    "mask_not":       lambda c: (~(c["A"] > 0.0)).astype(float),
    # NumPy selects in row-major (C) order too, so this lines up directly.
    "mask_select":    lambda c: c["A"][c["A"] > 0.0].reshape(-1, 1),
    "mask_assign":    lambda c: np.where(c["A"] < 0.0, 0.0, c["A"]),
    "mask_nnz":       lambda c: np.array([[float(np.count_nonzero(c["A"] > 0.0))]]),
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


def check_schur_complex(c, name):
    A, T, Q = c["A"], c["T"], c["Q"]
    # A complex Schur form is not unique (any eigenvalue ordering is valid), so
    # the DEFINING properties are what to check, plus agreement with SciPy on
    # the eigenvalue multiset.
    ok, d = close(Q @ T @ Q.conj().T, A)
    report(f"{name}  A == Q T Q^H", ok, d)
    ok, d = close(Q.conj().T @ Q, np.eye(A.shape[0]))
    report(f"{name}  Q unitary", ok, d)
    report(f"{name}  T upper triangular", np.allclose(np.tril(T, -1), 0), "")
    if HAVE_SCIPY:
        key = lambda z: (round(z.real, 8), round(z.imag, 8))
        mine = sorted(np.diag(T), key=key)
        theirs = sorted(sla.schur(A, output="complex")[0].diagonal(), key=key)
        ok, d = close(np.array(mine), np.array(theirs))
        report(f"{name}  eigenvalues match scipy.linalg.schur", ok, d)


def check_hess_complex(c, name):
    A, H, Q = c["A"], c["H"], c["Q"]
    ok, d = close(Q @ H @ Q.conj().T, A)
    report(f"{name}  A == Q H Q^H", ok, d)
    report(f"{name}  H is Hessenberg", np.allclose(np.tril(H, -2), 0), "")
    if HAVE_SCIPY:
        # The reduction is unique only up to phase, so compare the invariant:
        # |subdiagonal|, which any correct Hessenberg reduction reproduces.
        theirs = sla.hessenberg(A)
        ok, d = close(np.abs(np.diag(H, -1)), np.abs(np.diag(theirs, -1)))
        report(f"{name}  |subdiagonal| matches scipy.linalg.hessenberg", ok, d)


def check_eig_complex(c, name):
    A, E, X = c["A"], c["E"].ravel(), c["X"]
    res = max(np.linalg.norm(A @ X[:, k] - E[k] * X[:, k]) for k in range(len(E)))
    report(f"{name}  A x == lambda x for every column", res < 1e-8, f"max resid {res:.2e}")
    key = lambda z: (round(z.real, 8), round(z.imag, 8))
    ok, d = close(np.array(sorted(E, key=key)),
                  np.array(sorted(np.linalg.eigvals(A), key=key)))
    report(f"{name}  eigenvalues match numpy", ok, d)


def check_qz(c, name):
    A, B, S, T, Q, Z = c["A"], c["B"], c["S"], c["T"], c["Q"], c["Z"]
    mine = c["RE"].ravel() + 1j * c["IM"].ravel()
    # The generalized Schur form is not unique (any eigenvalue ordering is a
    # valid one), so the DEFINING properties are what to check, and SciPy is
    # asked only for the eigenvalue multiset.
    ok, d = close(Q @ S @ Z.conj().T, A)
    report(f"{name}  A == Q S Z^H", ok, d)
    ok, d = close(Q @ T @ Z.conj().T, B)
    report(f"{name}  B == Q T Z^H", ok, d)
    n = A.shape[0]
    ok, d = close(Q.conj().T @ Q, np.eye(n))
    report(f"{name}  Q unitary", ok, d)
    ok, d = close(Z.conj().T @ Z, np.eye(n))
    report(f"{name}  Z unitary", ok, d)
    report(f"{name}  S upper triangular", np.allclose(np.tril(S, -1), 0), "")
    report(f"{name}  T upper triangular", np.allclose(np.tril(T, -1), 0), "")
    # Eigenvectors are checked in the HOMOGENEOUS form beta*A*x == alpha*B*x,
    # which stays finite when beta is zero. They are not compared to scipy's
    # directly: an eigenvector is only defined up to a scalar.
    X = c["X"]
    al, be = np.diag(S), np.diag(T)
    res = max(np.linalg.norm(be[k] * (A @ X[:, k]) - al[k] * (B @ X[:, k]))
              for k in range(n))
    report(f"{name}  beta*A*x == alpha*B*x for every column", res < 1e-8, f"max resid {res:.2e}")
    if HAVE_SCIPY:
        theirs = sla.eig(A, B, right=False)
        key = lambda z: (round(z.real, 6), round(z.imag, 6))
        m = np.array(sorted(mine, key=key))
        t = np.array(sorted(theirs, key=key))
        d = float(np.max(np.abs(m - t) / np.maximum(np.abs(t), 1.0)))
        report(f"{name}  eigenvalues match scipy.linalg.eig(A,B)", d < 1e-8, f"relative {d:.2e}")
        # And as (alpha,beta) pairs on the Riemann sphere — the scale-invariant
        # comparison, and the only one that can compare an INFINITE eigenvalue.
        sa, sb = sla.eig(A, B, right=False, homogeneous_eigvals=True)
        def chord(p, q):
            return abs(p[0] * q[1] - p[1] * q[0]) / (
                np.sqrt(abs(p[0]) ** 2 + abs(p[1]) ** 2) *
                np.sqrt(abs(q[0]) ** 2 + abs(q[1]) ** 2))
        theirs_p = list(zip(sa, sb))
        used = [False] * n
        worst = 0.0
        for p in zip(al, be):
            best, bj = 9e9, -1
            for j, q in enumerate(theirs_p):
                if used[j]:
                    continue
                dd = chord(p, q)
                if dd < best:
                    best, bj = dd, j
            used[bj] = True
            worst = max(worst, best)
        report(f"{name}  (alpha,beta) match scipy on the Riemann sphere", worst < 1e-8,
               f"chordal {worst:.2e}")


def _scipy_fn(name_, fn):
    def check(c, cname):
        if not HAVE_SCIPY:
            global skipped
            skipped += 1
            print(f"  {YELLOW}SKIP{OFF}  {cname}  (scipy missing)")
            return
        ok, d = close(c["R"], fn(c["A"]))
        report(f"{cname}  matches scipy.linalg.{name_}", ok, d)
    return check


FACTORISATIONS = {
    "lu": check_lu, "qr": check_qr, "svd": check_svd,
    "eig_symmetric": check_eig_sym, "eigvals": check_eigvals,
    "schur_complex": check_schur_complex,
    "hess_complex": check_hess_complex,
    "eig_complex": check_eig_complex,
    "sqrtm_complex": _scipy_fn("sqrtm", lambda A: sla.sqrtm(A)),
    "logm_complex": _scipy_fn("logm", lambda A: sla.logm(A)),
    "expm_complex": _scipy_fn("expm", lambda A: sla.expm(A)),
    "qz_complex": check_qz,
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
