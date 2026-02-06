from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ensure we can import the src modules (scripts are flat modules in tea_spec/src)
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import postprocess  # noqa: E402

from oracles import (apply_center_and_matrix, assert_all_finite,
                     constant_vectors, cov_errors_identity, cov_n, cov_n1,
                     exact_diagonal_covariance, isotropic_gaussian, low_rank,
                     mean_inf, rotated_anisotropy, zca_whiten_params_svd)


def _whiten(
    X: np.ndarray, *, eps: float = 1e-6, max_fit_rows: int | None = None, seed: int = 0
):
    mats = [X.astype(np.float32, copy=False)]
    params = postprocess.fit(
        "whiten",
        mats,
        whiten_eps=eps,
        max_fit_rows=max_fit_rows or X.shape[0],
        seed=seed,
    )
    Y = postprocess.transform(mats, params)[0].astype(np.float64)
    return params, Y


def test_whiten_isotropic_gaussian_cov_identity():
    X = isotropic_gaussian(n=5000, d=32, seed=0)
    _params, Y = _whiten(X)

    assert_all_finite(Y)
    assert mean_inf(Y) <= 1e-2

    C = cov_n1(Y)
    errs = cov_errors_identity(C)
    diag_err, offdiag = errs.diag_err_max, errs.offdiag_abs_max
    assert diag_err <= 5e-2
    assert offdiag <= 5e-2


def test_whiten_rotated_anisotropy_cov_identity():
    X = rotated_anisotropy(n=8000, d=48, seed=1)
    _params, Y = _whiten(X)

    assert_all_finite(Y)
    C = cov_n1(Y)
    errs = cov_errors_identity(C)
    diag_err, offdiag = errs.diag_err_max, errs.offdiag_abs_max

    # rotated covariance stresses eigenvector handling.
    assert diag_err <= 5e-2
    assert offdiag <= 5e-2


def test_whiten_exact_constructed_diagonal_covariance_is_tight():
    # construct Xc such that sample covariance under denom (n-1) is exactly diag(scales^2)
    X = exact_diagonal_covariance(n=256, d=32, seed=2)
    _params, Y = _whiten(X)

    C = cov_n1(Y)
    errs = cov_errors_identity(C)
    diag_err, offdiag = errs.diag_err_max, errs.offdiag_abs_max

    # this case should be very close to exact identity (up to float32 W casting)
    assert diag_err <= 1e-2
    assert offdiag <= 1e-2


def test_whiten_cov_denominator_mismatch_detectable():
    X = rotated_anisotropy(n=6000, d=32, seed=3)
    _params, Y = _whiten(X)

    C_n1 = cov_n1(Y)
    C_n = cov_n(Y)

    errs = cov_errors_identity(C_n1)
    d1, o1 = errs.diag_err_max, errs.offdiag_abs_max
    errs = cov_errors_identity(C_n)
    d0, o0 = errs.diag_err_max, errs.offdiag_abs_max

    # Whitening here is defined with denom (n-1). Using denom n must be worse.
    assert (d1 + o1) < (d0 + o0)


def test_whiten_matches_independent_svd_reference_matrix():
    X = rotated_anisotropy(n=4000, d=24, seed=4)

    mats = [X.astype(np.float32, copy=False)]
    params = postprocess.fit(
        "whiten", mats, whiten_eps=1e-6, max_fit_rows=X.shape[0], seed=0
    )

    mu_repo = params["mean"].astype(np.float64)
    W_repo = params["W"].astype(np.float64)

    mu_ref, W_ref = zca_whiten_params_svd(X.astype(np.float64), eps=1e-6)

    # mean is simple; matrix match is the real check
    assert np.max(np.abs(mu_repo - mu_ref)) <= 1e-3

    # the repo stores W in float32, so allow small absolute drift
    max_abs = float(np.max(np.abs(W_repo - W_ref)))
    assert max_abs <= 5e-3


def test_whiten_row_permutation_invariance():
    X = rotated_anisotropy(n=7000, d=16, seed=5)

    mats1 = [X.astype(np.float32, copy=False)]
    p1 = postprocess.fit(
        "whiten", mats1, whiten_eps=1e-6, max_fit_rows=X.shape[0], seed=0
    )

    rng = np.random.default_rng(123)
    perm = rng.permutation(X.shape[0])
    mats2 = [X[perm].astype(np.float32, copy=False)]
    p2 = postprocess.fit(
        "whiten", mats2, whiten_eps=1e-6, max_fit_rows=X.shape[0], seed=0
    )

    # floating reduction order may drift slightly, require close agreement
    assert (
        np.max(np.abs(p1["mean"].astype(np.float64) - p2["mean"].astype(np.float64)))
        <= 1e-5
    )
    assert (
        np.max(np.abs(p1["W"].astype(np.float64) - p2["W"].astype(np.float64))) <= 1e-3
    )


def test_whiten_chunked_transform_equivalence_exact():
    X = rotated_anisotropy(n=10000, d=32, seed=6).astype(np.float32)

    mats = [X]
    params = postprocess.fit(
        "whiten", mats, whiten_eps=1e-6, max_fit_rows=X.shape[0], seed=0
    )
    Y_full = postprocess.transform(mats, params)[0]

    mu = params["mean"].astype(np.float32)
    W = params["W"].astype(np.float32)

    parts = []
    step = 777
    for i in range(0, X.shape[0], step):
        Xi = X[i : i + step]
        parts.append((Xi - mu) @ W)
    Y_chunk = np.vstack(parts)

    # bitwise equality is not guaranteed across different BLAS kernels / matrix shapes
    # the operation is per-row but float32 reduction order can vary across GEMM calls
    diff = float(np.max(np.abs(Y_full - Y_chunk)))
    assert diff <= 2e-5


def test_whiten_low_rank_nullspace_behavior():
    X = low_rank(n=8000, d=40, rank=8, seed=7)
    _params, Y = _whiten(X)

    C = cov_n1(Y)
    w = np.linalg.eigvalsh(C)
    w = np.sort(w)[::-1]

    top = w[:8]
    tail = w[8:]

    assert np.max(np.abs(top - 1.0)) <= 5e-2
    assert np.max(tail) <= 5e-2


def test_whiten_constant_vectors_remain_zero():
    X = constant_vectors(n=4096, d=32, value=3.14)
    _params, Y = _whiten(X)

    assert_all_finite(Y)
    assert float(np.max(np.abs(Y))) <= 1e-8

    C = cov_n1(Y)
    assert float(np.max(np.abs(C))) <= 1e-12


def test_whiten_numeric_scale_stability():
    X = rotated_anisotropy(n=7000, d=32, seed=8)
    X_big = (1e6 * X).astype(np.float64)

    _params, Y = _whiten(X_big)

    assert_all_finite(Y)
    C = cov_n1(Y)
    errs = cov_errors_identity(C)
    diag_err, offdiag = errs.diag_err_max, errs.offdiag_abs_max

    assert diag_err <= 5e-2
    assert offdiag <= 5e-2


def test_whiten_idempotence_smoke():
    X = rotated_anisotropy(n=6000, d=24, seed=9)
    _p1, Y1 = _whiten(X)

    # re-fit whitening on already whitened output
    _p2, Y2 = _whiten(Y1)

    assert_all_finite(Y2)
    C = cov_n1(Y2)
    errs = cov_errors_identity(C)
    diag_err, offdiag = errs.diag_err_max, errs.offdiag_abs_max

    assert diag_err <= 5e-2
    assert offdiag <= 5e-2


def test_whiten_fit_transform_split_behavior():
    X = rotated_anisotropy(n=20000, d=32, seed=10)
    mats = [X.astype(np.float32, copy=False)]

    # fit on a strict subset, apply to all
    params = postprocess.fit("whiten", mats, whiten_eps=1e-6, max_fit_rows=5000, seed=0)
    Y_all = postprocess.transform(mats, params)[0].astype(np.float64)

    assert_all_finite(Y_all)

    # on the fit subset, covariance should be tight
    Y_fit = Y_all[:5000]
    C_fit = cov_n1(Y_fit)
    errs = cov_errors_identity(C_fit)
    d_fit, o_fit = errs.diag_err_max, errs.offdiag_abs_max
    assert d_fit <= 1e-1
    assert o_fit <= 1e-1

    # on the full set, it's not guaranteed to be exactly identity, but should be well-conditioned
    C_all = cov_n1(Y_all)
    cond = float(np.linalg.cond(C_all))
    assert cond <= 3.0
