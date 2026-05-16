# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Numeric oracles and synthetic data generators for postprocess validation.

This module supports deterministic unit tests for the ZCA whitening and ABTT
("all-but-the-top") transforms implemented in `tea_spec/src/postprocess.py`.

The repository's whitening fit uses the *sample covariance* convention with
DENOMINATOR (n - 1):

    C = (Xc.T @ Xc) / (n - 1)

All whitening covariance oracles in these tests use the same convention unless
explicitly noted.

These tests are a correctness proof for the numerical transform, not a claim
that whitening improves any downstream metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def mean_inf(X: np.ndarray) -> float:
    # infinity norm of the feature-wise mean
    m = X.mean(axis=0)
    return float(np.max(np.abs(m))) if m.size else 0.0


def cov_n1(X: np.ndarray) -> np.ndarray:
    # sample covariance using denominator (n - 1)
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= 1:
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    return (X.T @ X) / float(n - 1)


def cov_n(X: np.ndarray) -> np.ndarray:
    # population covariance using denominator n
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= 0:
        return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    return (X.T @ X) / float(max(1, n))


@dataclass(frozen=True)
class CovErrors:
    diag_err_max: float
    offdiag_abs_max: float


def cov_errors_identity(C: np.ndarray) -> CovErrors:
    # max absolute diagonal error from 1, and max absolute off-diagonal magnitude
    C = np.asarray(C, dtype=np.float64)
    d = C.shape[0]
    diag = np.diag(C)
    diag_err = float(np.max(np.abs(diag - 1.0))) if d else 0.0
    off = C.copy()
    np.fill_diagonal(off, 0.0)
    off_max = float(np.max(np.abs(off))) if off.size else 0.0
    return CovErrors(diag_err_max=diag_err, offdiag_abs_max=off_max)


def assert_all_finite(X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        bad = np.logical_not(np.isfinite(X))
        n_bad = int(bad.sum())
        raise AssertionError(f"found {n_bad} non-finite values")


def isotropic_gaussian(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, d), dtype=np.float64)


def _default_scales(d: int) -> np.ndarray:
    # default anisotropy scales used by generators when none are provided
    if d <= 0:
        return np.empty((0,), dtype=np.float64)
    # wide but not extreme: geometric progression from large -> small
    return np.geomspace(50.0, 0.5, num=d, dtype=np.float64)


def diagonal_anisotropy(
    n: int, d: int, *, scales: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n, d), dtype=np.float64)
    if scales is None:
        scales = _default_scales(d)
    s = np.asarray(scales, dtype=np.float64)
    if s.size != d:
        raise ValueError("scales must have length d")
    return Z * s


def _random_orthonormal(d: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(d, d), dtype=np.float64)
    Q, _ = np.linalg.qr(A)
    return Q


def rotated_anisotropy(
    n: int, d: int, *, scales: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    X = diagonal_anisotropy(n, d, scales=scales, seed=seed)
    R = _random_orthonormal(d, seed=seed + 1337)
    return X @ R


def low_rank(n: int, d: int, *, rank: int, seed: int = 0) -> np.ndarray:
    if not (1 <= rank <= d):
        raise ValueError("rank must be in [1, d]")
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(size=(n, rank), dtype=np.float64)
    A = rng.standard_normal(size=(rank, d), dtype=np.float64)
    return U @ A


def constant_vectors(n: int, d: int, *, value: float = 3.14159) -> np.ndarray:
    return np.full((n, d), float(value), dtype=np.float64)


def exact_diagonal_covariance(
    n: int, d: int, *, scales: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """
    Construct X with exact sample covariance diag(scales^2) under denom (n-1).

    Construction:
        Let Q be n x d with orthonormal columns (Q^T Q = I).
        Set Xc = sqrt(n-1) * Q * diag(scales)
        Then (Xc^T Xc)/(n-1) = diag(scales^2)

    We add a small random mean vector so fit/center logic is exercised.
    """
    if n < d:
        raise ValueError("n must be >= d for exact construction")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal(size=(n, d), dtype=np.float64)
    Q, _ = np.linalg.qr(A)

    if scales is None:
        scales = _default_scales(d)
    s = np.asarray(scales, dtype=np.float64)
    if s.size != d:
        raise ValueError("scales must have length d")

    Xc = (np.sqrt(float(n - 1)) * Q) * s

    # non-zero mean to validate mean subtraction in fit/transform
    mu = rng.standard_normal(size=(1, d), dtype=np.float64) * 0.1
    return Xc + mu


def zca_whiten_params_eigh(
    X: np.ndarray, *, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference ZCA whitening params using covariance eigendecomposition.

    Matches the repository's convention:
      - mean is removed first
      - covariance uses denom (n - 1)
      - eigenvalues are clipped to eps (not w + eps)
    Returns (mu, W).
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    n = Xc.shape[0]
    denom = float(max(1, n - 1))
    C = (Xc.T @ Xc) / denom

    w, E = np.linalg.eigh(C)  # w asc
    inv_sqrt = 1.0 / np.sqrt(np.clip(w, eps, None))
    W = (E * inv_sqrt) @ E.T
    return mu, W


def zca_whiten_params_svd(
    X: np.ndarray, *, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference ZCA whitening params using SVD of the centered data.

    For Xc = U S V^T, sample covariance under denom (n-1) is:
        C = V diag(S^2/(n-1)) V^T

    This reference matches the repository's eigenvalue clipping convention.
    Returns (mu, W).
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    n = Xc.shape[0]
    denom = float(max(1, n - 1))

    # full_matrices=False -> U: (n, d), S: (d,), Vt: (d, d)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T

    w = (S**2) / denom
    inv_sqrt = 1.0 / np.sqrt(np.clip(w, eps, None))
    W = (V * inv_sqrt) @ V.T
    return mu, W


def apply_center_and_matrix(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    return (X - mu) @ W
