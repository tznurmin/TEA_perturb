from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import postprocess  # noqa: E402

from oracles import (assert_all_finite, cov_n1, isotropic_gaussian,
                     rotated_anisotropy)


def _top_pc(X: np.ndarray) -> np.ndarray:
    """Top principal component (unit vector) under sample covariance denom (n-1)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = cov_n1(Xc)
    w, E = np.linalg.eigh(C)
    v = E[:, -1]
    v = v / np.linalg.norm(v)
    return v


def test_abtt_k1_removes_variance_along_top_pc():
    # use an anisotropic rotated dataset so there is a meaningful top component
    X = rotated_anisotropy(n=8000, d=32, seed=10).astype(np.float32)

    v1 = _top_pc(X)
    Xc = X - X.mean(axis=0, keepdims=True)

    before = float(np.var(Xc @ v1))

    mats = [X]
    params = postprocess.fit("abtt", mats, abtt_k=1, max_fit_rows=X.shape[0], seed=0)
    Y = postprocess.transform(mats, params)[0].astype(np.float64)

    assert_all_finite(Y)

    after = float(np.var(Y @ v1))

    # ABTT k=1 should collapse the top component variance strongly
    # use a loose ratio because float32 storage and rotated covariance can leak tiny energy
    assert after <= 1e-3 * before


def test_abtt_k0_is_identity():
    X = isotropic_gaussian(n=1024, d=16, seed=11).astype(np.float32)

    mats = [X]
    params = postprocess.fit("abtt", mats, abtt_k=0, max_fit_rows=X.shape[0], seed=0)
    Y = postprocess.transform(mats, params)[0]

    # with k=0 the fit returns method=none and transform is identity
    assert np.max(np.abs(Y - X)) == 0.0
