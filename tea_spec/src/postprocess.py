from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

Method = Literal["none", "abtt", "whiten"]


def _stack_sample(
    mats: List[np.ndarray], max_rows: int = 100_000, seed: int = 0
) -> np.ndarray:
    """
    Stack a random subset of rows from each matrix (keeps row order within each sample).
    Used to fit the postprocessing transform on a manageable subset.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    # simple fair share per matrix
    quota = max_rows // max(1, len(mats))
    for X in mats:
        if X.shape[0] <= quota:
            chunks.append(X)
        else:
            idx = rng.choice(X.shape[0], size=quota, replace=False)
            chunks.append(X[idx])
    return (
        np.vstack(chunks)
        if chunks
        else np.empty((0, mats[0].shape[1]), dtype=np.float32)
    )


def _cov_eig(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance eigen-decomposition for centered data Xc (n x d).
    Returns (eigvals ascending, eigvecs columns).
    """
    # (d x d) covariance; use float64 for stability and cast back
    C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    w, E = np.linalg.eigh(C)
    return w, E  # w asc, columns of E are eigenvectors


def fit(
    method: Method,
    mats: List[np.ndarray],
    *,
    abtt_k: int = 10,
    whiten_eps: float = 1e-6,
    max_fit_rows: int = 100_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Fit parameters for 'abtt' (all-but-top k PCs) or 'whiten' on a sample of rows.
    Returns dict with keys: method, mean, (and either 'E_top' for abtt or 'W' for whiten).
    """
    if method == "none":
        return {"method": "none"}

    sample = _stack_sample(mats, max_rows=max_fit_rows, seed=seed).astype(
        np.float64, copy=False
    )
    if sample.size == 0:
        return {"method": "none"}

    mu = sample.mean(axis=0, keepdims=True)
    Xc = sample - mu
    w, E = _cov_eig(Xc)  # w asc

    if method == "abtt":
        k = max(0, min(abtt_k, E.shape[1]))
        if k == 0:
            return {"method": "none"}
        E_top = E[:, -k:]  # top-k eigenvectors (largest eigenvalues)
        # projector: P = I - E_top E_top^T
        return {
            "method": "abtt",
            "mean": mu.astype(np.float32),
            "E_top": E_top.astype(np.float32),
        }

    elif method == "whiten":
        # whitening matrix: W = E diag(1/sqrt(w+eps)) E^T
        inv_sqrt = 1.0 / np.sqrt(np.clip(w, whiten_eps, None))
        W = (E * inv_sqrt) @ E.T
        return {
            "method": "whiten",
            "mean": mu.astype(np.float32),
            "W": W.astype(np.float32),
        }

    else:
        return {"method": "none"}


def transform(mats: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """
    Apply fitted postprocessing to each matrix (keeps original dimensionality).
    Inputs/outputs are float32 numpy arrays.
    """
    method = params.get("method", "none")
    if method == "none":
        return [X.astype(np.float32, copy=False) for X in mats]

    mean = params["mean"].astype(np.float32)
    out: List[np.ndarray] = []
    if method == "abtt":
        E_top = params["E_top"].astype(np.float32)  # (d x k)
        # P = I - E_top E_top^T, but apply as Xc - (Xc @ E_top) @ E_top^T to avoid forming dense P
        for X in mats:
            Xf = X.astype(np.float32, copy=False)
            Xc = Xf - mean
            proj = (Xc @ E_top) @ E_top.T
            out.append((Xc - proj).astype(np.float32, copy=False))
        return out

    if method == "whiten":
        W = params["W"].astype(np.float32)  # (d x d)
        for X in mats:
            Xf = X.astype(np.float32, copy=False)
            Xc = Xf - mean
            out.append((Xc @ W).astype(np.float32, copy=False))
        return out

    return [X.astype(np.float32, copy=False) for X in mats]
