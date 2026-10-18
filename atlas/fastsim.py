"""Vectorised multi-horizon score streams and e-process paths.

Used by the horizon-resolved, interleaving and ablation experiments, which need many
horizons x many chains x many seeds. Everything here is exact (no approximation of the
ATLAS construction): the calibration excess ``Z = mahal^2/d - 1`` of the h-step Gaussian
predictive is computed for *every* issue time at once, and the mixture-betting wealth is
evaluated for a whole path with cumulative sums.
"""

from __future__ import annotations

import numpy as np

__all__ = ["excess_all_issue_times", "chain", "log_wealth_path", "first_cross",
           "lam_grid"]


def excess_all_issue_times(sim, z, a, h, clip_hi=None):
    """Calibration excess for the h-step predictive issued at every ``s=0..T-h``.

    ``z`` has shape ``(T+1, d)``, ``a`` shape ``(T, m)``. Returns an array of length
    ``T-h+1``; entry ``s`` resolves at time ``s+h``.
    """
    T = a.shape[0]
    d = sim.dim
    Am, Bm = sim.A_model, sim.B_model
    n = T - h + 1
    # mean: A^h z_s + sum_i A^{h-1-i} B a_{s+i}  (Horner-style accumulation)
    mu = np.zeros((n, d))
    for i in range(h):
        mu = mu @ Am.T + a[i:i + n] @ Bm.T
    # the loop above builds sum_i A^{h-1-i} B a_{s+i}; add A^h z_s
    Ah = np.linalg.matrix_power(Am, h)
    mu = mu + z[:n] @ Ah.T
    Sigma = np.zeros((d, d))
    P = np.eye(d)
    for _ in range(h):
        Sigma += P @ sim.Q_model @ P.T
        P = Am @ P
    Sigma += 1e-9 * np.eye(d)
    diff = z[h:h + n] - mu
    sol = np.linalg.solve(Sigma, diff.T).T
    ex = np.einsum("ij,ij->i", diff, sol) / d - 1.0
    hi = sim.B if clip_hi is None else clip_hi
    return np.clip(ex, -1.0, hi)


def chain(ex, h, offset=0):
    """Non-overlapping sub-chain of issue times ``offset, offset+h, ...``.

    Returns ``(values, resolve_times)`` where ``resolve_times = issue + h``.
    """
    idx = np.arange(offset, len(ex), h)
    return ex[idx], idx + h


def lam_grid(z_lo, z_hi, n=20, scale=1.0):
    lam_max = scale / (z_hi - z_lo)
    return np.linspace(lam_max / n, lam_max, n)


def log_wealth_path(zv, eps, lams):
    """Mixture-betting log-wealth path for a stream ``zv`` (uniform prior over ``lams``)."""
    d = zv - eps
    comp = np.log(np.maximum(1.0 + np.outer(d, lams), 1e-300))        # (n, m)
    cs = np.cumsum(comp, axis=0)
    mx = cs.max(axis=1, keepdims=True)
    return (mx[:, 0] + np.log(np.mean(np.exp(cs - mx), axis=1)))


def first_cross(logw, thresh):
    """Index of the first time the log-wealth reaches ``thresh`` (or ``None``)."""
    hit = np.nonzero(logw >= thresh)[0]
    return int(hit[0]) if len(hit) else None
