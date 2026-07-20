"""A simple *latent* world model for real image sequences.

PCA maps frames to a low-dimensional latent code; a linear dynamics model
``z_{t+1} = A z_t`` (ridge least squares) is fit on training sequences, with a
per-step Gaussian residual covariance ``Q``. The h-step predictive is the Gaussian
``N(A^h z, sum_{i<h} A^i Q (A^i)^T)``.

This is intentionally lightweight (CPU-only, no training loop) but is a genuine
learned latent simulator: it is exactly the object ATLAS monitors. The same ATLAS
code (``atlas.eprocess`` / ``atlas.frontier``) then certifies its faithfulness, so
this doubles as an end-to-end test of ATLAS on real data. A Dreamer/JEPA latent
model would slot in behind the identical interface.
"""

from __future__ import annotations

import numpy as np


class LatentLinearWM:
    def __init__(self, latent_dim=16, ridge=1e-2, max_radius=0.98):
        self.d = latent_dim
        self.ridge = ridge
        self.max_radius = max_radius

    # ------------------------------------------------------------------ fit
    def fit(self, sequences):
        """Fit PCA + linear latent dynamics from a list of ``(T, H, W)`` sequences."""
        frames = np.concatenate([s.reshape(s.shape[0], -1) for s in sequences], axis=0)
        self.mean_ = frames.mean(axis=0)
        X = frames - self.mean_
        # PCA via SVD on the (centered) frame matrix
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[: self.d]                 # (d, D)
        # latent dynamics regression on consecutive pairs, per sequence
        Z0, Z1 = [], []
        for s in sequences:
            z = self.encode(s)                          # (T, d)
            Z0.append(z[:-1]); Z1.append(z[1:])
        Z0 = np.concatenate(Z0, axis=0); Z1 = np.concatenate(Z1, axis=0)
        d = self.d
        self.A = np.linalg.solve(
            Z0.T @ Z0 + self.ridge * np.eye(d), Z0.T @ Z1).T   # (d, d)
        radius = np.max(np.abs(np.linalg.eigvals(self.A)))
        if radius > self.max_radius:                           # stabilize
            self.A *= self.max_radius / radius
        R = Z1 - Z0 @ self.A.T
        self.Q = np.cov(R.T) + 1e-6 * np.eye(d)
        return self

    # -------------------------------------------------------------- encode
    def encode(self, seq):
        """Frames ``(T, H, W)`` -> latent codes ``(T, d)``."""
        X = seq.reshape(seq.shape[0], -1) - self.mean_
        return X @ self.components_.T

    # -------------------------------------------------------------- predict
    def predict(self, z0, h):
        """h-step Gaussian predictive ``(mu, Sigma)`` from latent ``z0``."""
        mu = z0.copy()
        Ai = np.eye(self.d)
        Sigma = np.zeros((self.d, self.d))
        for _ in range(h):
            Sigma = self.A @ Sigma @ self.A.T + self.Q
            mu = self.A @ mu
        return mu, Sigma

    # ----------------------------------------------------- calibration excess
    def excess_stream(self, seq, horizons, B=8.0):
        """Non-overlapping calibration excess ``mahalanobis^2/d - 1`` per horizon.

        Returns ``by_time`` (list of ``{h: Z}``), matching ``atlas.simulator``'s
        interface so the same frontier code consumes it.
        """
        z = self.encode(seq)                            # (T, d)
        T = z.shape[0] - 1
        by_time = [dict() for _ in range(T + 1)]
        d = self.d
        for h in horizons:
            for s in range(0, len(z) - h, h):           # non-overlapping windows
                mu, Sigma = self.predict(z[s], h)
                diff = z[s + h] - mu
                Sigma = Sigma + 1e-6 * np.eye(d)
                mahal2 = float(diff @ np.linalg.solve(Sigma, diff))
                by_time[s + h][h] = float(np.clip(mahal2 / d - 1.0, -1.0, B))
        return by_time

    def raw_excess(self, seqs, horizons, B=None):
        """Per-horizon excess over a list of sequences (for eps calibration).

        Pass ``B`` to clip to ``[-1, B]`` so the calibration statistic matches what
        ``excess_stream`` feeds the e-process (real-video excess is heavy-tailed, so
        clipped and unclipped means differ substantially)."""
        acc = {h: [] for h in horizons}
        d = self.d
        for seq in seqs:
            z = self.encode(seq)
            for h in horizons:
                for s in range(0, len(z) - h, h):
                    mu, Sigma = self.predict(z[s], h)
                    diff = z[s + h] - mu
                    Sigma = Sigma + 1e-6 * np.eye(d)
                    e = diff @ np.linalg.solve(Sigma, diff) / d - 1.0
                    acc[h].append(np.clip(e, -1.0, B) if B is not None else e)
        return {h: np.array(v) for h, v in acc.items()}
