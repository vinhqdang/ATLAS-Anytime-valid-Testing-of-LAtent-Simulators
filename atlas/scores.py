"""Proper scoring rules for ATLAS nonconformity scores.

All scores are *negatively oriented* (smaller = more faithful), matching
``docs/01_problem_setup.md`` §3. Each takes a predictive object and a realized
observation and returns a scalar. The energy / Mahalanobis scores make ATLAS
representation-agnostic: they score a predicted *embedding* against the realized
embedding, so JEPA-style models that never decode are covered.

Boundedness (assumption A1) is obtained downstream by clipping to ``[0, B]`` via
:func:`clip_score`; the raw rules are returned unclipped here so the caller
controls the normalization constant ``B``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "gaussian_nll",
    "crps_gaussian",
    "mahalanobis_score",
    "energy_score",
    "clip_score",
]


def gaussian_nll(mu, sigma, y):
    """Negative log-likelihood of ``y`` under ``N(mu, sigma^2)`` (per-dim summed).

    For the log score, the induced divergence ``d_S`` is the KL divergence, so the
    score-to-metric modulus (A3) is Pinsker: ``TV <= sqrt(KL / 2)``.
    """
    mu, sigma, y = np.atleast_1d(mu), np.atleast_1d(sigma), np.atleast_1d(y)
    var = sigma ** 2
    return float(np.sum(0.5 * ((y - mu) ** 2 / var + np.log(2 * np.pi * var))))


def crps_gaussian(mu, sigma, y):
    """Closed-form CRPS for a Gaussian predictive ``N(mu, sigma^2)`` (per-dim summed).

    CRPS(N(mu,s), y) = s * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z=(y-mu)/s.
    A strictly proper rule whose divergence controls an L2/Wasserstein-type metric.
    """
    mu, sigma, y = np.atleast_1d(mu), np.atleast_1d(sigma), np.atleast_1d(y)
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.sum(crps))


def mahalanobis_score(mu, cov, y):
    """Gaussian energy/Mahalanobis score: 0.5 (y-mu)' C^-1 (y-mu) + 0.5 logdet(2 pi C).

    This is the multivariate Gaussian log score. Its divergence controls a squared
    Wasserstein-2 distance between the predictive and the true conditional, giving
    the modulus used in Theorem 2 for latent (JEPA/Dreamer) predictors.
    """
    mu, y = np.atleast_1d(mu), np.atleast_1d(y)
    cov = np.atleast_2d(cov)
    d = mu.shape[0]
    cov = cov + 1e-9 * np.eye(d)
    diff = y - mu
    sign, logdet = np.linalg.slogdet(2 * np.pi * cov)
    quad = diff @ np.linalg.solve(cov, diff)
    return float(0.5 * quad + 0.5 * logdet)


def energy_score(samples, y, max_pairs=256, rng=None):
    """Sample-based energy score for an ensemble predictive.

    ES(P, y) = E||X - y|| - 0.5 E||X - X'||,   X, X' ~ P i.i.d.

    ``samples`` is ``(n, d)``. Strictly proper for the Euclidean norm; its
    divergence is an integral-probability / MMD-type metric (energy distance),
    the modulus (A3) for pixel/video predictors.
    """
    samples = np.atleast_2d(samples)
    y = np.atleast_1d(y)
    n = samples.shape[0]
    term1 = np.mean(np.linalg.norm(samples - y, axis=1))
    if n <= 1:
        return float(term1)
    # subsample pairs for the O(n^2) term when the ensemble is large
    if n * n > max_pairs:
        rng = np.random.default_rng() if rng is None else rng
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        term2 = np.mean(np.linalg.norm(samples[i] - samples[j], axis=1))
    else:
        diffs = samples[:, None, :] - samples[None, :, :]
        term2 = np.mean(np.linalg.norm(diffs, axis=2))
    return float(term1 - 0.5 * term2)


def clip_score(s, B):
    """Clip a raw score into ``[0, B]`` to enforce boundedness (assumption A1)."""
    return float(np.clip(s, 0.0, B))
