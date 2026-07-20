"""Fast unit tests for ATLAS core: e-variable validity and CS coverage.

Run: ``python -m pytest tests -q``  (or ``python tests/test_validity.py``)
"""

from __future__ import annotations

import numpy as np

from atlas.eprocess import EProcess, BettingCS
from atlas.scores import crps_gaussian, gaussian_nll, mahalanobis_score


def test_e_variable_is_conditional_supermartingale():
    """Under the null (mean-zero bounded stream), E[e_k | past] <= 1: the wealth
    should not systematically grow. Averaged over many null streams, the mean
    final wealth stays ~1 (a valid e-process has E[W] <= 1)."""
    rng = np.random.default_rng(0)
    finals = []
    for _ in range(400):
        ep = EProcess(eps=0.0, z_lo=-1.0, z_hi=6.0, alpha=0.05)
        # null: chi^2_4/4 - 1 has mean exactly 0, bounded below by -1
        z = rng.chisquare(4, size=300) / 4.0 - 1.0
        for zi in z:
            ep.update(min(zi, 6.0))
        finals.append(ep.wealth)
    assert np.mean(finals) <= 1.25, np.mean(finals)   # E[W] <= 1 up to MC noise


def test_ville_false_alarm_rate():
    """Empirical P(ever reject | null) <= alpha (Ville)."""
    rng = np.random.default_rng(1)
    alpha = 0.1
    rejects = 0
    N = 500
    for _ in range(N):
        ep = EProcess(eps=0.0, z_lo=-1.0, z_hi=6.0, alpha=alpha)
        rej = False
        for _ in range(300):
            zi = rng.chisquare(4) / 4.0 - 1.0
            ep.update(min(zi, 6.0))
            if ep.rejected:
                rej = True
                break
        rejects += rej
    assert rejects / N <= alpha + 0.03, rejects / N   # <= alpha (+ MC slack)


def test_power_under_alternative():
    """Under a clear alternative (positive mean), the e-process rejects."""
    rng = np.random.default_rng(2)
    ep = EProcess(eps=0.0, z_lo=-1.0, z_hi=6.0, alpha=0.05)
    for _ in range(200):
        zi = rng.chisquare(4, ) / 4.0 - 1.0 + 1.5   # shifted up
        ep.update(min(zi, 6.0))
    assert ep.rejected


def test_confidence_sequence_covers_mean():
    """The BettingCS should contain the true mean with high probability, and its
    upper edge should be a valid upper bound."""
    rng = np.random.default_rng(3)
    covered = 0
    N = 200
    true_mean = 0.0
    for _ in range(N):
        cs = BettingCS(z_lo=-1.0, z_hi=6.0, alpha=0.1)
        for _ in range(400):
            zi = rng.chisquare(4) / 4.0 - 1.0
            cs.update(min(zi, 6.0))
        lo, hi = cs.interval()
        if lo <= true_mean <= hi:
            covered += 1
    assert covered / N >= 0.85, covered / N          # >= 1 - 2*alpha


def test_scores_are_proper_ordering():
    """A well-specified predictive scores better (lower) than a misspecified one."""
    y = np.array([0.3])
    assert gaussian_nll([0.3], [1.0], y) < gaussian_nll([3.0], [1.0], y)
    assert crps_gaussian([0.3], [1.0], y) < crps_gaussian([3.0], [1.0], y)
    good = mahalanobis_score(np.array([0.3]), np.array([[1.0]]), y)
    bad = mahalanobis_score(np.array([3.0]), np.array([[1.0]]), y)
    assert good < bad


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
