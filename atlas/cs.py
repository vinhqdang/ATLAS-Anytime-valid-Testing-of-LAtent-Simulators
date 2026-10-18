"""Confidence sequences for the running average of *conditional* means.

The one-step faithfulness gap is not constant over a deployment: the quantity the
simulation lemma needs is the running average

    m_K = (1/K) sum_{k<=K} mu_k,        mu_k = E[Z_k | F_{k-1}],

of the conditional means of the score statistic. A betting confidence sequence for a
*constant* mean (``atlas.eprocess.BettingCS``) does not certify this object when the
conditional means vary; the construction below does, for arbitrary adapted dependence
and arbitrary time variation of ``mu_k``.

Construction (empirical-Bernstein supermartingale, Fan-Grama-Liu 2015; Waudby-Smith &
Ramdas 2024). Rescale ``Z`` to ``[0, 1]``. For a predictable guess ``Zhat_k in [0,1]`` and
a fixed ``lam in [0, 1)``,

    M_K(lam) = exp( lam * sum_k (mu_k - Z_k) - psi(lam) * sum_k (Z_k - Zhat_k)^2 ),
    psi(lam) = -log(1 - lam) - lam,

is a nonnegative supermartingale (each factor has conditional expectation <= 1 for *any*
value of ``mu_k``). A finite mixture over a grid ``lam_j`` with prior ``nu_j`` is again a
supermartingale, so by Ville's inequality, with probability >= 1 - alpha, simultaneously
for all K,

    m_K <= min_j [ Zbar_K + ( psi(lam_j) V_K + log(1/(alpha nu_j)) ) / (lam_j K) ],
    V_K = sum_k (Z_k - Zhat_k)^2.

Because ``M_K(lam_j) <= mixture / nu_j``, the event ``nu_j M_K(lam_j) >= 1/alpha`` for
*some* j and K is contained in ``{sup_K mixture >= 1/alpha}``; hence the displayed bound
holds simultaneously for every ``j`` and ``K`` and the min over the grid is free.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AvgCondMeanCS", "WindowedCS"]


def _psi(lam):
    return -np.log1p(-lam) - lam


class AvgCondMeanCS:
    """Anytime-valid upper (and lower) confidence sequence for ``m_K``.

    Parameters
    ----------
    z_lo, z_hi : float
        Known bounds of the (possibly clipped) statistic.
    alpha : float
        Level; the one-sided bounds each hold uniformly in ``K`` w.p. ``>= 1-alpha``.
    n_grid : int
        Size of the geometric grid of ``lam`` values (uniform prior).
    """

    def __init__(self, z_lo, z_hi, alpha=0.05, n_grid=24):
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.w = self.z_hi - self.z_lo
        self.alpha = float(alpha)
        self.lams = np.geomspace(0.005, 0.95, n_grid)
        self.log_nu = -np.log(n_grid)
        self.k = 0
        self._sum = 0.0            # sum of rescaled Z
        self._vu = 0.0             # sum (Z - Zhat)^2 (rescaled), for the upper side
        self._guess = 0.5

    # -- rescale to [0,1]
    def _r(self, z):
        return (z - self.z_lo) / self.w

    def update(self, z):
        x = float(np.clip(self._r(z), 0.0, 1.0))
        d = x - self._guess
        self._vu += d * d
        self._sum += x
        self.k += 1
        # predictable running-mean guess for the next round (clipped away from 0/1)
        self._guess = float(np.clip((0.5 + self._sum) / (self.k + 1.0), 0.0, 1.0))

    def _bound(self):
        k = self.k
        xbar = self._sum / k
        lam = self.lams
        up = xbar + (_psi(lam) * self._vu + np.log(1.0 / self.alpha) - self.log_nu) \
            / (lam * k)
        return up

    @property
    def upper(self):
        """Certified anytime upper bound on ``m_K`` (in the original ``Z`` units)."""
        if self.k == 0:
            return self.z_hi
        up = float(np.min(self._bound()))
        return self.z_lo + self.w * min(up, 1.0)

    @property
    def mean(self):
        return self.z_lo + self.w * (self._sum / max(self.k, 1))


class WindowedCS:
    """Upper confidence sequence for the average conditional mean over a recent window,
    valid uniformly over time.

    The CS is restarted on a grid of block starts ``s_i = i * block`` (``i = 0, 1, ...``)
    and start ``s_i`` is given level ``alpha_i = alpha / ((i+1)(i+2))``; since
    ``sum_i alpha_i = alpha`` the family is valid simultaneously over all starts and all
    end times (union bound + Ville per start). At time ``t`` we report the bound for the
    window ``(s_i, t]`` with the *latest* start ``s_i <= t - window``, so the window
    length lies in ``[window, window + block)``. The cost relative to the full-history CS
    is the larger ``log(1/alpha_i) ~ 2 log i`` term; with a coarse block it is mild.
    """

    def __init__(self, z_lo, z_hi, window, block=None, alpha=0.05):
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.window = int(window)
        self.block = int(block) if block else max(self.window // 2, 1)
        self.alpha = float(alpha)
        self.buf = []
        self.t = 0
        self._cache = None             # (start index, CS) reused until the start moves

    def update(self, z):
        self.buf.append(float(z))
        self.t += 1
        if self._cache is not None and self._cache[0] == self._start():
            self._cache[1].update(z)

    def _start(self):
        return max((self.t - self.window) // self.block, 0) * self.block

    @property
    def upper(self):
        if self.t == 0:
            return self.z_hi
        s = self._start()
        if self._cache is None or self._cache[0] != s:
            i = s // self.block
            cs = AvgCondMeanCS(self.z_lo, self.z_hi,
                               alpha=self.alpha / ((i + 1.0) * (i + 2.0)))
            for z in self.buf[s:]:
                cs.update(z)
            self._cache = (s, cs)
        return self._cache[1].upper
