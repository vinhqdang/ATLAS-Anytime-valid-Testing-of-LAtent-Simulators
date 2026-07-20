"""E-processes for ATLAS: testing-by-betting on world-model faithfulness.

Implements, verbatim from ``docs/02_construction.md``:

- :class:`EProcess`  — one-sided test supermartingale for the faithfulness null
  ``H0: E[Z_k | F_{k-1}] <= eps`` via the betting e-variable
  ``e_k = 1 + lambda_k (Z_k - eps)`` with a predictable GRAPA plug-in bet.
  Ville's inequality gives the Theorem-1 guarantee.
- :class:`BettingCS` — the two-sided confidence sequence dual (Waudby-Smith &
  Ramdas style) that certifies an anytime upper bound on the one-step gap, feeding
  the anytime-valid simulation lemma (Theorem 2).
- :class:`EDetector` — a Shiryaev-Roberts-style e-detector for the change-detection
  setting of Theorem 3 (order-optimal delay).

Everything is *conditional* (one bet at a time): validity needs only the
conditional-mean null, never i.i.d./exchangeability, which is what keeps ATLAS
sound on the endogenous, policy-dependent stream (setup §5).
"""

from __future__ import annotations

import numpy as np

__all__ = ["EProcess", "BettingCS", "EDetector"]


def _grapa_bet(sum_d, sum_d2, count, lam_max):
    """Predictable GRAPA plug-in for the log-optimal bet.

    argmax_lambda E[log(1 + lambda * d)] ~= E[d] / E[d^2] for small edges.
    Uses only statistics from strictly past rounds (predictable), truncated to
    ``[0, lam_max]`` so the e-variable stays nonnegative and valid.
    """
    if count == 0:
        return 0.0
    mean_d = sum_d / count
    mean_d2 = sum_d2 / count
    lam = mean_d / (mean_d2 + 1e-12)
    return float(np.clip(lam, 0.0, lam_max))


def _logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


class EProcess:
    """One-sided testing-by-betting e-process for ``H0: mean(Z) <= eps``.

    Uses **mixture betting**: the wealth is the average, over a fixed grid of
    betting fractions ``lambda_j in (0, lam_max]``, of the single-bet capital
    processes ``prod_k (1 + lambda_j (Z_k - eps))``. Each component is a test
    supermartingale under ``H0`` (``E[1 + lambda_j (Z_k - eps) | F_{k-1}] <= 1``),
    and an average of e-processes is an e-process, so the mixture is valid (Ville
    applies). Mixture betting is parameter-free, needs no edge estimate, does not
    deplete wealth under the null, and reacts fast to an unknown post-change margin
    — exactly the "mixture/GRO" betting of ``docs/02_construction.md`` §2. Pass
    ``mixture=False`` for the single-``lambda`` GRAPA plug-in instead.

    Parameters
    ----------
    eps : float
        Faithfulness tolerance ``eps_h`` for this horizon.
    z_lo, z_hi : float
        Known bounds ``Z_k in [z_lo, z_hi]`` (assumption A1). The bet is capped at
        ``lam_max = 1/(z_hi - z_lo)``, guaranteeing every component ``e_k in [0, 2]``
        and hence a valid nonnegative supermartingale.
    alpha : float
        Test level; ``rejected`` once wealth crosses ``1/alpha``.
    mixture : bool
        Mixture betting (default) vs single-lambda GRAPA plug-in.
    n_grid : int
        Number of mixture components.
    """

    def __init__(self, eps, z_lo, z_hi, alpha=0.05, mixture=True, n_grid=20,
                 lam_scale=1.0):
        assert z_hi > z_lo, "need z_hi > z_lo"
        self.eps = float(eps)
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.alpha = float(alpha)
        self.lam_max = lam_scale / (z_hi - z_lo)
        self.mixture = mixture
        self.k = 0
        self.history = []     # log-wealth trace
        if mixture:
            self.lams = np.linspace(self.lam_max / n_grid, self.lam_max, n_grid)
            self.comp_logw = np.zeros(n_grid)   # per-component log capital
            self.log_prior = -np.log(n_grid)
        else:
            self.lam_max_single = self.lam_max
            self.log_wealth_single = 0.0
            self._sum_d = 0.0
            self._sum_d2 = 0.0

    def update(self, z):
        """Process one resolved round with score advantage ``z``."""
        d = z - self.eps
        if self.mixture:
            self.comp_logw += np.log(np.maximum(1.0 + self.lams * d, 1e-300))
        else:
            lam = _grapa_bet(self._sum_d, self._sum_d2, self.k, self.lam_max_single)
            self.log_wealth_single += np.log(max(1.0 + lam * d, 1e-300))
            self._sum_d += d
            self._sum_d2 += d * d
        self.k += 1
        self.history.append(self.log_wealth)
        return self.log_wealth

    @property
    def log_wealth(self):
        if self.mixture:
            return float(_logsumexp(self.comp_logw + self.log_prior))
        return float(self.log_wealth_single)

    @property
    def wealth(self):
        return float(np.exp(self.log_wealth))

    @property
    def rejected(self):
        """True once the horizon is revoked: wealth >= 1/alpha (Ville)."""
        return self.log_wealth >= np.log(1.0 / self.alpha)


class BettingCS:
    """Two-sided anytime-valid confidence sequence for the mean of a bounded stream.

    Maintains, over a grid of candidate means ``m in [z_lo, z_hi]``, two capital
    processes per grid point:

        K+(m) = prod (1 + lam+ (Z_k - m))    grows when true mean > m
        K-(m) = prod (1 + lam- (m - Z_k))    grows when true mean < m

    The confidence sequence is ``{m : K+(m) < 1/alpha and K-(m) < 1/alpha}``; its
    upper edge ``U`` is the certified anytime upper bound on the one-step gap used
    by the simulation lemma (Theorem 2). Vectorized over the grid for O(grid) per
    step. Each K is a test supermartingale, so ``P(exists t: mean not in CS_t) <=
    2*alpha`` (union of the two one-sided Ville bounds).
    """

    def __init__(self, z_lo, z_hi, alpha=0.05, grid=400):
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.alpha = float(alpha)
        self.m = np.linspace(z_lo, z_hi, grid)
        self.lam_max = 1.0 / (z_hi - z_lo)
        self.logKp = np.zeros(grid)
        self.logKm = np.zeros(grid)
        # predictable running stats of (Z - m) and (m - Z) per grid point
        self._s_p = np.zeros(grid)
        self._s2_p = np.zeros(grid)
        self._s_m = np.zeros(grid)
        self._s2_m = np.zeros(grid)
        self.k = 0

    def _bets(self):
        if self.k == 0:
            return np.zeros_like(self.m), np.zeros_like(self.m)
        mp = self._s_p / self.k
        vp = self._s2_p / self.k
        lam_p = np.clip(mp / (vp + 1e-12), 0.0, self.lam_max)
        mm = self._s_m / self.k
        vm = self._s2_m / self.k
        lam_m = np.clip(mm / (vm + 1e-12), 0.0, self.lam_max)
        return lam_p, lam_m

    def update(self, z):
        lam_p, lam_m = self._bets()
        dp = z - self.m           # positive when z > m
        dm = self.m - z           # positive when z < m
        self.logKp += np.log(np.maximum(1.0 + lam_p * dp, 1e-300))
        self.logKm += np.log(np.maximum(1.0 + lam_m * dm, 1e-300))
        self._s_p += dp
        self._s2_p += dp * dp
        self._s_m += dm
        self._s2_m += dm * dm
        self.k += 1

    def interval(self):
        """Return ``(L, U)`` = current confidence sequence for the mean gap."""
        thresh = np.log(1.0 / self.alpha)
        keep = (self.logKp < thresh) & (self.logKm < thresh)
        if not np.any(keep):
            return (self.z_hi, self.z_lo)  # empty -> degenerate (shouldn't happen early)
        ms = self.m[keep]
        return (float(ms.min()), float(ms.max()))

    @property
    def upper(self):
        """Certified anytime upper bound ``hat_delta_1`` on the one-step gap (Thm 2)."""
        return self.interval()[1]


class EDetector:
    """Shiryaev-Roberts-style e-detector for faithfulness *drift* (Theorem 3).

    Recursion ``R_t = (1 + R_{t-1}) * e_t`` accumulates evidence over every
    candidate changepoint, so statistical strength is not depleted before the true
    change ``tau``. Alarm when ``R_t >= 1/alpha``. Under the null each ``e_t`` is a
    conditional e-variable, so ``R_t - t`` is a supermartingale and the average run
    length satisfies ``E_null[T] >= 1/alpha`` (false-alarm control); after a change
    of margin ``Delta`` the detection delay is ``~ log(1/alpha)/D*`` (GRO growth).
    """

    def __init__(self, eps, z_lo, z_hi, alpha=0.05, lam_scale=1.0):
        self.eps = float(eps)
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.alpha = float(alpha)
        self.lam_max = lam_scale / (z_hi - z_lo)
        self.R = 0.0
        self.k = 0
        self._sum_d = 0.0
        self._sum_d2 = 0.0
        self.history = []

    def update(self, z):
        lam = _grapa_bet(self._sum_d, self._sum_d2, self.k, self.lam_max)
        d = z - self.eps
        e = max(1.0 + lam * d, 1e-300)
        self.R = (1.0 + self.R) * e
        self._sum_d += d
        self._sum_d2 += d * d
        self.k += 1
        self.history.append(self.R)
        return self.R

    @property
    def alarm(self):
        return self.R >= 1.0 / self.alpha
