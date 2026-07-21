"""Published competitor drift/change detectors for head-to-head comparison with ATLAS.

All monitors consume the same per-round score stream ``Z_k`` (the one-step calibration
excess) and emit an alarm time. Unlike the toy thresholds one might reach for, these are
established algorithms from the streaming / sequential-analysis literature:

* ``CUSUM``        -- Page's cumulative-sum change detector (Page, Biometrika 1954).
* ``PageHinkley``  -- the Page--Hinkley sequential test (river implementation).
* ``ADWIN``        -- ADaptive WINdowing with Hoeffding-bounded cuts
  (Bifet & Gavald{\`a}, SDM 2007; river implementation).
* ``KSWIN``        -- Kolmogorov--Smirnov Windowing (Raab, Heusinger & Schleif,
  Neurocomputing 2020; river implementation).
* ``BOCPD``        -- Bayesian Online Changepoint Detection (Adams & MacKay 2007),
  Gaussian observation model with a constant hazard.
* ATLAS (``atlas.eprocess.EProcess``) -- the anytime-valid e-process.

Each detector exposes a scalar *sensitivity* knob that ``exp4_baselines.py`` calibrates on
null runs to a common target false-alarm rate at a nominal horizon; the point of the
experiment is that the calibrated competitors nonetheless inflate their false-alarm rate
under *longer* monitoring (the multiple-looks problem), whereas ATLAS controls it at every
round by construction (Ville).
"""

from __future__ import annotations

import numpy as np


class CUSUM:
    """Page's positive CUSUM change detector. Sensitivity knob: ``thresh``."""

    def __init__(self, ref_mean, slack, thresh):
        self.m0, self.slack, self.thresh = ref_mean, slack, thresh
        self.S = 0.0

    def step(self, z):
        self.S = max(0.0, self.S + (z - self.m0 - self.slack))
        return self.S >= self.thresh


class PageHinkley:
    """Page--Hinkley test (via ``river.drift.PageHinkley``). Sensitivity knob: ``thresh``."""

    def __init__(self, thresh, delta=0.005, min_instances=30):
        from river.drift import PageHinkley as _PH
        self._d = _PH(threshold=thresh, delta=delta, min_instances=min_instances)

    def step(self, z):
        self._d.update(float(z))
        return bool(self._d.drift_detected)


class ADWIN:
    """ADaptive WINdowing (via ``river.drift.ADWIN``). Sensitivity knob: ``delta``."""

    def __init__(self, delta):
        from river.drift import ADWIN as _ADWIN
        self._d = _ADWIN(delta=delta)

    def step(self, z):
        self._d.update(float(z))
        return bool(self._d.drift_detected)


class KSWIN:
    """Kolmogorov--Smirnov Windowing (via ``river.drift.KSWIN``). Sensitivity knob: ``alpha``."""

    def __init__(self, alpha, window_size=100, stat_size=30):
        from river.drift import KSWIN as _KSWIN
        self._d = _KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size,
                         seed=0)

    def step(self, z):
        self._d.update(float(z))
        return bool(self._d.drift_detected)


class BOCPD:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Gaussian observation model with a Normal--inverse-gamma conjugate prior and a
    constant hazard ``1/hazard``. Alarms when the posterior probability mass on short
    run lengths (a recent changepoint) exceeds ``thresh`` -- the sensitivity knob.
    """

    def __init__(self, thresh, hazard=250.0, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                 rmax=300, burnin=30):
        self.thresh = thresh                            # MAP run-length reset threshold
        self.burnin = burnin
        self.t = 0
        self.h = 1.0 / hazard
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.rmax = rmax
        self.R = np.array([1.0])                    # run-length posterior
        self.mu = np.array([mu0]); self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0]); self.beta = np.array([beta0])

    def _pred(self, z):
        # Student-t predictive probability of z under each run-length hypothesis
        from scipy.special import gammaln
        df = 2 * self.alpha
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        x = (z - self.mu) ** 2 / scale2
        logp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale2)
                - (df + 1) / 2 * np.log1p(x / df))
        return np.exp(logp)

    def step(self, z):
        pred = self._pred(z)
        growth = self.R * pred * (1 - self.h)
        cp = np.sum(self.R * pred * self.h)
        newR = np.concatenate([[cp], growth])
        newR /= newR.sum() + 1e-300
        # conjugate update of the sufficient statistics; prepend the prior for r=0
        mu = np.concatenate([[self.mu0], self.mu + (z - self.mu) / (self.kappa + 1)])
        kappa = np.concatenate([[self.kappa0], self.kappa + 1])
        alpha = np.concatenate([[self.alpha0], self.alpha + 0.5])
        beta = np.concatenate([[self.beta0],
                               self.beta + self.kappa * (z - self.mu) ** 2
                               / (2 * (self.kappa + 1))])
        if len(newR) > self.rmax:                   # truncate for O(rmax) per step
            newR = newR[:self.rmax]; mu = mu[:self.rmax]; kappa = kappa[:self.rmax]
            alpha = alpha[:self.rmax]; beta = beta[:self.rmax]
            newR /= newR.sum() + 1e-300
        self.R, self.mu, self.kappa, self.alpha, self.beta = newR, mu, kappa, alpha, beta
        self.t += 1
        # textbook BOCPD detection: after burn-in the MAP run length grows with t;
        # a changepoint makes it collapse. Alarm when it drops below ``thresh``.
        r_map = int(np.argmax(self.R))
        return self.t > self.burnin and r_map <= self.thresh


class ConformalMartingale:
    """Conformal test (betting) martingale on the online score stream -- the modern
    conformal-monitoring family (e.g.\\ Vovk's test martingales; the 2024--2026 online
    conformal / drift-detection line of Gibbs & Cand{\\`e}s, Opoku & Banahene, Leong).

    Each score is mapped to a right-tailed conformal p-value against a null calibration
    set; under exchangeability the p-values are uniform, and a capital process bets
    against uniformity with a predictable GRAPA plug-in. The wealth is a nonnegative
    martingale under the null, so alarming at ``1/alpha`` is anytime-valid (Ville) with
    no threshold tuning -- it does not inflate under longer monitoring.
    """

    def __init__(self, ref, alpha=0.05):
        self.ref = np.sort(np.asarray(ref, dtype=float))
        self.alpha = float(alpha)
        self.logM = 0.0
        self._s = 0.0; self._s2 = 0.0; self._k = 0
        self._latched = False

    def step(self, z):
        n = len(self.ref)
        n_ge = n - int(np.searchsorted(self.ref, z, side="left"))   # #{ref >= z}
        p = float(np.clip((1.0 + n_ge) / (n + 1.0), 1e-6, 1 - 1e-6))
        d = 0.5 - p                                   # >0 when p small (score in tail)
        lam = np.clip(self._s / (self._s2 + 1e-12), 0.0, 2.0) if self._k else 0.0
        self.logM += np.log(max(1.0 + lam * d, 1e-300))
        self._s += d; self._s2 += d * d; self._k += 1
        if self.logM >= np.log(1.0 / self.alpha):
            self._latched = True
        return self._latched


def run_monitor(monitor, stream, cp=None):
    """Return the first alarm index over ``stream`` (or ``None``). If ``cp`` is given,
    return the post-change delay (alarm - cp), ignoring pre-cp alarms as false."""
    for t, z in enumerate(stream):
        if monitor.step(z):
            if cp is None:
                return t
            if t >= cp:
                return t - cp
    return None
