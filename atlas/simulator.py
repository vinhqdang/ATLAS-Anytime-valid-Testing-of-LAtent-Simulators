"""A controllable latent world model for exercising ATLAS end-to-end.

We use a linear-Gaussian latent system because it gives us *ground-truth* control
over faithfulness (we know the true dynamics and can compute the exact predictive),
lets us inject a dynamics shift at a known changepoint, and keeps everything
CPU-cheap and fully reproducible. This stands in for a Dreamer/JEPA latent
simulator; the ATLAS machinery is identical for a learned one.

True dynamics (regime-dependent):
    z_{t+1} = A z_t + B a_t + w_t,     w_t ~ N(0, Q).

At the changepoint an unmodelled *dynamics drift* switches on: the true update
gains a constant force term ``+ drift_mag * u`` (a fixed direction ``u``), modelling
a mid-deployment mass/friction/actuator perturbation. The world model ``M`` keeps
its *pre-change* parameters, so it is faithful before the change and progressively
unfaithful after — and unfaithful *sooner at long horizons* (the drift compounds
over the rollout), which is exactly the staircase the frontier should reveal. A
tight-but-wrong model predictive is heavily penalised by the log score, while the
wide climatology reference tolerates the shift — so the score advantage ``Z`` turns
sharply positive after the change. ``drift_mag`` doubles as the detectable margin
``Delta`` for the detection-delay study.

The behaviour policy draws exogenous exploration actions ``a_t ~ N(0, sigma_a^2 I)``.
The resulting state stream is still non-i.i.d. and adaptive (a controlled Markov
chain; scores at different rounds are dependent and the betting is sequential),
which is already beyond what exchangeability-based conformal prediction handles.
Exogenous actions make the action-conditioned predictive *exactly* calibrated under
the null, so the Type-I / no-false-revocation study has a genuine null to test. The
fully endogenous policy-in-the-loop regime (a planner that acts through the model
under test) is the target of the real-world-model experiments; the conditional
e-process construction is valid there for the same reason — it never invokes
exchangeability.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import scores

__all__ = ["Simulator"]


class Simulator:
    def __init__(self, dim=4, action_dim=2, seed=0,
                 process_noise=0.1, model_bias=0.0, B=6.0):
        self.dim = dim
        self.action_dim = action_dim
        self.B = B              # score-advantage clip bound (assumption A1)
        rng = np.random.default_rng(seed)

        # --- true dynamics (pre-change), a stable random system -------------
        M = rng.normal(size=(dim, dim))
        # make spectrally stable: scale so spectral radius ~ 0.9
        eig = np.max(np.abs(np.linalg.eigvals(M)))
        self.A = 0.9 * M / eig
        self.Bmat = rng.normal(size=(dim, action_dim)) * 0.5
        self.Q = process_noise * np.eye(dim)

        # fixed unit drift direction switched on post-change (unmodelled force)
        u = rng.normal(size=dim)
        self.drift_dir = u / np.linalg.norm(u)

        # --- world model M: keeps pre-change dynamics (+ optional bias) ------
        self.A_model = self.A + model_bias * rng.normal(size=(dim, dim))
        self.B_model = self.Bmat.copy()
        self.Q_model = self.Q.copy()

        # --- exogenous exploration policy -----------------------------------
        self.sigma_a = 0.3      # exploration action scale

        # --- stationary state law (for initial-state sampling / climatology) -
        # z_{t+1} = A z_t + B a_t + w,  a_t ~ N(0, sigma_a^2 I) exogenous.
        action_cov = self.sigma_a ** 2 * (self.Bmat @ self.Bmat.T)
        self.mu0 = np.zeros(dim)
        self.Sigma0 = solve_discrete_lyapunov(self.A, self.Q + action_cov)

        self._rng = rng

    # ------------------------------------------------------------------ roll
    def rollout(self, T, changepoint=None, drift_mag=1.0):
        """Simulate a trajectory of ``T`` steps. Returns ``(z[T+1], a[T])``.

        From step ``changepoint`` on, an unmodelled force ``drift_mag * drift_dir``
        is added to the true update (None = never). ``drift_mag`` is the detectable
        margin ``Delta``.
        """
        rng = self._rng
        z = np.zeros((T + 1, self.dim))
        # Q = process_noise * I, so noise sampling is a cheap vectorized draw
        sig_w = np.sqrt(self.Q[0, 0])
        w_all = sig_w * rng.normal(size=(T, self.dim))
        a = self.sigma_a * rng.normal(size=(T, self.action_dim))
        L0 = np.linalg.cholesky(self.Sigma0)
        z[0] = L0 @ rng.normal(size=self.dim)
        Ba = a @ self.Bmat.T
        for t in range(T):
            force = (drift_mag * self.drift_dir
                     if (changepoint is not None and t >= changepoint) else 0.0)
            z[t + 1] = self.A @ z[t] + Ba[t] + w_all[t] + force
        return z, a

    # -------------------------------------------------------------- predict
    def model_predict(self, z_t, actions):
        """Model's action-conditioned h-step predictive N(mu, Sigma), h=len(actions)."""
        mu = z_t.copy()
        Sigma = np.zeros((self.dim, self.dim))
        for k in range(len(actions)):
            mu = self.A_model @ mu + self.B_model @ actions[k]
            Sigma = self.A_model @ Sigma @ self.A_model.T + self.Q_model
        return mu, Sigma

    def reference_predict(self, h):
        """Climatology reference predictive (horizon-independent marginal)."""
        return self.mu0, self.Sigma0

    # -------------------------------------------------- calibration statistic
    def _excess(self, z, a, s, h):
        """Calibration excess ``mahalanobis^2 / d - 1`` for the window issued at
        ``s`` and resolving at ``s + h``."""
        d = self.dim
        y = z[s + h]
        mu_m, Sig_m = self.model_predict(z[s], a[s:s + h])
        diff = y - mu_m
        Sig_m = Sig_m + 1e-9 * np.eye(d)
        mahal2 = float(diff @ np.linalg.solve(Sig_m, diff))
        return mahal2 / d - 1.0

    def resolved_stream(self, z, a, horizons, changepoint=None):
        """Per-resolution-time calibration excess on *non-overlapping* windows.

        Reference-free, PIT/calibration instantiation of ``docs/02_construction.md``
        §3. The excess ``Z = mahalanobis^2 / d - 1`` is chi-square-based: under exact
        ``eps``-faithfulness the h-step predictive is calibrated, so ``mahalanobis^2
        ~ chi^2_d`` and ``E[Z] = 0``, giving the null ``E[Z] <= eps``.

        Crucially, horizon ``h`` is only updated on **non-overlapping** windows
        (issue times ``0, h, 2h, ...``; resolution times that are multiples of
        ``h``). Overlapping windows would make the residuals a moving-average
        process with ``E[Z_k | F_{k-1}] != 0`` — the recursive-multi-step failure
        of naive multi-step CP. Non-overlap restores i.i.d. residuals within a
        horizon and hence exact e-process validity. The (real) cost is ``h``-fold
        fewer updates at horizon ``h``: long horizons are genuinely harder to certify.

        ``Z`` is clipped to ``[-1, B]`` (A1; lower bound exact since mahalanobis^2 >= 0).

        Returns ``by_time`` (list of length ``T+1`` of ``{h: Z}`` for horizons
        resolving at that time) and ``truth`` (diagnostics).
        """
        T = a.shape[0]
        by_time = [dict() for _ in range(T + 1)]
        raw = {h: [] for h in horizons}
        for h in horizons:
            for s in range(0, T - h + 1, h):          # non-overlapping issue times
                t = s + h                              # resolution time
                excess = self._excess(z, a, s, h)
                raw[h].append(excess)
                by_time[t][h] = float(np.clip(excess, -1.0, self.B))
        truth = {
            "changepoint": changepoint,
            "raw_excess": {h: np.asarray(v) for h, v in raw.items()},
        }
        return by_time, truth
