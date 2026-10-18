"""A closed-loop (policy-endogenous) latent world model with state-dependent noise.

Why a new simulator: ``atlas.simulator`` draws exogenous exploration actions, which
makes the visited-state distribution independent of the model under test. Here the
action is chosen by a planner that *optimises through the world model*, so the stream the
monitor sees is a function of the object being certified -- the endogeneity regime of the
paper -- and, because the aleatoric noise scale depends on the state, the planner's choice
of where to go changes the score distribution.

True dynamics      z' = A z + B a + sqrt(q(z)) eps,   eps ~ N(0, I_d),
                   q(z) = q0 * (1 + beta * s(z)),     s(z) = |z|^2 / (|z|^2 + d)  in [0, 1).
World model        N(A z + B a, q(z) I): the *faithful* model (it knows the noise law).
Drift (alt.)       from the changepoint on, the true update gains ``+ f * u``.

Planners (all certainty-equivalent one-step MPC *through the model*):
    ``exo``       random exploration, independent of the state (the exogenous baseline);
    ``low``       steer to ``z* = 0``                 (a low-noise region);
    ``high``      steer to ``|z*| = r_high`` along a fixed direction (a high-noise region)
                  -- the *evasive* planner: noise is large there, so a mean shift of fixed
                  size moves the standardised statistic least.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ClosedLoopWM"]


class ClosedLoopWM:
    def __init__(self, dim=4, seed=0, q0=0.1, beta=4.0, a_max=2.0, rho=0.05,
                 r_high=3.0, explore=0.05):
        rng = np.random.default_rng(seed)
        self.dim = dim
        M = rng.normal(size=(dim, dim))
        self.A = 0.9 * M / np.max(np.abs(np.linalg.eigvals(M)))
        Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        self.B = 0.8 * Q                          # fully actuated, so targets are reachable
        self.q0, self.beta = q0, beta
        self.a_max, self.rho, self.explore = a_max, rho, explore
        u = rng.normal(size=dim)
        self.u = u / np.linalg.norm(u)            # drift direction
        self.z_high = r_high * self.u             # high-noise target
        self._G = np.linalg.solve(self.B.T @ self.B + rho * np.eye(dim), self.B.T)
        self._rng = rng

    # noise scale at state z
    def q(self, z):
        n2 = float(z @ z)
        return self.q0 * (1.0 + self.beta * n2 / (n2 + self.dim))

    def act(self, z, policy, rng):
        if policy == "exo":
            return 0.3 * rng.normal(size=self.dim)
        target = np.zeros(self.dim) if policy == "low" else self.z_high
        # one-step certainty-equivalent MPC *through the model*
        a = self._G @ (target - self.A @ z)
        a = np.clip(a, -self.a_max, self.a_max)
        return a + self.explore * rng.normal(size=self.dim)

    def step(self, z, a, rng, force=0.0):
        """One true transition. Returns ``(z_next, raw, std, q)``."""
        d = self.dim
        mu = self.A @ z + self.B @ a
        qq = self.q(z)
        zn = mu + force + np.sqrt(qq) * rng.normal(size=d)
        r = float((zn - mu) @ (zn - mu))
        return zn, r / d, r / (qq * d) - 1.0, qq

    def run(self, T, policy="exo", cp=None, drift=0.0, seed=None, chooser=None):
        """Simulate ``T`` rounds. Returns the raw squared residual ``r`` (model-side
        residual ``|y - mu|^2 / d``) and the standardised statistic ``Z`` (both length T),
        plus the noise level path ``q``.

        ``chooser(t, z, std_history) -> policy name`` lets a *monitor-aware* adversary
        pick the planner each round from everything observed so far."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        d = self.dim
        z = np.zeros(d)
        raw = np.empty(T)
        std = np.empty(T)
        qs = np.empty(T)
        for t in range(T):
            pol = chooser(t, z, std[:t]) if chooser is not None else policy
            a = self.act(z, pol, rng)
            force = drift * self.u if (cp is not None and t >= cp) else 0.0
            z, raw[t], std[t], qs[t] = self.step(z, a, rng, force)
        return raw, std, qs
