"""A small nonlinear control task for the gated-planner study.

A damped point mass is steered along a moving target. State ``s = (x, y, vx, vy)``,
action ``a in [-1, 1]^2`` (force), time step ``dt``:

    p' = p + dt * v
    v' = v + dt * ( G a - c1 v - c2 |v| v ) + sqrt(dt) * sigma * eps,    eps ~ N(0, I).

The quadratic drag makes the dynamics nonlinear. Three mid-deployment faults change the
true dynamics while the learned world model keeps its pre-fault behaviour:

    ``gain``   actuator degradation: G -> 0.4 G;
    ``flip``   one axis is mis-wired: the x-action changes sign;
    ``wind``   a constant unmodelled force pushes along x.
"""

from __future__ import annotations

import numpy as np

DT = 0.1
C1, C2 = 0.6, 0.5
SIGMA = 0.15
A_MAX = 1.0
FAULTS = ("none", "gain", "flip", "wind")


def target(t):
    """Target position at time step ``t`` (circle of radius 1, period 8 s)."""
    w = 2 * np.pi / 80.0
    return np.array([np.cos(w * t), np.sin(w * t)])


class PointMass:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.s = np.zeros(4)

    def reset(self):
        self.s = np.concatenate([target(0) + 0.3 * self.rng.normal(size=2),
                                 0.1 * self.rng.normal(size=2)])
        return self.s.copy()

    @staticmethod
    def drift_dynamics(s, a, fault):
        """Deterministic part of the true transition under ``fault`` (numpy, one state)."""
        a = np.clip(a, -A_MAX, A_MAX)
        g = np.array([1.0, 1.0])
        wind = np.zeros(2)
        if fault == "gain":
            g = g * 0.4
        elif fault == "flip":
            g = np.array([-1.0, 1.0])
        elif fault == "wind":
            wind = np.array([0.8, 0.0])
        v = s[2:]
        dv = DT * (g * a - C1 * v - C2 * np.linalg.norm(v) * v + wind)
        return np.concatenate([s[:2] + DT * v, v + dv])

    def step(self, a, fault="none"):
        mean = self.drift_dynamics(self.s, np.asarray(a, float), fault)
        noise = np.concatenate([np.zeros(2), np.sqrt(DT) * SIGMA * self.rng.normal(size=2)])
        self.s = mean + noise
        return self.s.copy()


def stage_cost(s, a, t):
    return float(np.sum((s[:2] - target(t)) ** 2) + 0.01 * np.sum(np.asarray(a) ** 2))
