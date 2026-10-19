"""Learned neural world model, planner and ATLAS statistic for the gated-control study.

* :class:`NNWorldModel` -- an MLP that predicts the next *velocity* as a heteroskedastic
  Gaussian (mean and log-variance); positions integrate velocity with the known kinematics.
  It is trained on random-action transitions of the nominal system only.
* :func:`plan` -- random-shooting + cross-entropy planner through the model's mean
  prediction over a committed horizon ``L``.
* :func:`predictive_stat` -- the h-step ATLAS statistic. The model's h-step predictive for the
  *committed* action sequence is sampled by rolling the stochastic model forward; the
  standardised excess uses the sample covariance on its non-degenerate subspace (positions
  are deterministic given velocities at h=1, so the covariance is rank deficient there).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ctrl_env import DT, A_MAX, target, PointMass

torch.set_num_threads(1)


class NNWorldModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 4))

    def forward(self, s, a):
        out = self.net(torch.cat([s, a], dim=-1))
        return out[..., :2], out[..., 2:].clamp(-8.0, 1.0)      # d(velocity), log-var

    def step(self, s, a, sample=False):
        a = a.clamp(-A_MAX, A_MAX)
        dv, lv = self(s, a)
        v = s[..., 2:] + dv
        if sample:
            v = v + torch.exp(0.5 * lv) * torch.randn_like(v)
        p = s[..., :2] + DT * s[..., 2:]
        return torch.cat([p, v], dim=-1)


def collect(n_steps=20000, seed=0):
    """Random-action transitions of the nominal system."""
    env = PointMass(seed)
    S, A, S2 = [], [], []
    s = env.reset()
    for t in range(n_steps):
        if t % 200 == 0:
            s = env.reset()
            env.s = s.copy()
            env.s[:2] = 1.5 * env.rng.normal(size=2)
            env.s[2:] = 0.6 * env.rng.normal(size=2)
            s = env.s.copy()
        a = env.rng.uniform(-A_MAX, A_MAX, size=2)
        s2 = env.step(a)
        S.append(s); A.append(a); S2.append(s2)
        s = s2
    return (np.array(S, np.float32), np.array(A, np.float32), np.array(S2, np.float32))


def train(seed=0, epochs=60, n_steps=20000):
    torch.manual_seed(seed)
    S, A, S2 = collect(n_steps, seed)
    S, A, S2 = map(torch.tensor, (S, A, S2))
    wm = NNWorldModel()
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    n = len(S)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            dv, lv = wm(S[idx], A[idx])
            tgt = S2[idx, 2:] - S[idx, 2:]
            nll = 0.5 * (((tgt - dv) ** 2) * torch.exp(-lv) + lv).mean()
            opt.zero_grad(); nll.backward(); opt.step()
    wm.eval()
    return wm


def plan(wm, s0, t0, L, iters=25, lr=0.25, prev=None):
    """Gradient-based planning through the model's mean dynamics over horizon ``L``
    (cost: tracking at every step, terminal tracking weighted 3x, small effort penalty)."""
    tgt = torch.tensor(np.stack([target(t0 + i + 1) for i in range(L)]), dtype=torch.float32)
    s_init = torch.tensor(s0, dtype=torch.float32)
    u0 = np.zeros((L, 2), np.float32) if prev is None else prev.astype(np.float32)
    u = torch.tensor(u0, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    w = torch.ones(L); w[-1] = 3.0
    for _ in range(iters):
        s = s_init
        cost = 0.0
        a = torch.tanh(u) * A_MAX
        for i in range(L):
            s = wm.step(s, a[i])
            cost = cost + w[i] * ((s[:2] - tgt[i]) ** 2).sum() + 0.01 * (a[i] ** 2).sum()
        opt.zero_grad(); cost.backward(); opt.step()
    return (torch.tanh(u) * A_MAX).detach().numpy()


@torch.no_grad()
def predictive_stat(wm, s0, actions, y, n=200, rng_seed=0):
    """Standardised excess of the h-step predictive for the committed ``actions`` (h x 2)
    against the realized state ``y``. Returns ``(z, d_eff)``."""
    torch.manual_seed(rng_seed)
    s = torch.tensor(s0, dtype=torch.float32).expand(n, 4).clone()
    a = torch.tensor(np.asarray(actions), dtype=torch.float32)
    for i in range(len(actions)):
        s = wm.step(s, a[i].expand(n, 2), sample=True)
    x = s.numpy().astype(np.float64)
    mu = x.mean(0)
    cov = np.cov(x.T) + 1e-9 * np.eye(4)
    w, V = np.linalg.eigh(cov)
    keep = w > 1e-5
    d_eff = int(keep.sum())
    if d_eff == 0:
        return 0.0, 0
    r = V[:, keep].T @ (np.asarray(y, np.float64) - mu)
    chi2 = float(np.sum(r ** 2 / w[keep]))
    return chi2 / d_eff - 1.0, d_eff
