"""Experiment 7 -- ablations and robustness (horizon 1, ground-truth simulator).

  A. bet strategy      mixture over a lambda grid vs fixed lambda vs GRAPA plug-in
  B. tolerance eps     a world model with a small *true* gap g: revocation rate as eps/g varies
  C. drift shape       abrupt step vs linear ramp vs transient pulse
  D. score statistic   standardised calibration excess vs log-score advantage over a
                       climatology reference
  E. noise tails       Gaussian vs heavy-tailed (Student-t, 5 d.o.f., variance-matched) truth

Every cell reports the false-revocation rate under the null (400 runs) and the mean delay /
detection rate under the alternative (200 runs). Run: ``python -m experiments.exp7_ablations``
"""

from __future__ import annotations

import json
import os

import numpy as np

from atlas.simulator import Simulator
from atlas.eprocess import EProcess
from atlas.fastsim import (excess_all_issue_times, log_wealth_path, first_cross, lam_grid)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
T, CP = 2400, 500
N_NULL, N_ALT = 400, 200
Z_LO, Z_HI = -1.0, 6.0
LAMS = lam_grid(Z_LO, Z_HI, 20)
THR = np.log(1 / ALPHA)


def rollout(sim, T, cp=None, drift=0.0, shape="step", noise="gauss", rng=None):
    """Controlled linear-Gaussian rollout with configurable drift shape and noise tails."""
    rng = rng or sim._rng
    d, m = sim.dim, sim.action_dim
    sig = np.sqrt(sim.Q[0, 0])
    if noise == "gauss":
        w = sig * rng.normal(size=(T, d))
    else:                                           # Student-t(5), unit variance
        nu = 5.0
        w = sig * rng.standard_t(nu, size=(T, d)) * np.sqrt((nu - 2) / nu)
    a = sim.sigma_a * rng.normal(size=(T, m))
    z = np.zeros((T + 1, d))
    z[0] = np.linalg.cholesky(sim.Sigma0) @ rng.normal(size=d)
    Ba = a @ sim.Bmat.T
    for t in range(T):
        if cp is None or t < cp:
            f = 0.0
        elif shape == "step":
            f = drift
        elif shape == "ramp":
            f = drift * min(1.0, (t - cp) / 300.0)
        else:                                       # pulse: on for 150 rounds, then off
            f = drift if t < cp + 150 else 0.0
        z[t + 1] = sim.A @ z[t] + Ba[t] + w[t] + f * sim.drift_dir
    return z, a


def stat_calib(sim, z, a):
    return excess_all_issue_times(sim, z, a, 1)


def stat_logscore(sim, z, a):
    """Log-score advantage of the model over the climatology reference, clipped."""
    d = sim.dim
    n = a.shape[0]
    mu = z[:n] @ sim.A_model.T + a @ sim.B_model.T
    Sm = sim.Q_model
    diff = z[1:n + 1] - mu
    nllm = 0.5 * np.einsum("ij,ij->i", diff, np.linalg.solve(Sm, diff.T).T) \
        + 0.5 * np.linalg.slogdet(2 * np.pi * Sm)[1]
    S0 = sim.Sigma0
    d0 = z[1:n + 1] - sim.mu0
    nllr = 0.5 * np.einsum("ij,ij->i", d0, np.linalg.solve(S0, d0.T).T) \
        + 0.5 * np.linalg.slogdet(2 * np.pi * S0)[1]
    return np.clip(nllm - nllr, -8.0, 8.0)       # advantage; negative = model better


def alarm_time(zv, eps, kind="mix", lo=Z_LO, hi=Z_HI, lam_fixed=None):
    if kind == "mix":
        lw = log_wealth_path(zv, eps, lam_grid(lo, hi, 20))
        return first_cross(lw, THR)
    if kind == "fixed":
        lw = np.cumsum(np.log(np.maximum(1 + lam_fixed * (zv - eps), 1e-300)))
        return first_cross(lw, THR)
    ep = EProcess(eps, lo, hi, alpha=ALPHA, mixture=False)
    for i, v in enumerate(zv):
        ep.update(float(v))
        if ep.rejected:
            return i
    return None


def evaluate(make_stat, eps, kind="mix", lo=Z_LO, hi=Z_HI, shape="step", drift=0.25,
             noise="gauss", lam_fixed=None, sim=None, n_null=N_NULL, n_alt=N_ALT):
    sim = sim or Simulator(dim=4, seed=1)
    fa = 0
    for _ in range(n_null):
        z, a = rollout(sim, T, noise=noise)
        if alarm_time(make_stat(sim, z, a), eps, kind, lo, hi, lam_fixed) is not None:
            fa += 1
    delays, det = [], 0
    for _ in range(n_alt):
        z, a = rollout(sim, T, cp=CP, drift=drift, shape=shape, noise=noise)
        s = make_stat(sim, z, a)
        al = alarm_time(s, eps, kind, lo, hi, lam_fixed)
        if al is not None and al >= CP:
            delays.append(al + 1 - CP); det += 1
    return dict(fa=fa / n_null, detect=det / max(n_alt, 1),
                delay=float(np.mean(delays)) if delays else None)


def show(title, rows):
    print(f"\n{title}")
    for k, v in rows.items():
        dl = "  --  " if v["delay"] is None else f"{v['delay']:6.1f}"
        print(f"  {k:<34} FA={v['fa']:.3f}  detect={v['detect']:.2f}  delay={dl}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = {}

    # A. bet strategy
    rows = {"mixture (20 lambdas)": evaluate(stat_calib, 0.0, "mix"),
            "GRAPA plug-in": evaluate(stat_calib, 0.0, "grapa")}
    lm = 1.0 / (Z_HI - Z_LO)
    for f in (0.1, 0.5, 1.0):
        rows[f"fixed lambda = {f:.1f} lam_max"] = evaluate(stat_calib, 0.0, "fixed",
                                                          lam_fixed=f * lm)
    show("A. bet strategy", rows); out["A_bet"] = rows

    # B. tolerance: a model with a genuine small gap
    simb = Simulator(dim=4, seed=1, model_bias=0.04)
    z, a = rollout(simb, 60000)
    g = float(stat_calib(simb, z, a).mean())
    rows = {}
    for mult in (0.0, 0.5, 1.0, 1.5, 2.5):
        rows[f"eps = {mult:.1f} x true gap g"] = evaluate(
            stat_calib, mult * g, sim=simb, drift=0.0, n_null=200, n_alt=0)
    print(f"\nB. tolerance (true one-step gap g={g:.4f}); 'FA' = revocation rate on a model "
          "whose gap is exactly g")
    for k, v in rows.items():
        print(f"  {k:<34} revoked={v['fa']:.3f}")
    out["B_tolerance"] = dict(g=g, rows=rows)

    # C. drift shape
    rows = {s: evaluate(stat_calib, 0.0, shape=s, n_null=100) for s in ("step", "ramp", "pulse")}
    show("C. drift shape (drift 0.25)", rows); out["C_shape"] = rows

    # D. score statistic
    rows = {"calibration excess (Mahalanobis)": evaluate(stat_calib, 0.0),
            "log-score vs climatology (eps=-1.0)":
                evaluate(stat_logscore, -1.0, lo=-8.0, hi=8.0)}
    # tolerance set to the null mean of the log-score advantage (calibrated reference gap)
    sim = Simulator(dim=4, seed=1)
    z, a = rollout(sim, 60000)
    m0 = float(stat_logscore(sim, z, a).mean())
    rows["log-score vs climatology (eps=null mean)"] = evaluate(
        stat_logscore, m0, lo=-8.0, hi=8.0)
    show(f"D. score statistic (log-score null mean {m0:.2f})", rows); out["D_score"] = rows

    # E. noise tails
    rows = {"Gaussian noise": evaluate(stat_calib, 0.0),
            "Student-t(5) noise (variance-matched)": evaluate(stat_calib, 0.0, noise="t")}
    show("E. noise tails", rows); out["E_tails"] = rows

    with open(os.path.join(RESULTS, "exp7_ablations.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
