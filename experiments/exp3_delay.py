"""Experiment 3 — Theorem 3: detection delay under drift.

Theorem 3 predicts a detection delay of ``log(1/alpha) / D*`` after an unknown
changepoint, where ``D* = sup_lambda E_post[log(1 + lambda (Z - eps))]`` is the
post-change growth rate (GRO). In the small-margin regime this is
``D* ~ m^2 / (2 sigma^2)`` with ``m`` the score-space margin, giving the familiar
``delay ~ 1/m^2`` scaling.

We validate the *formula itself*: for a range of small drift magnitudes we
  1. measure the realized score margin ``m`` and estimate ``D*`` empirically,
  2. measure ATLAS's mean detection delay (Shiryaev-Roberts e-detector) over seeds,
  3. compare the measured delay against the predicted ``log(1/alpha)/D*``.

Run: ``python -m experiments.exp3_delay``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.eprocess import EProcess

ALPHA = 0.05
H = 1
DRIFTS = [0.20, 0.26, 0.33, 0.42, 0.55, 0.75]
N_SEEDS = 100
T = 2600
CP = 400
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def gro_rate(z_post, lam_max, eps=0.0, n_grid=60):
    """Empirical GRO growth rate D* = max_lambda mean log(1 + lambda (Z - eps))."""
    lams = np.linspace(lam_max / n_grid, lam_max, n_grid)
    d = z_post - eps
    vals = [np.mean(np.log(np.maximum(1.0 + lam * d, 1e-12))) for lam in lams]
    return float(np.max(vals))


def one_delay(seed, drift):
    """Delay = first round after the changepoint that the (mixture) e-process wealth
    crosses 1/alpha. With eps=0 the pre-change wealth is a martingale (~1), so the
    crossing is signal-driven and the delay tracks log(1/alpha)/D* (Theorem 3)."""
    sim = Simulator(dim=4, seed=seed)
    z, a = sim.rollout(T, changepoint=CP, drift_mag=drift)
    by_time, _ = sim.resolved_stream(z, a, [H], changepoint=CP)
    ep = EProcess(eps=0.0, z_lo=-1.0, z_hi=sim.B, alpha=ALPHA)
    for t, row in enumerate(by_time):
        if H in row:
            ep.update(row[H])
        if ep.rejected:
            if t >= CP:
                return t - CP
            return np.nan          # rare pre-change false alarm (prob <= alpha): drop
    return np.nan


def measure_margin_and_rate(drift, seeds=20):
    """Realized post-change margin m and GRO rate D* for a drift (averaged)."""
    ms, ds = [], []
    for s in range(seeds):
        sim = Simulator(dim=4, seed=1000 + s)
        z, a = sim.rollout(T, changepoint=CP, drift_mag=drift)
        by_time, truth = sim.resolved_stream(z, a, [H], changepoint=CP)
        zpost = np.array([by_time[t][H] for t in range(CP + 1, T + 1) if H in by_time[t]])
        ms.append(zpost.mean())
        ds.append(gro_rate(zpost, lam_max=1.0 / (sim.B + 1.0)))
    return float(np.mean(ms)), float(np.mean(ds))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    drifts = np.array(DRIFTS)
    margins, rates, mean_delay, se_delay = [], [], [], []
    for d in drifts:
        m, Dstar = measure_margin_and_rate(d)
        ds = np.array([one_delay(s, d) for s in range(N_SEEDS)], dtype=float)
        ds = ds[~np.isnan(ds)]
        margins.append(m); rates.append(Dstar)
        mean_delay.append(ds.mean()); se_delay.append(ds.std() / np.sqrt(max(len(ds), 1)))
    margins = np.array(margins); rates = np.array(rates)
    mean_delay = np.array(mean_delay); se_delay = np.array(se_delay)
    # mixture betting pays an extra log(n_grid) to fund the near-optimal component,
    # so the Theorem-3 delay for the mixture e-process is (log(1/a)+log m)/D*.
    N_GRID = 20
    thresh_logs = np.log(1.0 / ALPHA) + np.log(N_GRID)
    predicted = thresh_logs / rates
    inv_D = 1.0 / rates

    b, _ = np.polyfit(np.log(margins), np.log(mean_delay), 1)
    slope_invD, intercept = np.polyfit(inv_D, mean_delay, 1)

    print("=" * 74)
    print("Experiment 3 — Theorem 3 (detection delay under drift)")
    print("=" * 74)
    print(f"alpha={ALPHA}, seeds={N_SEEDS}, changepoint={CP}, "
          f"log(1/a)+log(n_grid)={thresh_logs:.2f}")
    print(f"{'drift':>6} {'margin m':>9} {'D*':>7} {'delay(meas)':>12} "
          f"{'(log1/a+logm)/D*':>17} {'delay*D*':>9}")
    for i, d in enumerate(drifts):
        print(f"{d:6.2f} {margins[i]:9.3f} {rates[i]:7.3f} "
              f"{mean_delay[i]:8.1f}+-{se_delay[i]:<3.0f} {predicted[i]:17.1f} "
              f"{mean_delay[i]*rates[i]:9.2f}")
    rel = np.abs(mean_delay - predicted) / mean_delay
    print(f"\nTheorem-3 formula (log1/a+logm)/D*:  mean relative error {rel.mean()*100:.0f}%")
    print(f"delay vs 1/D* linear fit: slope={slope_invD:.2f} "
          f"(~ log(1/a)+log(n_grid)={thresh_logs:.2f}), intercept={intercept:.1f}")
    print(f"empirical D* ~ m^{np.polyfit(np.log(margins), np.log(rates),1)[0]:.2f} "
          f"(capped-bet regime -> D* prop m -> delay prop 1/m; slope={b:.2f})")

    # ---- figure -------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax0.errorbar(inv_D, mean_delay, yerr=se_delay, fmt="o", color="#3b7dd8",
                 capsize=3, label="ATLAS (measured)")
    xs = np.linspace(0, inv_D.max() * 1.05, 50)
    ax0.plot(xs, thresh_logs * xs, "--", color="crimson",
             label=r"Theorem 3: $(\log\frac{1}{\alpha}+\log m)\,/\,D^*$")
    ax0.set_xlabel(r"$1/D^*$  (inverse GRO growth rate)")
    ax0.set_ylabel("detection delay (rounds)")
    ax0.set_title(r"Delay $\propto 1/D^*$ — the Theorem-3 law")
    ax0.legend(frameon=False)

    order = np.argsort(margins)
    ax1.loglog(margins[order], mean_delay[order], "o-", color="#3b7dd8",
               label="ATLAS (measured)")
    ax1.loglog(margins[order], predicted[order], "s--", color="crimson",
               label=r"$(\log\frac{1}{\alpha}+\log m)/D^*$")
    ax1.set_xlabel(r"score-space margin $m$ (log)")
    ax1.set_ylabel("delay (log)")
    ax1.set_title(fr"Delay shrinks with margin (slope ${b:.2f}$)")
    ax1.legend(frameon=False)

    fig.tight_layout()
    out = os.path.abspath(os.path.join(RESULTS, "exp3_delay.png"))
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")
    return dict(margins=margins.tolist(), mean_delay=mean_delay.tolist(),
                predicted=predicted.tolist(), slope_invD=float(slope_invD))


if __name__ == "__main__":
    main()
