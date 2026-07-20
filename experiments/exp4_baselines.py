"""Experiment 4 -- ATLAS vs competitor drift monitors.

We compare ATLAS's anytime-valid e-process against three standard/online monitors
(offline fixed threshold, sliding-window z-test, CUSUM) on the controllable
simulator, where ground truth is known. Every monitor consumes the same one-step
score stream. The baselines' thresholds are calibrated on null runs to hit a 5%
false-alarm rate at a nominal monitoring horizon T_cal; we then report:

  * false-alarm rate at T_cal (calibration horizon),
  * false-alarm rate at 3x T_cal (longer monitoring -- more looks),
  * mean detection delay after a drift.

The point: ATLAS controls the false-alarm probability at *every* horizon by
construction (Ville), so it stays valid when monitored longer; the baselines,
tuned at T_cal, inflate their false-alarm rate under continued monitoring -- the
multiple-looks problem that anytime-valid inference is designed to solve.

Run: ``python -m experiments.exp4_baselines``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.eprocess import EProcess
from experiments.baselines import FixedThreshold, SlidingZTest, CUSUM, run_monitor

ALPHA = 0.05
T_CAL = 400
T_EVAL = 1200
CP = 400
DRIFT = 0.6
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def stream(seed, T, changepoint=None, drift=0.0):
    sim = Simulator(dim=4, seed=seed)
    z, a = sim.rollout(T, changepoint=changepoint, drift_mag=drift)
    by_time, _ = sim.resolved_stream(z, a, [1], changepoint=changepoint)
    return np.array([by_time[t][1] for t in range(len(by_time)) if 1 in by_time[t]]), sim.B


def q95_of_run_max(stat_fn, n=150, seed0=0):
    """Calibrate a threshold: 95th percentile of the per-run maximum statistic over
    null runs of length T_CAL (gives ~5% false alarm at T_CAL by construction)."""
    maxes = []
    for s in range(n):
        z, _ = stream(seed0 + s, T_CAL)
        maxes.append(stat_fn(z))
    return float(np.quantile(maxes, 1 - ALPHA))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # reference null statistics (mean0, sd0) for z-test / cusum
    zref, B = stream(9999, 2000)
    m0, s0 = float(zref.mean()), float(zref.std())

    # --- calibrate baseline thresholds on null runs (target 5% FA at T_CAL) ------
    W = 20
    def roll_max(z):
        c = np.convolve(z, np.ones(W) / W, mode="valid"); return c.max() if len(c) else 0
    def z_max(z):
        c = np.convolve(z, np.ones(W) / W, mode="valid")
        return ((c - m0) / (s0 / np.sqrt(W) + 1e-12)).max() if len(c) else 0
    def cusum_max(z):
        S, mx = 0.0, 0.0
        for v in z:
            S = max(0.0, S + (v - m0 - 0.1)); mx = max(mx, S)
        return mx
    thr_fixed = q95_of_run_max(roll_max)
    thr_z = q95_of_run_max(z_max)
    thr_cusum = q95_of_run_max(cusum_max)

    def make_monitors():
        return {
            "ATLAS (e-process)": EProcess(eps=0.0, z_lo=-1.0, z_hi=B, alpha=ALPHA),
            "Fixed threshold": FixedThreshold(thr_fixed, window=W),
            "Sliding z-test": SlidingZTest(m0, s0, thr_z, window=W),
            "CUSUM": CUSUM(m0, 0.1, thr_cusum),
        }
    names = list(make_monitors().keys())

    def hit(mon, v):
        if isinstance(mon, EProcess):
            mon.update(v); return mon.rejected
        return mon.step(v)

    def first_alarm(mon, z):
        for t, v in enumerate(z):
            if hit(mon, v):
                return t
        return None

    # --- evaluate false-alarm at T_CAL and T_EVAL (fresh null runs) --------------
    Nfa = 300
    fa_cal = {n: 0 for n in names}; fa_eval = {n: 0 for n in names}
    for s in range(Nfa):
        z, _ = stream(20000 + s, T_EVAL)
        for n, mon in make_monitors().items():
            a = first_alarm(mon, z)
            if a is not None and a < T_CAL:
                fa_cal[n] += 1
            if a is not None:
                fa_eval[n] += 1

    # --- detection delay after a drift ------------------------------------------
    Ndel = 200
    delays = {n: [] for n in names}
    for s in range(Ndel):
        z, _ = stream(30000 + s, T_EVAL, changepoint=CP, drift=DRIFT)
        for n, mon in make_monitors().items():
            for t, v in enumerate(z):
                if hit(mon, v) and t >= CP:
                    delays[n].append(t - CP); break

    print("=" * 74)
    print("Experiment 4 -- ATLAS vs competitor drift monitors")
    print("=" * 74)
    print(f"{'monitor':<20}{'FA@T_cal':>10}{'FA@3T_cal':>11}{'mean delay':>12}")
    for n in names:
        md = np.mean(delays[n]) if delays[n] else float('nan')
        print(f"{n:<20}{fa_cal[n]/Nfa:>10.3f}{fa_eval[n]/Nfa:>11.3f}{md:>12.1f}")
    print(f"\ntarget false-alarm rate alpha = {ALPHA}")

    # --- figure -----------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))
    ax0.bar(x - 0.2, [fa_cal[n] for n in names], 0.4, label=r"at $T_{\rm cal}$", color="#9ecae1")
    ax0.bar(x + 0.2, [fa_eval[n] for n in names], 0.4, label=r"at $3\,T_{\rm cal}$", color="#3b7dd8")
    ax0.axhline(ALPHA, color="crimson", ls="--", label=r"target $\alpha$")
    ax0.set_xticks(x); ax0.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax0.set_ylabel("false-alarm rate"); ax0.set_title("Validity under longer monitoring")
    ax0.legend(frameon=False, fontsize=8)
    ax1.bar(x, [np.mean(delays[n]) if delays[n] else 0 for n in names], color="#1b8a5a")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("mean detection delay (rounds)"); ax1.set_title("Detection delay after drift")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp4_baselines.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"figure -> {os.path.abspath(out)}")
    return dict(fa_cal=fa_cal, fa_eval=fa_eval,
                mean_delay={n: (float(np.mean(delays[n])) if delays[n] else None) for n in names})


if __name__ == "__main__":
    main()
