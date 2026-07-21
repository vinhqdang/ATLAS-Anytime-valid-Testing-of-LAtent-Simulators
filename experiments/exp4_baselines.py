"""Experiment 4 -- ATLAS vs published drift/change detectors.

We compare ATLAS's anytime-valid e-process against established detectors from the
streaming and sequential-analysis literature---CUSUM (Page 1954), the Page--Hinkley test,
ADWIN (Bifet & Gavald{\`a} 2007), KSWIN (Raab et al. 2020), and Bayesian Online Changepoint
Detection (Adams & MacKay 2007)---on the controllable simulator where ground truth is
known. Every monitor consumes the same one-step calibration-excess stream.

Each competitor has a scalar sensitivity knob; we calibrate it on null runs so that its
false-alarm rate at a nominal horizon T_cal matches the 5% target, then report:

  * false-alarm rate at T_cal (the calibration horizon),
  * false-alarm rate at 3x T_cal (longer monitoring -- more looks),
  * mean detection delay after a drift.

ATLAS is not tuned: its false-alarm probability is controlled at *every* horizon by
construction (Ville), so it stays valid when monitored longer, whereas the calibrated
competitors inflate their false-alarm rate under continued monitoring -- the multiple-looks
problem that anytime-valid inference is designed to solve.

Run: ``python -m experiments.exp4_baselines``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.eprocess import EProcess, EDetector
from experiments.baselines import CUSUM, PageHinkley, ADWIN, KSWIN, BOCPD, ConformalMartingale

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


def hit(mon, v):
    """Advance a monitor one step; return True on alarm/rejection (type-dispatched)."""
    if isinstance(mon, EProcess):
        mon.update(v); return mon.rejected
    if isinstance(mon, EDetector):
        mon.update(v); return mon.alarm
    return mon.step(float(v))


def first_alarm(mon, z, tmax=None):
    for t, v in enumerate(z):
        if tmax is not None and t >= tmax:
            return None
        if hit(mon, float(v)):
            return t
    return None


def fa_rate(factory, s, n, seed0, horizon):
    """False-alarm rate at ``horizon`` over ``n`` null runs for sensitivity ``s``."""
    hits = 0
    for i in range(n):
        z, _ = stream(seed0 + i, horizon)
        if first_alarm(factory(s), z) is not None:
            hits += 1
    return hits / n


def calibrate(factory, grid, n=60, seed0=1000, target=ALPHA):
    """Pick the sensitivity on ``grid`` whose T_cal false-alarm rate is closest to target."""
    best, best_gap = grid[0], 1e9
    for s in grid:
        r = fa_rate(factory, s, n, seed0, T_CAL)
        if abs(r - target) < best_gap:
            best, best_gap = s, abs(r - target)
    return best


def main():
    os.makedirs(RESULTS, exist_ok=True)
    zref, B = stream(9999, 4000)
    m0, s0 = float(zref.mean()), float(zref.std())

    # detector families: name -> (factory(sensitivity), calibration grid)
    families = {
        "CUSUM": (lambda s: CUSUM(m0, 0.1, s),
                  list(np.linspace(0.5, 12.0, 16))),
        "Page-Hinkley": (lambda s: PageHinkley(thresh=s),
                         list(np.linspace(1.0, 40.0, 16))),
        "ADWIN": (lambda s: ADWIN(delta=s),
                  list(np.logspace(-6, -0.5, 14))),
        "KSWIN": (lambda s: KSWIN(alpha=s),
                  list(np.logspace(-5, -1.3, 14))),
        "BOCPD": (lambda s: BOCPD(thresh=int(s), mu0=m0),
                  [0, 1, 2, 3, 5, 8, 12, 20, 30]),
        "e-detector (2024)": (lambda s: EDetector(eps=0.0, z_lo=-1.0, z_hi=B, alpha=s),
                              list(np.logspace(-4, -0.3, 14))),
    }

    print("=" * 78)
    print("Experiment 4 -- ATLAS vs published drift/change detectors")
    print("=" * 78)

    # --- calibrate each competitor to ~5% FA at T_cal ---------------------------
    calib = {}
    for name, (fac, grid) in families.items():
        calib[name] = calibrate(fac, grid)
        print(f"  calibrated {name}: sensitivity={calib[name]:.4g}")

    # anytime-valid detectors need NO calibration: valid at level alpha by construction.
    # the conformal martingale uses a held-out null reference set of scores.
    ref_scores, _ = stream(7777, 3000)

    def make_all():
        d = {}
        # anytime-valid TESTS (untuned; control P(ever reject) <= alpha at every horizon)
        d["ATLAS (e-process)"] = EProcess(eps=0.0, z_lo=-1.0, z_hi=B, alpha=ALPHA)
        d["Conformal MG"] = ConformalMartingale(ref_scores, alpha=ALPHA)
        for name, (fac, _) in families.items():          # change detectors, threshold-tuned
            d[name] = fac(calib[name])
        return d
    names = list(make_all().keys())

    # --- false-alarm at T_cal and T_eval (fresh null runs) ----------------------
    Nfa = 300
    fa_cal = {n: 0 for n in names}; fa_eval = {n: 0 for n in names}
    for s in range(Nfa):
        z, _ = stream(20000 + s, T_EVAL)
        for n, mon in make_all().items():
            a = None
            for t, v in enumerate(z):
                if hit(mon, v):
                    a = t; break
            if a is not None and a < T_CAL:
                fa_cal[n] += 1
            if a is not None:
                fa_eval[n] += 1

    # --- detection delay after a drift ------------------------------------------
    Ndel = 200
    delays = {n: [] for n in names}
    for s in range(Ndel):
        z, _ = stream(30000 + s, T_EVAL, changepoint=CP, drift=DRIFT)
        for n, mon in make_all().items():
            for t, v in enumerate(z):
                if hit(mon, v) and t >= CP:
                    delays[n].append(t - CP); break

    print(f"\n{'monitor':<20}{'FA@T_cal':>10}{'FA@3T_cal':>11}{'mean delay':>12}")
    for n in names:
        md = np.mean(delays[n]) if delays[n] else float('nan')
        print(f"{n:<20}{fa_cal[n]/Nfa:>10.3f}{fa_eval[n]/Nfa:>11.3f}{md:>12.1f}")
    print(f"\ntarget false-alarm rate alpha = {ALPHA}")

    # --- figure -----------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(names))
    rc = [fa_cal[n] / Nfa for n in names]
    re = [fa_eval[n] / Nfa for n in names]
    ax0.bar(x - 0.2, rc, 0.4, label=r"at $T_{\rm cal}$", color="#9ecae1")
    ax0.bar(x + 0.2, re, 0.4, label=r"at $3\,T_{\rm cal}$", color="#3b7dd8")
    ax0.axhline(ALPHA, color="crimson", ls="--", label=r"target $\alpha=0.05$")
    ax0.axvline(1.5, color="grey", ls="-", lw=0.8, alpha=0.5)   # AV tests | tuned detectors
    ax0.text(0.75, 0.96, "anytime-valid\ntests", ha="center", va="top", fontsize=7,
             transform=ax0.get_xaxis_transform())
    ax0.set_xticks(x); ax0.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax0.set_ylabel("false-alarm rate"); ax0.set_title("Validity under longer monitoring")
    ax0.legend(frameon=False, fontsize=8)
    dvals = [np.mean(delays[n]) if delays[n] else 0 for n in names]
    ax1.bar(x, dvals, color="#1b8a5a")
    for xi, n in zip(x, names):                    # mark detectors that miss the drift
        if not delays[n]:
            ax1.text(xi, 1, "miss", rotation=90, ha="center", va="bottom", fontsize=7)
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
