"""Experiment 9 -- empirical check of the anytime-valid simulation lemma.

On the linear-Gaussian simulator the true h-step total-variation gap between the model's
action-conditioned predictive and the true law is available in closed form, so the lemma

    avg_{issue times in a window} TV_h  <=  h * phi( c * U_t + rho )

(open-loop version, Theorem 2(a); phi(u) = sqrt(u/2), certifiability c = d/2, rho = 0 for the
standardised Gaussian statistic with matched covariance) can be checked against the truth.
We report (i) the violation rate of the certified bound over independent deployments,
at checkpoints before and after a dynamics drift (the guarantee is <= alpha = 0.05), and
(ii) its slack, bound / truth.

Run: ``python -m experiments.exp9_simlemma``
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import norm

from atlas.simulator import Simulator
from atlas.cs import AvgCondMeanCS, WindowedCS
from atlas.fastsim import excess_all_issue_times

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
CP, T, W = 500, 1800, 200
DRIFT = 0.25
HORIZONS = [1, 3, 8]
CHECK = [CP - 1, CP + 150, CP + 400, T - 1]
N = 150


def tv_gauss(delta):
    return 2.0 * norm.cdf(delta / 2.0) - 1.0


def main():
    sim = Simulator(dim=4, seed=1)
    d = sim.dim
    # exact post-drift Mahalanobis offsets delta_h and TV_h
    tv_true = {}
    for h in HORIZONS:
        m = np.zeros(d)
        P = np.eye(d)
        Sig = np.zeros((d, d))
        for i in range(h):
            m = m + P @ (DRIFT * sim.drift_dir)
            Sig += P @ sim.Q @ P.T
            P = sim.A @ P
        tv_true[h] = float(tv_gauss(np.sqrt(m @ np.linalg.solve(Sig, m))))
    print("exact post-drift TV_h:", {h: round(v, 4) for h, v in tv_true.items()})

    viol = {(c, h): 0 for c in CHECK for h in HORIZONS}
    ratio = {(c, h): [] for c in CHECK for h in HORIZONS}
    bound_hist = {c: [] for c in CHECK}
    for _ in range(N):
        z, a = sim.rollout(T, changepoint=CP, drift_mag=DRIFT)
        zs = excess_all_issue_times(sim, z, a, 1)           # one-step statistic, all rounds
        full = AvgCondMeanCS(-1.0, sim.B, alpha=ALPHA / 2)
        win = WindowedCS(-1.0, sim.B, window=W, alpha=ALPHA / 2)
        ups = {}
        for t, v in enumerate(zs):
            full.update(v); win.update(v)
            if t in CHECK:
                # pre-drift: the whole history is in-distribution; afterwards use the window
                u = full.upper if t < CP else win.upper
                ups[t] = max(u, 0.0)
        for c in CHECK:
            kl_bound = (d / 2.0) * ups[c]                    # c * U (+ rho = 0)
            tv_1 = np.sqrt(kl_bound / 2.0)                  # Pinsker modulus
            lo = max(c - W, 0)
            frac_post = np.mean([1.0 if t >= CP else 0.0 for t in range(lo, c + 1)])
            for h in HORIZONS:
                truth = frac_post * tv_true[h]              # window-average of the true TV_h
                bound = min(h * tv_1, 1.0)
                if truth > bound + 1e-9:
                    viol[(c, h)] += 1
                if truth > 1e-6:
                    ratio[(c, h)].append(bound / truth)
            bound_hist[c].append(tv_1)

    print("\ncheckpoint rounds:", CHECK, "(drift opens at", CP, ")")
    out = {}
    for c in CHECK:
        for h in HORIZONS:
            r = ratio[(c, h)]
            vr = viol[(c, h)] / N
            print(f"  t={c:>5} h={h:>2}  violation rate={vr:.3f}  "
                  f"median slack (bound/true)={np.median(r) if r else float('nan'):.2f}")
            out[f"t{c}_h{h}"] = dict(violation=vr, median_slack=float(np.median(r)) if r else None)
    out["tv_true"] = tv_true
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "exp9_simlemma.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
