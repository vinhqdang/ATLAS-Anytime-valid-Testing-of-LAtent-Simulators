"""Experiment 9 -- empirical check of the anytime-valid certified horizon (Theorem 2).

Each horizon ``h`` carries a confidence sequence on the average conditional mean of its
own (non-overlapping-chain) calibration statistic. For the standardised Gaussian statistic
with matched covariance the divergence is ``KL = (d/2) E[Z]`` (certifiability constants
``c = d/2``, ``rho = 0``), so with Pinsker's modulus the lemma certifies

    window-average TV_h  <=  min(1, sqrt( c * U^{(h)}_t / 2 )),   simultaneously in t and h.

On the linear-Gaussian simulator the true TV_h is available in closed form, so we report
(i) the violation rate of the certified bound over independent deployments at checkpoints
before and after a dynamics drift (the guarantee is <= alpha = 0.05) and (ii) its slack,
bound / truth.

Run: ``python -m experiments.exp9_simlemma``
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import norm

from atlas.simulator import Simulator
from atlas.cs import AvgCondMeanCS, WindowedCS
from atlas.fastsim import excess_all_issue_times, chain

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
CP, T, W = 500, 2500, 400
DRIFT = 0.25
HORIZONS = [1, 3, 8]
CHECK = [CP - 1, CP + 300, CP + 800, T - 30]
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
    for _ in range(N):
        z, a = sim.rollout(T, changepoint=CP, drift_mag=DRIFT)
        for h in HORIZONS:
            ex = excess_all_issue_times(sim, z, a, h)
            zv, rt = chain(ex, h, 0)                         # resolution rounds rt
            wh = max(W // h, 4)                              # window length in chain updates
            full = AvgCondMeanCS(-1.0, sim.B, alpha=ALPHA / (2 * len(HORIZONS)))
            win = WindowedCS(-1.0, sim.B, window=wh, alpha=ALPHA / (2 * len(HORIZONS)))
            ups = {}
            ci = 0
            for k, v in enumerate(zv):
                full.update(v); win.update(v)
                while ci < len(CHECK) and rt[k] >= CHECK[ci]:
                    c = CHECK[ci]
                    ups[c] = max(full.upper if c < CP else win.upper, 0.0)
                    # fraction of the window's chain updates that are post-drift
                    lo = win._start() if c >= CP else 0   # the CS's own window
                    issue = rt[lo:k + 1] - h
                    frac = float(np.mean(issue >= CP))
                    truth = frac * tv_true[h]
                    bound = min(1.0, np.sqrt((d / 2.0) * ups[c] / 2.0))
                    if truth > bound + 1e-9:
                        viol[(c, h)] += 1
                    if truth > 1e-6:
                        ratio[(c, h)].append(bound / truth)
                    ci += 1

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
