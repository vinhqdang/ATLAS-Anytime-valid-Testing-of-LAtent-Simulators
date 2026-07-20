"""Experiment 2 — the faithfulness frontier & the anytime-valid simulation lemma.

A single long deployment with a dynamics drift injected at a known changepoint.
ATLAS maintains:

  * ``delta1_hat(t)`` — the certified anytime upper bound on the one-step
    faithfulness gap (upper edge of the BettingCS at h=1), Theorem-1-as-CS;
  * ``h*(t)`` — the Theorem-2 trust horizon
    ``max{h : C(h,gamma) * phi(delta1_hat(t)) <= budget}``,
    the largest horizon a planner may legally imagine while keeping the certified
    value gap within budget (docs/03_theorems.md, Theorem 2).

The signature result: ``h*(t)`` holds high while the model is faithful and
**collapses right when the drift opens**, whereas an offline metric fixed at its
pre-deployment value is blind by construction.

Run: ``python -m experiments.exp2_frontier``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.eprocess import BettingCS
from atlas.frontier import amplification, trust_horizon_from_gap

ALPHA = 0.05
GAMMA = 0.98
BUDGET = 350.0         # planner's tolerated certified value gap
T = 3000
CP = 900
DRIFT = 0.7
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sim = Simulator(dim=4, seed=1)
    z, a = sim.rollout(T, changepoint=CP, drift_mag=DRIFT)
    by_time, truth = sim.resolved_stream(z, a, [1], changepoint=CP)

    cs = BettingCS(z_lo=-1.0, z_hi=sim.B, alpha=ALPHA)
    delta1 = np.full(T + 1, np.nan)
    hstar = np.full(T + 1, np.nan)
    h_grid = np.arange(1, 81)
    frozen_delta1 = np.nan
    for t, row in enumerate(by_time):
        if 1 in row:
            cs.update(row[1])
        if cs.k > 0:
            d = cs.upper
            delta1[t] = d
            hstar[t] = trust_horizon_from_gap(d, GAMMA, BUDGET, h_grid=h_grid)
        if t == CP:                       # freeze the offline estimate at deployment
            frozen_delta1 = cs.upper

    # offline metric: certified once at deployment (t=CP), then frozen -> blind
    offline_gap = frozen_delta1
    offline_hstar = trust_horizon_from_gap(max(offline_gap, 0), GAMMA, BUDGET,
                                           h_grid=h_grid)

    true_pre = float(truth["raw_excess"][1][:CP].mean())
    true_post = float(truth["raw_excess"][1][CP:].mean())

    print("=" * 64)
    print("Experiment 2 — frontier collapse & anytime-valid simulation lemma")
    print("=" * 64)
    print(f"changepoint={CP}, drift={DRIFT}, gamma={GAMMA}, budget={BUDGET}")
    print(f"true one-step gap: pre={true_pre:+.3f}  post={true_post:+.3f}")
    valid = np.nanmax(delta1[10:CP]) >= true_pre    # CS upper-bounds the pre gap
    print(f"delta1_hat: pre~{np.nanmean(delta1[CP-100:CP]):.3f}  "
          f"post-end~{delta1[T-1]:.3f}  (valid upper bound pre: {valid})")
    print(f"h*: pre={int(np.nanmax(hstar[CP-100:CP]))}  final={int(hstar[T-1])}  "
          f"| offline (frozen) h*={offline_hstar}")
    # when did h* first drop after cp?
    post = hstar[CP:]
    base = int(np.nanmax(hstar[CP-100:CP]))
    drop = np.argmax(post < base)
    print(f"h* first drops {drop} rounds after the changepoint "
          f"(from {base} toward {int(hstar[T-1])})")

    # ---- figure -------------------------------------------------------------
    tt = np.arange(T + 1)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0.plot(tt, delta1, color="#3b7dd8", lw=1.6, label=r"$\hat\delta_1(t)$ (ATLAS, anytime-valid)")
    ax0.axhline(offline_gap, color="#999", ls=":", lw=1.6,
                label="offline metric (frozen pre-deployment)")
    ax0.axvline(CP, color="crimson", ls="--", lw=1.2, label="dynamics drift onset")
    ax0.set_ylabel("certified one-step gap")
    ax0.set_title("ATLAS tracks the faithfulness gap; the offline metric is blind")
    ax0.legend(frameon=False, loc="upper left")

    ax1.plot(tt, hstar, color="#1b8a5a", lw=2.0, label=r"$h^*(t)$ ATLAS trust horizon")
    ax1.axhline(offline_hstar, color="#999", ls=":", lw=1.6,
                label="offline-metric trust horizon (blind)")
    ax1.axvline(CP, color="crimson", ls="--", lw=1.2)
    ax1.set_xlabel("deployment round")
    ax1.set_ylabel(r"trust horizon $h^*(t)$")
    ax1.set_title(r"$h^*(t)$ collapses exactly when the sim-to-real gap opens")
    ax1.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out = os.path.abspath(os.path.join(RESULTS, "exp2_frontier.png"))
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")
    return dict(true_pre=true_pre, true_post=true_post,
                hstar_pre=int(np.nanmax(hstar[CP-100:CP])),
                hstar_final=int(hstar[T-1]), offline_hstar=offline_hstar)


if __name__ == "__main__":
    main()
