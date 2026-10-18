"""Experiment 2 -- the faithfulness frontier and the certified one-step gap.

One long deployment with a dynamics drift injected at a known changepoint. ATLAS maintains

  * ``U(t)``   -- the anytime-valid *windowed* upper confidence bound on the average one-step
                  gap (``atlas.cs.WindowedCS``; the object the simulation lemma consumes), and
  * ``h*(t)``  -- the faithfulness frontier from the per-horizon e-processes
                  (non-overlapping chains, level alpha/H each, monotone rule).

An offline metric frozen at its pre-deployment value is blind by construction.

Run: ``python -m experiments.exp2_frontier``
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.cs import WindowedCS
from atlas.fastsim import (excess_all_issue_times, chain, log_wealth_path, first_cross,
                           lam_grid)

ALPHA = 0.05
HORIZONS = [1, 2, 3, 5, 8, 12, 20]
T, CP, DRIFT, WINDOW = 2500, 900, 0.6, 150
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sim = Simulator(dim=4, seed=1)
    z, a = sim.rollout(T, changepoint=CP, drift_mag=DRIFT)
    H = len(HORIZONS)
    lams = lam_grid(-1.0, sim.B, 20)

    # ---- frontier from per-horizon e-processes
    revoke_round = {}
    for h in HORIZONS:
        ex = excess_all_issue_times(sim, z, a, h)
        zv, rt = chain(ex, h, 0)
        i = first_cross(log_wealth_path(zv, 0.0, lams), np.log(H / ALPHA))
        revoke_round[h] = int(rt[i]) if i is not None else None
    tt = np.arange(1, T + 1)
    hstar = np.empty(len(tt), int)
    for j, t in enumerate(tt):
        hs = 0
        for h in HORIZONS:
            if revoke_round[h] is not None and revoke_round[h] <= t:
                break
            hs = h
        hstar[j] = hs

    # ---- certified windowed one-step gap (KL units: c * U with c = d/2)
    ex1 = excess_all_issue_times(sim, z, a, 1)
    cs = WindowedCS(-1.0, sim.B, window=WINDOW, alpha=ALPHA)
    kl_cert = np.empty(len(ex1))
    for t, v in enumerate(ex1):
        cs.update(v)
        kl_cert[t] = (sim.dim / 2.0) * max(cs.upper, 0.0)
    d = sim.dim
    kl_true = np.where(np.arange(len(ex1)) >= CP, 0.5 * DRIFT ** 2
                       * float(sim.drift_dir @ np.linalg.solve(sim.Q, sim.drift_dir)), 0.0)
    # the certificate is for the *window average*; compare against the true window average
    kl_true_win = np.empty(len(kl_true))
    for t in range(len(kl_true)):
        st = max((t + 1 - WINDOW) // cs.block, 0) * cs.block
        kl_true_win[t] = kl_true[st:t + 1].mean()
    offline_kl = float(kl_cert[CP - 1])
    offline_h = int(hstar[CP - 1])

    first_drop = next((int(t) for t, hs in zip(tt, hstar) if t >= CP and hs < hstar[CP - 1]),
                      None)
    print("=" * 64)
    print("Experiment 2 -- frontier and certified one-step gap")
    print("=" * 64)
    print(f"drift={DRIFT}, changepoint={CP}; revocation round per horizon:")
    print({h: (None if r is None else r - CP) for h, r in revoke_round.items()},
          "(rounds after the change)")
    print(f"h*: pre={hstar[CP-1]}  final={hstar[-1]}  first drop {None if first_drop is None else first_drop - CP} rounds after change")
    print(f"certified gap (KL units): pre {kl_cert[CP-1]:.3f}  post {kl_cert[-1]:.3f}  "
          f"true post {kl_true[-1]:.3f}")
    print("certified >= true window average everywhere:", bool(np.all(kl_cert + 1e-9 >= kl_true_win)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    ax0.plot(kl_cert, color="#3b7dd8", lw=1.5, label=r"certified bound (windowed, anytime-valid)")
    ax0.plot(kl_true_win, color="k", lw=1.2, ls="--", label="true window-average gap (KL)")
    ax0.axhline(offline_kl, color="#999", ls=":", lw=1.5, label="offline estimate (frozen)")
    ax0.axvline(CP, color="crimson", ls="--", lw=1)
    ax0.set_ylabel("one-step gap (nats)"); ax0.legend(frameon=False, fontsize=8)
    ax0.set_title("Certified one-step gap tracks the drift")
    ax1.step(tt, hstar, where="post", color="#1b8a5a", lw=2.0, label=r"ATLAS frontier $h^\star(t)$")
    ax1.axhline(offline_h, color="#999", ls=":", lw=1.5, label="offline trust horizon (frozen)")
    ax1.axvline(CP, color="crimson", ls="--", lw=1, label="drift onset")
    ax1.set_xlabel("deployment round"); ax1.set_ylabel(r"$h^\star(t)$")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("The frontier holds while faithful and collapses after the drift")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp2_frontier.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    with open(os.path.join(RESULTS, "exp2_frontier.json"), "w") as f:
        json.dump(dict(revocation_delay={h: (None if r is None else r - CP)
                                         for h, r in revoke_round.items()},
                       hstar_pre=int(hstar[CP - 1]), hstar_final=int(hstar[-1]),
                       first_drop=None if first_drop is None else first_drop - CP,
                       kl_pre=float(kl_cert[CP - 1]), kl_post=float(kl_cert[-1]),
                       kl_true_post=float(kl_true[-1])), f, indent=2)
    print("figure ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
