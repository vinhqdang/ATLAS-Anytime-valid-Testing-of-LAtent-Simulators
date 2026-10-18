"""Experiment 5 -- horizon-resolved detection delay and the interleaved estimator.

Theorem 3 (delay) is stated per horizon: horizon ``h`` is updated only once every ``h``
rounds (non-overlapping windows), but a drift compounds along the rollout, so the
post-change margin grows with ``h``. The two effects compete, which predicts a
*non-monotone* delay profile with an interior optimum -- the shortest horizon is not
always the fastest detector. We

  (a) measure, per horizon, the number of rounds from the changepoint to revocation for the
      paper's non-overlapping estimator and for the interleaved estimator (all ``h``
      offset chains, threshold ``h H / alpha``);
  (b) compare against the prediction ``h * (log(H h?) ... ) / D*_h`` with the growth
      rate ``D*_h = sup_lambda E log(1 + lambda Z)`` evaluated by Monte Carlo on the
      post-change stationary statistic; and
  (c) verify validity of both estimators under the null (false-revocation rate).

Run: ``python -m experiments.exp5_horizon_delay``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.fastsim import (excess_all_issue_times, chain, log_wealth_path,
                           first_cross, lam_grid)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
HORIZONS = [1, 2, 3, 5, 8, 12, 20]
H = len(HORIZONS)
T = 2400
CP = 400
DRIFT = 0.25
N_RUNS = int(os.environ.get("N_RUNS", 150))
Z_LO, Z_HI = -1.0, 6.0
NGRID = 20


def growth_rate(zs, lams):
    """sup_lambda E log(1 + lambda Z) over the mixture grid's range, by Monte Carlo."""
    ll = np.log(np.maximum(1.0 + np.outer(zs, lams), 1e-300)).mean(axis=0)
    return float(max(ll.max(), 1e-9))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sim = Simulator(dim=4, seed=1)
    lams = lam_grid(Z_LO, Z_HI, NGRID)
    thr_single = np.log(H / ALPHA)

    # -------- post-change stationary statistic -> growth rate D*_h ----------------
    zpost, apost = sim.rollout(120_000, changepoint=0, drift_mag=DRIFT)
    dstar = {}
    for h in HORIZONS:
        ex = excess_all_issue_times(sim, zpost[2000:], apost[2000:], h)
        dstar[h] = growth_rate(chain(ex, h)[0], lams)
    pred = {h: h * (thr_single + np.log(NGRID)) / dstar[h] for h in HORIZONS}

    # -------- delay under drift: non-overlapping vs interleaved -------------------
    d_non = {h: [] for h in HORIZONS}
    d_int = {h: [] for h in HORIZONS}
    miss_non = {h: 0 for h in HORIZONS}
    miss_int = {h: 0 for h in HORIZONS}
    for r in range(N_RUNS):
        z, a = sim.rollout(T, changepoint=CP, drift_mag=DRIFT)
        for h in HORIZONS:
            ex = excess_all_issue_times(sim, z, a, h)
            # (i) non-overlapping, offset 0
            zv, rt = chain(ex, h, 0)
            lw = log_wealth_path(zv, 0.0, lams)
            i0 = first_cross(lw, thr_single)
            # a crossing before the change is a false revocation; count only post-change
            if i0 is None or rt[i0] < CP:
                if i0 is None:
                    miss_non[h] += 1
            else:
                d_non[h].append(int(rt[i0] - CP))
            # (ii) interleaved: h offset chains, each at level alpha/(H h)
            best = None
            for c in range(h):
                zc, rc = chain(ex, h, c)
                if len(zc) == 0:
                    continue
                lwc = log_wealth_path(zc, 0.0, lams)
                ic = first_cross(lwc, thr_single + np.log(h))
                if ic is not None and rc[ic] >= CP:
                    best = rc[ic] if best is None else min(best, rc[ic])
            if best is None:
                miss_int[h] += 1
            else:
                d_int[h].append(int(best - CP))

    # -------- validity under the null --------------------------------------------
    fa_non = {h: 0 for h in HORIZONS}
    fa_int = {h: 0 for h in HORIZONS}
    fa_any_non = 0
    fa_any_int = 0
    N_NULL = 400
    for r in range(N_NULL):
        z, a = sim.rollout(T, changepoint=None)
        any_n = any_i = False
        for h in HORIZONS:
            ex = excess_all_issue_times(sim, z, a, h)
            zv, _ = chain(ex, h, 0)
            if first_cross(log_wealth_path(zv, 0.0, lams), thr_single) is not None:
                fa_non[h] += 1; any_n = True
            hit = False
            for c in range(h):
                zc, _ = chain(ex, h, c)
                if len(zc) and first_cross(log_wealth_path(zc, 0.0, lams),
                                           thr_single + np.log(h)) is not None:
                    hit = True; break
            if hit:
                fa_int[h] += 1; any_i = True
        fa_any_non += any_n
        fa_any_int += any_i

    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    print("=" * 86)
    print("Experiment 5 -- horizon-resolved delay (drift=%.2f, %d runs)" % (DRIFT, N_RUNS))
    print("=" * 86)
    print(f"{'h':>3} {'D*_h':>8} {'pred':>8} {'non-ovl mean':>13} {'(miss)':>7} "
          f"{'interleaved':>12} {'(miss)':>7} {'speed-up':>9}")
    rows = []
    for h in HORIZONS:
        mn, mi = mean(d_non[h]), mean(d_int[h])
        print(f"{h:>3} {dstar[h]:>8.4f} {pred[h]:>8.0f} {mn:>13.1f} "
              f"{miss_non[h]:>7d} {mi:>12.1f} {miss_int[h]:>7d} {mn/mi:>9.2f}")
        rows.append(dict(h=h, dstar=dstar[h], predicted=pred[h], non_overlap=mn,
                         miss_non=miss_non[h], interleaved=mi, miss_int=miss_int[h],
                         median_non=med(d_non[h]), median_int=med(d_int[h])))
    print("\nnull false-revocation (any horizon): non-overlap=%.3f  interleaved=%.3f "
          "(guarantee %.2f)" % (fa_any_non / N_NULL, fa_any_int / N_NULL, ALPHA))
    print("per-horizon null revocation non-overlap:",
          {h: round(fa_non[h] / N_NULL, 3) for h in HORIZONS})
    print("per-horizon null revocation interleaved:",
          {h: round(fa_int[h] / N_NULL, 3) for h in HORIZONS})

    # -------- figure -------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    hs = np.array(HORIZONS)
    ax[0].plot(hs, [r["non_overlap"] for r in rows], "o-", color="#3b7dd8",
               label="measured, non-overlapping")
    ax[0].plot(hs, [r["interleaved"] for r in rows], "s-", color="#1b8a5a",
               label="measured, interleaved")
    ax[0].plot(hs, [r["predicted"] for r in rows], "k--", lw=1.2,
               label=r"theory $h(\log\frac{H}{\alpha}+\log m)/D^\star_h$")
    ax[0].set_xlabel("horizon $h$"); ax[0].set_ylabel("rounds to revoke after the drift")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Delay profile across horizons")
    ax[1].plot(hs, [r["dstar"] for r in rows], "o-", color="#c0392b")
    ax[1].set_xlabel("horizon $h$"); ax[1].set_ylabel(r"growth rate $D^\star_h$ (per update)")
    ax[1].set_xscale("log"); ax[1].set_title("Drift compounds along the rollout")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp5_horizon_delay.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    with open(os.path.join(RESULTS, "exp5_horizon_delay.json"), "w") as f:
        json.dump(dict(rows=rows, null_any_non=fa_any_non / N_NULL,
                       null_any_int=fa_any_int / N_NULL,
                       null_non={h: fa_non[h] / N_NULL for h in HORIZONS},
                       null_int={h: fa_int[h] / N_NULL for h in HORIZONS},
                       n_runs=N_RUNS, n_null=N_NULL, drift=DRIFT), f, indent=2)
    print("figure ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
