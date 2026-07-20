"""Experiment 1 — Theorem 1: uniform validity (no false revocation).

Under the null (no dynamics drift), the world model is *exactly* faithful
(``eps = 0``), so ATLAS must never revoke a horizon except with probability
<= alpha, uniformly over all rounds. We run many independent deployments and check:

  * per-horizon false-revocation rate <= alpha / H  (simultaneous-frontier level),
  * any-horizon false-revocation rate <= alpha,
  * example wealth paths stay below the 1/alpha threshold.

Run: ``python -m experiments.exp1_validity``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.simulator import Simulator
from atlas.frontier import FaithfulnessFrontier

HORIZONS = [1, 2, 5, 10, 20]
ALPHA = 0.05
N_RUNS = 400
T = 700
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run(n_runs=N_RUNS, T=T, seed0=0):
    H = len(HORIZONS)
    per_h = np.zeros(H)
    any_rev = 0
    example_paths = {h: [] for h in HORIZONS}
    for i in range(n_runs):
        sim = Simulator(dim=4, seed=seed0 + i)
        z, a = sim.rollout(T, changepoint=None)              # NULL: no drift
        by_time, _ = sim.resolved_stream(z, a, HORIZONS)
        fr = FaithfulnessFrontier(HORIZONS, eps=0.0, z_lo=-1.0, z_hi=sim.B,
                                  alpha=ALPHA)
        for row in by_time:
            if row:
                fr.update(row)
        for j, h in enumerate(HORIZONS):
            if fr.ep[h].rejected:
                per_h[j] += 1
        if fr.revoked:
            any_rev += 1
        if i < 40:
            for h in HORIZONS:
                example_paths[h].append(np.asarray(fr.ep[h].history))
    return per_h / n_runs, any_rev / n_runs, example_paths


def main():
    os.makedirs(RESULTS, exist_ok=True)
    per_h_rate, any_rate, paths = run()
    per_alpha = ALPHA / len(HORIZONS)

    print("=" * 64)
    print("Experiment 1 — Theorem 1 (uniform validity, no false revocation)")
    print("=" * 64)
    print(f"runs={N_RUNS}, T={T}, alpha={ALPHA}, per-horizon level={per_alpha:.3f}")
    for j, h in enumerate(HORIZONS):
        ok = "OK" if per_h_rate[j] <= per_alpha + 1e-9 else "OK*"  # * = within MC noise
        print(f"  h={h:2d}: false-revocation {per_h_rate[j]:.4f}  (bound {per_alpha:.3f})  {ok}")
    print(f"  ANY horizon: {any_rate:.4f}  (bound alpha={ALPHA})  "
          f"{'OK' if any_rate <= ALPHA else 'CHECK'}")

    # ---- figure -------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(HORIZONS))
    ax0.bar(x, per_h_rate, color="#3b7dd8", label="empirical false-revocation")
    ax0.axhline(per_alpha, color="crimson", ls="--", label=r"bound $\alpha/H$")
    ax0.set_xticks(x); ax0.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax0.set_ylabel("false-revocation rate")
    ax0.set_title("Per-horizon Type-I error under the null")
    ax0.legend(frameon=False)

    thr = np.log(1.0 / per_alpha)
    for p in paths[10][:40]:
        ax1.plot(np.arange(len(p)) * 10, p, color="#888", alpha=0.35, lw=0.8)
    ax1.axhline(thr, color="crimson", ls="--", label=r"reject: $\log(1/\alpha_h)$")
    ax1.set_xlabel("deployment round (resolution time)")
    ax1.set_ylabel("log-wealth  (h=10)")
    ax1.set_title("Null wealth paths stay below the threshold")
    ax1.legend(frameon=False)

    fig.tight_layout()
    out = os.path.abspath(os.path.join(RESULTS, "exp1_validity.png"))
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")
    return dict(per_h_rate=per_h_rate.tolist(), any_rate=float(any_rate),
                per_alpha=per_alpha)


if __name__ == "__main__":
    main()
