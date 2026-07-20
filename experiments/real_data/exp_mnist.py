"""Real-data experiment 1 — Moving MNIST, frame-rate dynamics shift.

A latent linear world model is fit on native-rate Moving MNIST. At deployment the
stream runs in-distribution (native rate) and then switches to a 2x-speed regime
(temporal subsampling) — a controllable dynamics shift the native-rate model cannot
track. ATLAS's tolerance ``eps_h`` is calibrated on held-out in-distribution clips,
then the frontier ``h*(t)`` is monitored: it stays high while in-distribution and
collapses after the shift, while an offline (frozen) trust horizon stays blind.

Run: ``python -m experiments.real_data.exp_mnist``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.frontier import FaithfulnessFrontier
from experiments.real_data.datasets import (
    load_moving_mnist, moving_mnist_speed_shift, sample_frames_figure)
from experiments.real_data.latent_wm import LatentLinearWM

HORIZONS = [1, 2, 3, 5, 8]
ALPHA = 0.05
LATENT = 20
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def downscale(frames, factor=2):
    return frames[:, :, ::factor, ::factor]


def main():
    os.makedirs(FIG, exist_ok=True)
    print("=" * 66)
    print("Real-data Exp 1 — Moving MNIST (native -> 2x speed dynamics shift)")
    print("=" * 66)
    data = load_moving_mnist(n=900)                 # (900, 20, 64, 64)
    data = downscale(data, 2)                        # (900, 20, 32, 32)
    train = list(data[:500])
    val = list(data[500:700])                        # in-distribution calibration
    test_id = list(data[700:800])                    # in-distribution deployment
    test_od = [moving_mnist_speed_shift(s[None], 2)[0] for s in data[800:900]]  # 2x

    wm = LatentLinearWM(latent_dim=LATENT).fit(train)
    print(f"fit latent WM: dim={LATENT}, spectral radius(A)="
          f"{np.max(np.abs(np.linalg.eigvals(wm.A))):.3f}")

    # Calibrate eps_h to the in-distribution mean excess (+ small safety margin):
    # the null becomes "as calibrated as in-distribution", so the pre-shift wealth is
    # a martingale (no depletion, Ville validity) and only a real shift grows it.
    val_ex = wm.raw_excess(val, HORIZONS)
    eps = {h: float(val_ex[h].mean() + 0.1 * val_ex[h].std()) for h in HORIZONS}
    print("calibrated eps_h (in-dist mean + 0.1 std):",
          {h: round(eps[h], 2) for h in HORIZONS})

    # sanity: in-distribution vs shifted mean excess
    od_ex = wm.raw_excess(test_od, HORIZONS)
    for h in HORIZONS:
        print(f"  h={h}: excess  in-dist(val) mean={val_ex[h].mean():+.2f}  "
              f"shifted mean={od_ex[h].mean():+.2f}")

    # deployment stream: in-distribution clips, then shifted clips
    fr = FaithfulnessFrontier(HORIZONS, eps=eps, z_lo=-1.0, z_hi=8.0, alpha=ALPHA)
    hstar, wealth = [], {h: [] for h in HORIZONS}
    round_t = 0
    cp_round = None
    clips = [("id", c) for c in test_id] + [("od", c) for c in test_od]
    for i, (kind, clip) in enumerate(clips):
        if kind == "od" and cp_round is None:
            cp_round = round_t
        by_time = wm.excess_stream(clip, HORIZONS)
        for row in by_time:
            if row:
                fr.update(row)
                round_t += 1
                hstar.append(fr.h_star)
                for h in HORIZONS:
                    wealth[h].append(fr.ep[h].log_wealth)
    hstar = np.array(hstar)
    pre = hstar[:cp_round]
    post = hstar[cp_round:]
    print(f"\nchangepoint at update {cp_round}")
    print(f"h*: in-distribution median={int(np.median(pre))}  "
          f"post-shift final={hstar[-1]}")
    print(f"revoked horizons after shift: {sorted(fr.revoked)}")

    # ---- figures ------------------------------------------------------------
    sample_frames_figure(
        test_id, test_od,
        os.path.join(FIG, "mnist_frames.png"),
        "Moving MNIST (native rate)", "Moving MNIST (2x speed, shifted)")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    thr = np.log(1.0 / fr.per_alpha)
    for h in HORIZONS:
        ax0.plot(wealth[h], lw=1.4, label=f"h={h}")
    ax0.axhline(thr, color="crimson", ls="--", lw=1, label=r"reject $\log(1/\alpha_h)$")
    ax0.axvline(cp_round, color="k", ls=":", lw=1.2, label="speed shift")
    ax0.set_ylabel("log-wealth"); ax0.legend(frameon=False, ncol=3, fontsize=8)
    ax0.set_title("ATLAS per-horizon wealth — Moving MNIST (native -> 2x speed)")

    ax1.plot(hstar, color="#1b8a5a", lw=2.0, label=r"$h^*(t)$ ATLAS")
    ax1.axhline(int(np.median(pre)), color="#999", ls=":", lw=1.5,
                label="offline trust horizon (frozen, blind)")
    ax1.axvline(cp_round, color="k", ls=":", lw=1.2)
    ax1.set_xlabel("deployment update"); ax1.set_ylabel(r"$h^*(t)$")
    ax1.legend(frameon=False)
    ax1.set_title(r"$h^*(t)$ collapses after the real dynamics shift")
    fig.tight_layout()
    out = os.path.join(FIG, "mnist_frontier.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"figures -> {os.path.abspath(out)}, mnist_frames.png")
    return dict(eps=eps, hstar_pre=int(np.median(pre)), hstar_final=int(hstar[-1]),
                revoked=sorted(fr.revoked), cp_round=cp_round)


if __name__ == "__main__":
    main()
