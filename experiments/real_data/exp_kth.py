"""Real-data experiment 2 — KTH Actions, walking -> running dynamics shift.

A genuine real-world shift: a latent world model is fit on *walking* videos and
deployed; the subject then switches to *running* (faster leg dynamics the
walking-trained model cannot track). ATLAS's tolerance is calibrated on held-out
walking clips; the frontier ``h*(t)`` holds while walking and collapses once running
begins, while an offline (frozen) trust horizon stays blind.

Run: ``python -m experiments.real_data.exp_kth``
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.frontier import FaithfulnessFrontier
from experiments.real_data.datasets import load_kth, sample_frames_figure
from experiments.real_data.latent_wm import LatentLinearWM

HORIZONS = [1, 2, 3]
ALPHA = 0.05
LATENT = 32
SIZE = 32
SEQ = 40
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    print("=" * 66)
    print("Real-data Exp 2 — KTH Actions (walking -> running dynamics shift)")
    print("=" * 66)
    walking = load_kth("walking", size=SIZE, max_videos=60, seq_len=SEQ)
    running = load_kth("running", size=SIZE, max_videos=40, seq_len=SEQ)
    print(f"loaded {len(walking)} walking, {len(running)} running sequences")

    n = len(walking)
    train = walking[: n // 2]
    val = walking[n // 2: n // 2 + max(4, n // 4)]
    deploy_id = walking[n // 2 + max(4, n // 4):]
    deploy_od = running
    # shuffle the in-distribution deploy clips so subject-to-subject appearance
    # differences don't form a sustained trend (the crude linear model is not
    # perfectly stationary across subjects); keeps the walking null well-behaved.
    rng = np.random.default_rng(0)
    rng.shuffle(deploy_id)

    wm = LatentLinearWM(latent_dim=LATENT).fit(train)
    print(f"fit latent WM: dim={LATENT}, spectral radius(A)="
          f"{np.max(np.abs(np.linalg.eigvals(wm.A))):.3f}")

    # calibrate on the clipped statistic the e-process actually consumes
    val_ex = wm.raw_excess(val, HORIZONS, B=8.0)
    od_ex = wm.raw_excess(deploy_od, HORIZONS, B=8.0)
    eps = {h: float(val_ex[h].mean() + 0.5 * val_ex[h].std()) for h in HORIZONS}
    print("calibrated eps_h (clipped walking mean + 0.5 std):",
          {h: round(eps[h], 2) for h in HORIZONS})
    for h in HORIZONS:
        print(f"  h={h}: excess  walking(val) mean={val_ex[h].mean():+.2f}  "
              f"running mean={od_ex[h].mean():+.2f}")

    fr = FaithfulnessFrontier(HORIZONS, eps=eps, z_lo=-1.0, z_hi=8.0, alpha=ALPHA)
    hstar, wealth = [], {h: [] for h in HORIZONS}
    round_t, cp_round = 0, None
    clips = [("id", c) for c in deploy_id] + [("od", c) for c in deploy_od]
    for kind, clip in clips:
        if kind == "od" and cp_round is None:
            cp_round = round_t
        for row in wm.excess_stream(clip, HORIZONS):
            if row:
                fr.update(row)
                round_t += 1
                hstar.append(fr.h_star)
                for h in HORIZONS:
                    wealth[h].append(fr.ep[h].log_wealth)
    hstar = np.array(hstar)
    print(f"\nchangepoint at update {cp_round}")
    print(f"h*: walking median={int(np.median(hstar[:cp_round]))}  "
          f"running final={hstar[-1]}")
    print(f"revoked horizons after shift: {sorted(fr.revoked)}")

    # ---- figures ------------------------------------------------------------
    sample_frames_figure(deploy_id, deploy_od,
                         os.path.join(FIG, "kth_frames.png"),
                         "KTH walking", "KTH running")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    thr = np.log(1.0 / fr.per_alpha)
    for h in HORIZONS:
        ax0.plot(wealth[h], lw=1.4, label=f"h={h}")
    ax0.axhline(thr, color="crimson", ls="--", lw=1, label=r"reject $\log(1/\alpha_h)$")
    ax0.axvline(cp_round, color="k", ls=":", lw=1.2, label="walking->running")
    ax0.set_ylabel("log-wealth"); ax0.legend(frameon=False, ncol=3, fontsize=8)
    ax0.set_title("ATLAS per-horizon wealth — KTH (walking -> running)")

    ax1.plot(hstar, color="#1b8a5a", lw=2.0, label=r"$h^*(t)$ ATLAS")
    ax1.axhline(int(np.median(hstar[:cp_round])), color="#999", ls=":", lw=1.5,
                label="offline trust horizon (frozen, blind)")
    ax1.axvline(cp_round, color="k", ls=":", lw=1.2)
    ax1.set_xlabel("deployment update"); ax1.set_ylabel(r"$h^*(t)$")
    ax1.legend(frameon=False)
    ax1.set_title(r"$h^*(t)$ collapses when the subject starts running")
    fig.tight_layout()
    out = os.path.join(FIG, "kth_frontier.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"figures -> {os.path.abspath(out)}, kth_frames.png")
    return dict(eps=eps, hstar_pre=int(np.median(hstar[:cp_round])),
                hstar_final=int(hstar[-1]), revoked=sorted(fr.revoked),
                cp_round=cp_round)


if __name__ == "__main__":
    main()
