"""Shared runner for KTH real-data ATLAS experiments.

A latent world model is fit on the in-distribution regime; ATLAS's tolerance is
calibrated on held-out in-distribution clips (on the same clipped statistic the
e-process consumes); deployment runs in-distribution and then switches to the
shifted regime. Produces a sample-frame figure and a two-panel frontier figure.
Used for two distinct real-world shift types:

  * exp_kth.py         — walking -> running   (a gait *dynamics* shift)
  * exp_kth_domain.py  — outdoor -> indoor    (a *domain / appearance* shift)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.frontier import FaithfulnessFrontier
from experiments.real_data.datasets import sample_frames_figure
from experiments.real_data.latent_wm import LatentLinearWM

FIG = os.path.join(os.path.dirname(__file__), "..", "..", "figures")


def run_shift(id_seqs, od_seqs, *, horizons, latent, eps_mult, alpha,
              label_id, label_od, shift_label, tag, title):
    """Run one KTH shift experiment and save figures. Returns a result dict."""
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(0)
    id_seqs = list(id_seqs)
    rng.shuffle(id_seqs)                       # de-trend subject/appearance order
    n = len(id_seqs)
    train = id_seqs[: n // 2]
    val = id_seqs[n // 2: n // 2 + max(3, n // 4)]
    deploy_id = id_seqs[n // 2 + max(3, n // 4):]
    deploy_od = list(od_seqs)

    wm = LatentLinearWM(latent_dim=latent).fit(train)
    print(f"fit latent WM: dim={latent}, spectral radius(A)="
          f"{np.max(np.abs(np.linalg.eigvals(wm.A))):.3f}")

    val_ex = wm.raw_excess(val, horizons, B=8.0)
    od_ex = wm.raw_excess(deploy_od, horizons, B=8.0)
    eps = {h: float(val_ex[h].mean() + eps_mult * val_ex[h].std()) for h in horizons}
    print(f"calibrated eps_h ({label_id} mean + {eps_mult} std):",
          {h: round(eps[h], 2) for h in horizons})
    for h in horizons:
        print(f"  h={h}: excess  {label_id}(val) mean={val_ex[h].mean():+.2f}  "
              f"{label_od} mean={od_ex[h].mean():+.2f}")

    fr = FaithfulnessFrontier(horizons, eps=eps, z_lo=-1.0, z_hi=8.0, alpha=alpha)
    hstar, wealth = [], {h: [] for h in horizons}
    round_t, cp_round = 0, None
    clips = [("id", c) for c in deploy_id] + [("od", c) for c in deploy_od]
    for kind, clip in clips:
        if kind == "od" and cp_round is None:
            cp_round = round_t
        for row in wm.excess_stream(clip, horizons):
            if row:
                fr.update(row)
                round_t += 1
                hstar.append(fr.h_star)
                for h in horizons:
                    wealth[h].append(fr.ep[h].log_wealth)
    hstar = np.array(hstar)
    print(f"\nchangepoint at update {cp_round}")
    print(f"h*: {label_id} median={int(np.median(hstar[:cp_round]))}  "
          f"{label_od} final={hstar[-1]}")
    print(f"revoked horizons after shift: {sorted(fr.revoked)}")

    # ---- figures ------------------------------------------------------------
    sample_frames_figure(deploy_id, deploy_od,
                         os.path.join(FIG, f"{tag}_frames.png"),
                         f"KTH {label_id}", f"KTH {label_od}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    thr = np.log(1.0 / fr.per_alpha)
    for h in horizons:
        ax0.plot(wealth[h], lw=1.4, label=f"h={h}")
    ax0.axhline(thr, color="crimson", ls="--", lw=1, label=r"reject $\log(1/\alpha_h)$")
    ax0.axvline(cp_round, color="k", ls=":", lw=1.2, label=shift_label)
    ax0.set_ylabel("log-wealth"); ax0.legend(frameon=False, ncol=3, fontsize=8)
    ax0.set_title(title)

    ax1.plot(hstar, color="#1b8a5a", lw=2.0, label=r"$h^*(t)$ ATLAS")
    ax1.axhline(int(np.median(hstar[:cp_round])), color="#999", ls=":", lw=1.5,
                label="offline trust horizon (frozen, blind)")
    ax1.axvline(cp_round, color="k", ls=":", lw=1.2)
    ax1.set_xlabel("deployment update"); ax1.set_ylabel(r"$h^*(t)$")
    ax1.legend(frameon=False)
    ax1.set_title(rf"$h^*(t)$ collapses at the {shift_label} shift")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{tag}_frontier.png"), dpi=130); plt.close(fig)
    print(f"figures -> {tag}_frames.png, {tag}_frontier.png")
    return dict(eps=eps, hstar_pre=int(np.median(hstar[:cp_round])),
                hstar_final=int(hstar[-1]), revoked=sorted(fr.revoked),
                cp_round=cp_round)
