"""GPU phase — ATLAS monitoring a learned NEURAL latent world model.

Swaps the linear latent WM for a conv-autoencoder + latent-dynamics net
(``NeuralLatentWM``, Dreamer/JEPA-lite) and runs the same ATLAS frontier under a
real distribution shift. Auto-uses CUDA if available.

    # full run (GPU recommended)
    python -m experiments.gpu.run_neural --dataset mnist
    python -m experiments.gpu.run_neural --dataset kth

    # quick CPU smoke test
    python -m experiments.gpu.run_neural --dataset mnist --smoke

See experiments/gpu/README.md for Colab / GPU-host instructions. A full DreamerV3 or
JEPA model drops in behind the same ``encode/predict/excess_stream`` interface.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.frontier import FaithfulnessFrontier
from experiments.gpu.neural_wm import NeuralLatentWM

FIG = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
HORIZONS = [1, 2, 3]
ALPHA = 0.05


def _mnist(smoke):
    from experiments.real_data.datasets import (load_moving_mnist,
                                                moving_mnist_speed_shift)
    n = 300 if smoke else 900
    d = load_moving_mnist(n=n)[:, :, ::2, ::2]           # 32x32
    a, b, c = (150, 200, 250) if smoke else (500, 700, 800)
    train, val = list(d[:a]), list(d[a:b])
    dep_id = list(d[b:c])
    dep_od = [moving_mnist_speed_shift(s[None], 2)[0] for s in d[c:c + (c - b)]]
    return train, val, dep_id, dep_od, "native", "2x-speed", "speed shift"


def _kth(smoke):
    from experiments.real_data.datasets import load_kth
    mv = 30 if smoke else 100
    # restrict to one recording scenario (outdoors, d1) so appearance is homogeneous
    # and the shift is purely the gait dynamics (walking vs running)
    walking = load_kth("walking", size=32, max_videos=mv, seq_len=40, scenario=1)
    running = load_kth("running", size=32, max_videos=mv, seq_len=40, scenario=1)
    n = len(walking)
    train = walking[:n // 2]
    rest = walking[n // 2:]
    # shuffle in-distribution deploy/val clips so cross-subject appearance
    # differences don't form a spurious trend (KTH is not perfectly stationary)
    np.random.default_rng(0).shuffle(rest)
    val = rest[:max(3, len(rest) // 2)]
    dep_id = rest[max(3, len(rest) // 2):]
    return train, val, dep_id, running, "walking", "running", "walking->running"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "kth"], default="mnist")
    ap.add_argument("--model", choices=["neural", "jepa"], default="neural",
                    help="neural = conv-AE + dynamics; jepa = frozen ResNet + dynamics")
    ap.add_argument("--smoke", action="store_true", help="tiny config for CPU test")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    train, val, dep_id, dep_od, lab_id, lab_od, shift = (
        _mnist(args.smoke) if args.dataset == "mnist" else _kth(args.smoke))

    ld = 16 if args.smoke else 64
    ep = 3 if args.smoke else 35
    print(f"=== ATLAS + {args.model} WM on {args.dataset} "
          f"({'smoke' if args.smoke else 'full'}) ===")
    if args.model == "jepa":
        from experiments.gpu.jepa_wm import JEPALatentWM
        wm = JEPALatentWM(latent_dim=ld, epochs_dyn=ep).fit(train, horizons=HORIZONS)
    else:
        wm = NeuralLatentWM(latent_dim=ld, img_size=32, epochs_ae=ep,
                            epochs_dyn=ep).fit(train, horizons=HORIZONS)
    print("device:", wm.dev)

    val_ex = wm.raw_excess(val, HORIZONS, B=8.0)
    od_ex = wm.raw_excess(dep_od, HORIZONS, B=8.0)
    if args.dataset == "mnist":
        # near-stationary WM: a tight window-level tolerance suffices
        eps = {h: float(val_ex[h].mean() + 0.1 * val_ex[h].std()) for h in HORIZONS}
    else:
        # real video is non-stationary across subjects: the window std badly
        # underestimates clip-to-clip variation, so calibrate eps on the spread of
        # PER-CLIP mean excess (95th percentile) — no in-distribution clip should
        # then trip the detector.
        per_clip = {h: [] for h in HORIZONS}
        for c in val:
            ce = wm.raw_excess([c], HORIZONS, B=8.0)
            for h in HORIZONS:
                per_clip[h].append(ce[h].mean())
        eps = {h: float(np.quantile(per_clip[h], 0.95)) for h in HORIZONS}
    for h in HORIZONS:
        print(f"  h={h}: {lab_id}(val) {val_ex[h].mean():+.2f} | "
              f"{lab_od} {od_ex[h].mean():+.2f}  eps={eps[h]:.2f}")

    fr = FaithfulnessFrontier(HORIZONS, eps=eps, z_lo=-1.0, z_hi=8.0, alpha=ALPHA)
    hstar, wealth = [], {h: [] for h in HORIZONS}
    t, cp = 0, None
    for kind, c in [("id", c) for c in dep_id] + [("od", c) for c in dep_od]:
        if kind == "od" and cp is None:
            cp = t
        for row in wm.excess_stream(c, HORIZONS):
            if row:
                fr.update(row); t += 1; hstar.append(fr.h_star)
                for h in HORIZONS:
                    wealth[h].append(fr.ep[h].log_wealth)
    hstar = np.array(hstar)
    print(f"h*: {lab_id} median={int(np.median(hstar[:cp]))}  "
          f"{lab_od} final={hstar[-1]}  revoked={sorted(fr.revoked)}")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    thr = np.log(1.0 / fr.per_alpha)
    for h in HORIZONS:
        ax0.plot(wealth[h], lw=1.4, label=f"h={h}")
    ax0.axhline(thr, color="crimson", ls="--", lw=1, label=r"reject $\log(1/\alpha_h)$")
    ax0.axvline(cp, color="k", ls=":", lw=1.2, label=shift)
    ax0.set_ylabel("log-wealth"); ax0.legend(frameon=False, ncol=3, fontsize=8)
    ax0.set_title(f"ATLAS + neural latent WM — {args.dataset} ({shift})")
    ax1.plot(hstar, color="#7a3b9d", lw=2.0, label=r"$h^*(t)$ ATLAS (neural WM)")
    ax1.axvline(cp, color="k", ls=":", lw=1.2)
    ax1.set_xlabel("deployment update"); ax1.set_ylabel(r"$h^*(t)$")
    ax1.legend(frameon=False); ax1.set_title(r"$h^*(t)$ collapse — learned neural WM")
    fig.tight_layout()
    tag = "smoke" if args.smoke else "full"
    mtag = "" if args.model == "neural" else f"_{args.model}"
    out = os.path.join(FIG, f"neural_{args.dataset}{mtag}_{tag}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"figure -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
