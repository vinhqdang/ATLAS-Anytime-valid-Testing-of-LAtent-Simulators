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
    return train, val, dep_id, dep_od, "native", "2x-speed", "speed shift", 32, None


def _kitti(smoke):
    """KITTI ego-view driving -- an ego-motion (dynamics) shift.

    A world model predicting its own future observations while driving is trained on
    nominal-speed ego-motion and deployed on faster ego-motion (2x frame skip, i.e.
    the vehicle covers twice the distance per step). Crucially the *scenes are held
    fixed*: the deploy-native and deploy-fast streams are the same held-out road
    segments, so the appearance is identical and only the dynamics change -- an
    unambiguous world-model faithfulness breach, the driving analogue of the Moving
    MNIST speed shift. Splits are disjoint for fitting, covariance calibration, and
    tolerance calibration, so no statistic is evaluated in-sample.
    """
    from experiments.real_data.datasets import load_kitti
    ids = ["0011", "0009"] if smoke else ["0011", "0005", "0001", "0009", "0013",
                                          "0027", "0015"]
    segs = []
    for d in ids:
        segs += load_kitti(d, size=64, seq_len=40, stride=20)   # overlapping segments
    np.random.default_rng(0).shuffle(segs)
    SPEED = 3                                        # deploy ego-speed multiplier
    native = [s[:len(s) // SPEED] for s in segs]     # nominal ego-speed
    fast = [s[::SPEED] for s in segs]               # SPEEDx ego-speed, same scene start
    n = len(native)
    a, b, c = int(0.45 * n), int(0.65 * n), int(0.82 * n)
    train = native[:a]
    cov = native[a:b]                               # covariance calibration (disjoint)
    val = native[b:c]                               # tolerance calibration (disjoint)
    dep_id = native[c:]                             # held-out native deploy (validity)
    dep_od = fast[c:]                               # same held-out scenes, SPEEDx speed
    return (train, val, dep_id, dep_od, "native", f"{SPEED}x-speed",
            "ego-speed shift", 64, cov)


def _kth(smoke):
    from experiments.real_data.datasets import load_kth
    mv = 30 if smoke else 100
    # homogeneous in-distribution walking (one scenario) -> a tight null; running
    # (the shift) uses all scenarios for ample detection data
    walking = load_kth("walking", size=32, max_videos=mv, seq_len=40, scenario=1)
    running = load_kth("running", size=32, max_videos=mv, seq_len=40)
    n = len(walking)
    train = walking[:n // 2]
    rest = walking[n // 2:]
    # shuffle in-distribution deploy/val clips so cross-subject appearance
    # differences don't form a spurious trend (KTH is not perfectly stationary)
    np.random.default_rng(0).shuffle(rest)
    val = rest[:max(3, len(rest) // 2)]
    dep_id = rest[max(3, len(rest) // 2):]
    return train, val, dep_id, running, "walking", "running", "walking->running", 32, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "kth", "kitti"], default="mnist")
    ap.add_argument("--model", choices=["neural", "jepa"], default="neural",
                    help="neural = conv-AE + dynamics; jepa = frozen ResNet + dynamics")
    ap.add_argument("--smoke", action="store_true", help="tiny config for CPU test")
    ap.add_argument("--recalib-cov", action="store_true",
                    help="recalibrate predictive covariance on held-out val residuals "
                         "(centres in-distribution excess near zero; needed for "
                         "high-dim latents where in-sample residuals are optimistic)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    loaders = {"mnist": _mnist, "kth": _kth, "kitti": _kitti}
    train, val, dep_id, dep_od, lab_id, lab_od, shift, img_size, cov = loaders[args.dataset](args.smoke)

    ld = 16 if args.smoke else 64
    ep = 3 if args.smoke else 35
    print(f"=== ATLAS + {args.model} WM on {args.dataset} "
          f"({'smoke' if args.smoke else 'full'}) ===")
    if args.model == "jepa":
        from experiments.gpu.jepa_wm import JEPALatentWM
        wm = JEPALatentWM(latent_dim=ld, epochs_dyn=ep).fit(train, horizons=HORIZONS)
    else:
        wm = NeuralLatentWM(latent_dim=ld, img_size=img_size, epochs_ae=ep,
                            epochs_dyn=ep).fit(train, horizons=HORIZONS)
    if args.recalib_cov:
        cov_set = cov if cov else val
        wm.recalibrate(cov_set, HORIZONS)
        print(f"recalibrated predictive covariance on held-out "
              f"{'cov' if cov else 'val'} residuals ({len(cov_set)} clips)")
    print("device:", wm.dev)

    val_ex = wm.raw_excess(val, HORIZONS, B=8.0)
    od_ex = wm.raw_excess(dep_od, HORIZONS, B=8.0)
    if args.dataset == "mnist":
        # near-stationary WM: a tight window-level tolerance suffices
        eps = {h: float(val_ex[h].mean() + 0.1 * val_ex[h].std()) for h in HORIZONS}
    elif args.dataset == "kitti":
        # driving excess is heavy-tailed at the round level; calibrate the tolerance
        # on the 90th percentile of the PER-ROUND held-out native excess so the
        # in-distribution deploy stream stays a valid null (no false revocation).
        eps = {h: float(np.quantile(val_ex[h], 0.90)) for h in HORIZONS}
    else:
        # real video is non-stationary across subjects: the window std badly
        # underestimates clip-to-clip variation, so calibrate eps on the spread of
        # PER-CLIP mean excess (90th percentile) — few in-distribution clips should
        # then trip the detector.
        per_clip = {h: [] for h in HORIZONS}
        for c in val:
            ce = wm.raw_excess([c], HORIZONS, B=8.0)
            for h in HORIZONS:
                per_clip[h].append(ce[h].mean())
        eps = {h: float(np.quantile(per_clip[h], 0.90)) for h in HORIZONS}
    for h in HORIZONS:
        print(f"  h={h}: {lab_id}(val) {val_ex[h].mean():+.2f} | "
              f"{lab_od} {od_ex[h].mean():+.2f}  eps={eps[h]:.2f}")

    # dump per-round excess streams (per clip) for offline eps calibration / analysis
    def _clip_streams(clips):
        out = {h: [] for h in HORIZONS}
        for c in clips:
            for row in wm.excess_stream(c, HORIZONS):
                for h in HORIZONS:
                    if h in row:
                        out[h].append(row[h])
        return {h: np.array(v) for h, v in out.items()}
    dump = dict(dataset=args.dataset, model=args.model)
    for name, clips in [("val", val), ("dep_id", dep_id), ("dep_od", dep_od)]:
        s = _clip_streams(clips)
        for h in HORIZONS:
            dump[f"{name}_h{h}"] = s[h]
    np.savez(os.path.join(FIG, f"dump_{args.dataset}_{args.model}.npz"), **dump)

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
    ax0.set_title(f"ATLAS + {args.model} latent WM — {args.dataset} ({shift})")
    ax1.plot(hstar, color="#7a3b9d", lw=2.0, label=rf"$h^*(t)$ ATLAS ({args.model} WM)")
    ax1.axvline(cp, color="k", ls=":", lw=1.2)
    ax1.set_xlabel("deployment update"); ax1.set_ylabel(r"$h^*(t)$")
    ax1.legend(frameon=False); ax1.set_title(rf"$h^*(t)$ collapse — learned {args.model} WM")
    fig.tight_layout()
    tag = "smoke" if args.smoke else "full"
    mtag = "" if args.model == "neural" else f"_{args.model}"
    out = os.path.join(FIG, f"neural_{args.dataset}{mtag}_{tag}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"figure -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
