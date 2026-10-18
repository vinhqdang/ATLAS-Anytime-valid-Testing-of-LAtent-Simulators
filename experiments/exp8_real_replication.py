"""Experiment 8 -- replicated real-data study with competitor monitors.

The single-run real-video demonstrations of the original study are replaced here by a
replicated protocol on three real shifts (Moving MNIST speed, KTH walking->running, KTH
outdoor->indoor). Each repetition draws a fresh random split (train / calibration /
in-distribution deployment / shifted deployment), refits the latent world model,
calibrates the tolerance on the calibration split, and then runs

  * ATLAS (frontier over several horizons, level alpha, per-horizon alpha/H), and
  * competitor monitors on the same one-step statistic stream, with sensitivities tuned on
    separate *calibration repetitions* to a 5% false-alarm rate over the in-distribution
    deployment length ``L`` (CUSUM, ADWIN, KSWIN, conformal martingale).

For each monitor we report (i) the false-alarm rate over ``L`` in-distribution clips and over
``3L`` (in-distribution clips resampled with replacement -- longer monitoring), and (ii) the
detection rate and delay (in clips) after the shift. Means carry bootstrap 95% CIs over
repetitions.

Run: ``python -m experiments.exp8_real_replication [mnist|kth|domain]``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from atlas.fastsim import log_wealth_path, first_cross, lam_grid
from experiments.baselines import CUSUM, ADWIN, KSWIN, ConformalMartingale
from experiments.real_data.latent_wm import LatentLinearWM

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
Z_LO, Z_HI = -1.0, 8.0
LAMS = lam_grid(Z_LO, Z_HI, 20)
REPS = int(os.environ.get("REPS", 30))
EPS_MULTS = [0.0, 0.25, 0.5, 1.0]


# ---------------------------------------------------------------- streams


def clip_streams(wm, clips, horizons):
    """Per-horizon score streams with the clip index of every update."""
    zs = {h: [] for h in horizons}
    ci = {h: [] for h in horizons}
    for c, clip in enumerate(clips):
        by_time = wm.excess_stream(clip, horizons, B=Z_HI)
        for row in by_time:
            for h, z in row.items():
                zs[h].append(z)
                ci[h].append(c)
    return {h: (np.array(zs[h]), np.array(ci[h])) for h in horizons}


def atlas_alarm(zs, eps, horizons):
    """First clip index at which any horizon is revoked (None if never)."""
    H = len(horizons)
    best = None
    for h in horizons:
        z, c = zs[h]
        if len(z) == 0:
            continue
        lw = log_wealth_path(z, eps[h], LAMS)
        i = first_cross(lw, np.log(H / ALPHA))
        if i is not None and (best is None or c[i] < best):
            best = int(c[i])
    return best


def monitor_alarm(mon, z, c):
    for v, ci in zip(z, c):
        if mon.step(float(v)):
            return int(ci)
    return None


# ---------------------------------------------------------------- datasets


def split_mnist(rng):
    from experiments.real_data.datasets import load_moving_mnist, moving_mnist_speed_shift
    global _MNIST
    if "_MNIST" not in globals():
        _MNIST = load_moving_mnist(n=4000)[:, :, ::2, ::2]
    idx = rng.permutation(len(_MNIST))
    train = list(_MNIST[idx[:500]])
    cal = list(_MNIST[idx[500:700]])
    dep_id = list(_MNIST[idx[700:800]])
    dep_od = [moving_mnist_speed_shift(s[None], 2)[0] for s in _MNIST[idx[800:900]]]
    return train, cal, dep_id, dep_od


def make_kth(kind):
    from experiments.real_data.datasets import load_kth
    if kind == "kth":
        idp = load_kth("walking", size=32, max_videos=60, seq_len=40)
        odp = load_kth("running", size=32, max_videos=40, seq_len=40)
    else:
        idp = load_kth("walking", size=32, max_videos=60, seq_len=40, scenario=1)
        odp = load_kth("walking", size=32, max_videos=60, seq_len=40, scenario=4)

    def split(rng):
        ids = [idp[i] for i in rng.permutation(len(idp))]
        n = len(ids)
        train = ids[: n // 2]
        cal = ids[n // 2: n // 2 + max(3, n // 4)]
        dep = ids[n // 2 + max(3, n // 4):]
        od = [odp[i] for i in rng.permutation(len(odp))]
        return train, cal, dep, od
    return split


CONFIG = {
    "mnist": dict(horizons=[1, 2, 3, 5, 8], latent=20, split=split_mnist),
    "kth": dict(horizons=[1, 2, 3], latent=32, split=None),
    "domain": dict(horizons=[1, 2, 3], latent=32, split=None),
}


# ---------------------------------------------------------------- tuning


def tune(name, factory, grid, null_streams, L_clips):
    """Pick the sensitivity whose false-alarm rate over L clips is closest to ALPHA."""
    best, gap = grid[0], 9
    for s in grid:
        fa = np.mean([monitor_alarm(factory(s), z, c) is not None
                      for z, c in null_streams])
        if abs(fa - ALPHA) < gap:
            best, gap = s, abs(fa - ALPHA)
    return best


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- main


def run(kind):
    cfg = CONFIG[kind]
    horizons, latent = cfg["horizons"], cfg["latent"]
    split = cfg["split"] or make_kth(kind)
    results = {m: dict(fa_L=[], fa_3L=[], det=[], delay=[]) for m in
               [f"ATLAS(eps+{k}sd)" for k in EPS_MULTS]}
    base_names = ["CUSUM", "ADWIN", "KSWIN", "Conformal MG"]
    for b in base_names:
        results[b] = dict(fa_L=[], fa_3L=[], det=[], delay=[])

    # ---- gather repetition data (h=1 raw streams + per-horizon ATLAS streams)
    reps = []
    for r in range(REPS * 2):          # first REPS for evaluation, next REPS to tune baselines
        rng = np.random.default_rng(1000 + r)
        train, cal, dep_id, dep_od = split(rng)
        wm = LatentLinearWM(latent_dim=latent).fit(train)
        cal_ex = wm.raw_excess(cal, horizons, B=Z_HI)
        s_id = clip_streams(wm, dep_id, horizons)
        s_od = clip_streams(wm, dep_od, horizons)
        s_cal1 = clip_streams(wm, cal, [1])[1]
        # 3L in-distribution stream: resample clips with replacement
        pick = rng.choice(len(dep_id), 3 * len(dep_id))
        s_id3 = clip_streams(wm, [dep_id[i] for i in pick], horizons)
        reps.append(dict(cal_ex=cal_ex, s_id=s_id, s_od=s_od, s_id3=s_id3, s_cal1=s_cal1,
                         L=len(dep_id)))
        print(f"  rep {r+1}/{2*REPS}", end="\r")
    print()

    # ---- tune baselines on the second half (null streams only)
    ref = np.concatenate([rp["s_cal1"][0] for rp in reps[REPS:]])
    m0, sd0 = float(ref.mean()), float(ref.std())
    factories = {
        "CUSUM": (lambda s: CUSUM(m0, 0.25 * sd0, s), np.linspace(0.5, 60, 24)),
        "ADWIN": (lambda s: ADWIN(delta=s), np.logspace(-8, -0.5, 14)),
        "KSWIN": (lambda s: KSWIN(alpha=s), np.logspace(-5, -1.3, 14)),
        "Conformal MG": (lambda s: ConformalMartingale(ref, alpha=ALPHA), [None]),
    }
    tuned = {}
    for name, (fac, grid) in factories.items():
        nulls = [(rp["s_id"][1][0], rp["s_id"][1][1]) for rp in reps[REPS:]]
        tuned[name] = tune(name, fac, grid, nulls, None) if name != "Conformal MG" else None
    print("tuned:", {k: (None if v is None else round(float(v), 5)) for k, v in tuned.items()})

    # ---- evaluate on the first half
    for rp in reps[:REPS]:
        L = rp["L"]
        for k in EPS_MULTS:
            eps = {h: float(rp["cal_ex"][h].mean() + k * rp["cal_ex"][h].std())
                   for h in horizons}
            nm = f"ATLAS(eps+{k}sd)"
            a_id = atlas_alarm(rp["s_id"], eps, horizons)
            a_id3 = atlas_alarm(rp["s_id3"], eps, horizons)
            a_od = atlas_alarm(rp["s_od"], eps, horizons)
            results[nm]["fa_L"].append(a_id is not None)
            results[nm]["fa_3L"].append(a_id3 is not None)
            results[nm]["det"].append(a_od is not None)
            results[nm]["delay"].append(float(a_od) if a_od is not None else np.nan)
        for name, (fac, _) in factories.items():
            mk = lambda: fac(tuned[name])
            z1, c1 = rp["s_id"][1]
            z3, c3 = rp["s_id3"][1]
            zo, co = rp["s_od"][1]
            a = monitor_alarm(mk(), z1, c1)
            a3 = monitor_alarm(mk(), z3, c3)
            ao = monitor_alarm(mk(), zo, co)
            results[name]["fa_L"].append(a is not None)
            results[name]["fa_3L"].append(a3 is not None)
            results[name]["det"].append(ao is not None)
            results[name]["delay"].append(float(ao) if ao is not None else np.nan)

    summary = {}
    print(f"\n{kind}: {REPS} repetitions (alarm clip index counted from the shift)")
    print(f"{'monitor':<20}{'FA@L':>16}{'FA@3L':>16}{'detect':>16}{'delay (clips)':>22}")
    for m, d in results.items():
        row = {k: boot_ci(np.array(v, float)) for k, v in d.items()}
        summary[m] = row
        f = lambda t: f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"
        print(f"{m:<20}{f(row['fa_L']):>16}{f(row['fa_3L']):>16}{f(row['det']):>16}"
              f"{row['delay'][0]:>10.1f} [{row['delay'][1]:.1f},{row['delay'][2]:.1f}]")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"exp8_{kind}.json"), "w") as f:
        json.dump(dict(reps=REPS, summary=summary,
                       tuned={k: (None if v is None else float(v)) for k, v in tuned.items()}),
                  f, indent=2)


if __name__ == "__main__":
    which = sys.argv[1:] or ["mnist", "kth", "domain"]
    for k in which:
        run(k)
