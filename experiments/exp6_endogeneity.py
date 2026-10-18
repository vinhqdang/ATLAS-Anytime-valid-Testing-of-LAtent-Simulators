"""Experiment 6 -- policy-endogenous streams: validity and power.

The paper argues that exchangeability-based monitors break when the planner optimises
through the world model, because the data distribution becomes a function of the model
under test. We *measure* this on a closed-loop latent world model with state-dependent
noise (``atlas.closedloop``), where the world model is exactly faithful, so every alarm
under the null is a false alarm.

Monitors consume one of two streams of the same rounds:
  * ``raw``  -- the squared one-step prediction error, the quantity practitioners monitor;
  * ``std``  -- the model-standardised calibration excess ATLAS uses (``|y-mu|^2/(q d)-1``).
Tunable detectors (CUSUM, Page-Hinkley, ADWIN, KSWIN) are calibrated to a 5% false-alarm
rate at ``T_cal`` on an *exogenous* calibration deployment (the usual offline protocol),
then deployed under three planners: exogenous exploration, a low-noise-seeking MPC, and
a high-noise-seeking (evasive) MPC.

Part A (validity): false-alarm rate at ``T_cal`` and ``3 T_cal`` under each planner.
Part B (power): detection delay after an unmodelled drift under each planner.

Run: ``python -m experiments.exp6_endogeneity``
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atlas.closedloop import ClosedLoopWM
from atlas.eprocess import EProcess
from experiments.baselines import CUSUM, PageHinkley, ADWIN, KSWIN, ConformalMartingale

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.05
T_CAL, T_EVAL = 400, 1200
POLICIES = ["exo", "low", "high", "adv"]
N_EVAL = int(os.environ.get("N_EVAL", 200))
N_POWER = int(os.environ.get("N_POWER", 150))
DRIFT, CP, T_POW = 0.4, 400, 2800
Z_LO, Z_HI = -1.0, 8.0
WM = ClosedLoopWM(seed=1)

# ---- monitor factories ----------------------------------------------------------------


def _ref(kind):
    raw, std, _ = WM.run(4000, "exo", seed=777)
    return raw if kind == "raw" else std


def build(kind, name, sens, ref, m0, s0):
    if name == "CUSUM":
        return CUSUM(m0, 0.25 * s0, sens)
    if name == "Page-Hinkley":
        return PageHinkley(thresh=sens)
    if name == "ADWIN":
        return ADWIN(delta=sens)
    if name == "KSWIN":
        return KSWIN(alpha=sens)
    if name == "Conformal MG":
        return ConformalMartingale(ref, alpha=ALPHA)
    raise KeyError(name)


GRIDS = {
    "CUSUM": np.linspace(0.5, 40.0, 24),
    "Page-Hinkley": np.linspace(0.5, 60.0, 24),
    "ADWIN": np.logspace(-8, -0.5, 18),
    "KSWIN": np.logspace(-5, -1.3, 18),
}


class AdversaryChooser:
    """A monitor-aware planner: runs ATLAS' own e-process on the stream it generates and
    steers to the *high-noise* region when the wealth is high (to shield an incipient
    rejection) and to the low-noise region otherwise. Under the null no planner can raise
    the false-revocation probability above alpha (Theorem 1); this tests that claim against
    the most natural adaptive attempt."""

    def __init__(self):
        self.ep = EProcess(eps=0.0, z_lo=Z_LO, z_hi=Z_HI, alpha=ALPHA)
        self.k = 0

    def __call__(self, t, z, hist):
        while self.k < len(hist):
            self.ep.update(float(np.clip(hist[self.k], Z_LO, Z_HI)))
            self.k += 1
        return "low" if self.ep.log_wealth < 0.0 else "high"


def run_policy(policy, T, seed, **kw):
    """Run a planner (``adv`` uses a monitor-aware chooser)."""
    if policy == "adv":
        return WM.run(T, seed=seed, chooser=AdversaryChooser(), **kw)
    return WM.run(T, policy, seed=seed, **kw)


def _first_alarm(mon, z, tmax=None):
    for t, v in enumerate(z):
        if tmax is not None and t >= tmax:
            break
        if mon.step(float(v)):
            return t
    return None


def _calib_run(args):
    kind, name, s, seed, ref_stats = args
    ref, m0, s0 = ref_stats
    raw, std, _ = WM.run(T_CAL, "exo", seed=seed)
    z = raw if kind == "raw" else std
    return _first_alarm(build(kind, name, s, ref, m0, s0), z) is not None


def calibrate(pool, kind, name, ref_stats, n=150):
    best, gap = None, 1e9
    for s in GRIDS[name]:
        hits = pool.map(_calib_run, [(kind, name, s, 5000 + i, ref_stats) for i in range(n)])
        g = abs(np.mean(hits) - ALPHA)
        if g < gap:
            best, gap = s, g
    return best


# ---- Part A: validity -----------------------------------------------------------------


def _null_run(args):
    policy, seed, specs, refs = args
    raw, std, _ = run_policy(policy, T_EVAL, seed)
    out = {}
    for (kind, name), sens in specs.items():
        ref, m0, s0 = refs[kind]
        z = raw if kind == "raw" else std
        a = _first_alarm(build(kind, name, sens, ref, m0, s0), z)
        out[f"{name}|{kind}"] = a
    ep = EProcess(eps=0.0, z_lo=Z_LO, z_hi=Z_HI, alpha=ALPHA)
    a = None
    for t, v in enumerate(std):
        ep.update(float(np.clip(v, Z_LO, Z_HI)))
        if ep.rejected:
            a = t
            break
    out["ATLAS|std"] = a
    return out


def _power_run(args):
    policy, seed, refs = args
    raw, std, _ = run_policy(policy, T_POW, seed, cp=CP, drift=DRIFT)
    out = {}
    ep = EProcess(eps=0.0, z_lo=Z_LO, z_hi=Z_HI, alpha=ALPHA)
    a = None
    for t, v in enumerate(std):
        ep.update(float(np.clip(v, Z_LO, Z_HI)))
        if ep.rejected:
            a = t
            break
    out["ATLAS"] = a
    ref, _, _ = refs["std"]
    cm = ConformalMartingale(ref, alpha=ALPHA)
    out["Conformal MG (std)"] = _first_alarm(cm, std)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    raw_ref, std_ref = _ref("raw"), _ref("std")
    refs = {"raw": (raw_ref, float(raw_ref.mean()), float(raw_ref.std())),
            "std": (std_ref, float(std_ref.mean()), float(std_ref.std()))}
    detectors = ["CUSUM", "Page-Hinkley", "ADWIN", "KSWIN"]

    with Pool(4) as pool:
        specs = {}
        for kind in ("raw", "std"):
            for name in detectors:
                specs[(kind, name)] = calibrate(pool, kind, name, refs[kind])
            specs[(kind, "Conformal MG")] = None
        print("calibrated sensitivities:", {f"{n}|{k}": (None if v is None else round(float(v), 4))
                                            for (k, n), v in specs.items()})

        # ---- Part A
        res = {}
        for j, pol in enumerate(POLICIES):
            runs = pool.map(_null_run, [(pol, 20000 + 1000 * j + i, specs, refs) for i in range(N_EVAL)])
            tab = {}
            for key in runs[0]:
                al = [r[key] for r in runs]
                fa_cal = np.mean([a is not None and a < T_CAL for a in al])
                fa_all = np.mean([a is not None for a in al])
                tab[key] = (float(fa_cal), float(fa_all))
            res[pol] = tab

        # ---- Part B
        pw = {}
        for j, pol in enumerate(POLICIES):
            runs = pool.map(_power_run, [(pol, 40000 + 1000 * j + i, refs) for i in range(N_POWER)])
            tab = {}
            for key in runs[0]:
                al = [r[key] for r in runs]
                det = [a - CP for a in al if a is not None and a >= CP]
                tab[key] = dict(detected=len(det) / N_POWER,
                                mean_delay=float(np.mean(det)) if det else None,
                                median_delay=float(np.median(det)) if det else None)
            pw[pol] = tab

    # ---- noise level induced by each planner (for the narrative)
    qlev = {}
    for pol in POLICIES:
        _, _, q = run_policy(pol, 3000, 1)
        qlev[pol] = float(q[300:].mean())

    names = list(res["exo"].keys())
    print("\nPart A -- false-alarm rate (T_cal=%d / 3*T_cal=%d), faithful model" %
          (T_CAL, T_EVAL))
    print(f"{'monitor':<24}" + "".join(f"{p:>20}" for p in POLICIES))
    for n in names:
        print(f"{n:<24}" + "".join(f"{res[p][n][0]:>10.3f}/{res[p][n][1]:<9.3f}" for p in POLICIES))
    print("mean noise level q under each planner:", {k: round(v, 3) for k, v in qlev.items()})
    print("\nPart B -- detection after drift %.2f (delay in rounds; detected fraction)" % DRIFT)
    for pol in POLICIES:
        for k, v in pw[pol].items():
            print(f"  {pol:<5}{k:<22} detected={v['detected']:.2f} mean delay={v['mean_delay']}")

    with open(os.path.join(RESULTS, "exp6_endogeneity.json"), "w") as f:
        json.dump(dict(validity=res, power=pw, noise_level=qlev, n_eval=N_EVAL,
                       n_power=N_POWER, t_cal=T_CAL, t_eval=T_EVAL, drift=DRIFT), f, indent=2)

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    show = [n for n in names if n.split("|")[0] != "ATLAS"] + ["ATLAS|std"]
    x = np.arange(len(show))
    w = 0.26
    cols = {"exo": "#c6dbef", "low": "#9ecae1", "high": "#4292c6", "adv": "#08519c"}
    w = 0.2
    for i, pol in enumerate(POLICIES):
        ax[0].bar(x + (i - 1.5) * w, [res[pol][n][1] for n in show], w, label=f"planner: {pol}",
                  color=cols[pol])
    ax[0].axhline(ALPHA, color="crimson", ls="--", lw=1, label=r"target $\alpha$")
    ax[0].set_xticks(x); ax[0].set_xticklabels([s.replace("|", "\n") for s in show], fontsize=7)
    ax[0].set_ylabel(f"false-alarm rate by round {T_EVAL}")
    ax[0].set_title("Validity on a closed-loop stream (faithful model)")
    ax[0].legend(frameon=False, fontsize=7)
    xs = np.arange(len(POLICIES))
    ax[1].bar(xs, [pw[p]["ATLAS"]["mean_delay"] or 0 for p in POLICIES], color="#1b8a5a")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(POLICIES)
    ax[1].set_ylabel("ATLAS mean delay after drift (rounds)")
    ax[1].set_title("Power depends on where the planner steers")
    fig.tight_layout()
    out = os.path.join(RESULTS, "exp6_endogeneity.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("figure ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
