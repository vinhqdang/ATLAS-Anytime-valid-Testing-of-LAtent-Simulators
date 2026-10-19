"""Experiment 10 -- does gating the planner with ATLAS improve control?

A neural world model (heteroskedastic MLP) is trained on the nominal point-mass system and
used by a gradient-based planner that *commits* to an open-loop action block of length ``L``
at every replanning step. Committing makes the h-step predictive well defined for h <= L, so
the h-step ATLAS statistic is valid (Section 2.3). Halfway through an episode a fault changes
the true dynamics (``gain``, ``flip``, ``wind``) while the model keeps its nominal behaviour.

Controllers (all use the same model and planner):
  fixed-L        commit L in {4, 16}, never gated
  ATLAS-gated    L = h*(t) from the revocation frontier over horizons {1,2,4,8,16};
                 h* <= 1 triggers the safe fallback (zero action, "brake")
  CUSUM-gated    L = 16 until a CUSUM on the one-step standardised residual alarms, then brake
  oracle         L = 16 until the true fault onset, then brake (aborting mid-block)
  brake          the safe fallback all episode (reference)

Tolerances and CUSUM thresholds are calibrated on separate nominal closed-loop episodes.
Per episode we record mean stage cost before / after the fault, the alarm time, and whether a
gate triggered before the fault (a false trigger).

Run: ``python -m experiments.exp10_gated_control``
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import torch

from atlas.ctrl_env import PointMass, stage_cost, FAULTS
from atlas.ctrl_wm import train, plan, predictive_stat
from atlas.eprocess import EProcess
from experiments.baselines import CUSUM

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
T, CP = 400, 200
LADDER = [1, 2, 4, 8, 16]
LMAX = 16
ALPHA = 0.05
Z_LO, Z_HI = -1.0, 8.0
N_SEEDS = int(os.environ.get("N_SEEDS", 20))
N_CAL = int(os.environ.get("N_CAL", 12))
K_LIST = [0.0, 0.25, 0.5, 1.0, 2.0]                # tolerance margins (sd units); 1.0 is the pre-set default
K_MAIN = 1.0
CONTROLLERS = (["fixed-4", "fixed-16"] + [f"ATLAS-gated k={k}" for k in K_LIST]
               + ["ATLAS-brake k=0.5", "ATLAS-brake k=1.0", "CUSUM-gated", "oracle", "brake"])
N_NULL = int(os.environ.get("N_NULL", 100))      # fault-free episodes for the false-trigger study

_MODELS = {}


def model(i):
    if i not in _MODELS:
        _MODELS[i] = train(seed=100 + i, epochs=40, n_steps=20000)
    return _MODELS[i]


def one_step_z(wm, s, a, s_next):
    """Standardised one-step residual (velocity subspace) for the CUSUM baseline."""
    z, _ = predictive_stat(wm, s, np.asarray(a)[None], s_next, n=64)
    return z


def run_episode(ctrl, fault, wm, seed, eps=None, cusum=None, collect=False):
    env = PointMass(seed)
    s = env.reset()
    t = 0
    prev = None
    costs = np.zeros(T)
    alarm = None
    braking = False
    eps = eps or {}
    frontier = {h: EProcess(eps.get(h, 0.0), Z_LO, Z_HI, alpha=ALPHA / len(LADDER))
                for h in LADDER}
    cus = CUSUM(cusum[1], 0.25 * cusum[2], cusum[0]) if cusum is not None else None
    stats = {h: [] for h in LADDER}
    one_step = []
    while t < T:
        f_now = fault if t >= CP else "none"
        # ---- choose block length / mode
        if ctrl == "brake":
            L, braking = 1, True
        elif ctrl == "oracle":
            braking = (t >= CP and fault != "none")
            L = 1 if braking else LMAX
        elif ctrl == "fixed-4":
            L = 4
        elif ctrl in ("fixed-16", "CUSUM-gated"):
            L = LMAX
            if braking:
                L = 1
        elif ctrl.startswith("ATLAS-gated") or ctrl.startswith("ATLAS-brake"):
            h_star = 0
            for h in LADDER:
                if frontier[h].rejected:
                    break
                h_star = h
            if h_star <= 1:
                braking = True
            if ctrl.startswith("ATLAS-brake"):
                L = 1 if braking else LMAX         # brake-only: never shrink the block
            else:
                L = 1 if braking else min(LMAX, h_star)
        else:
            raise KeyError(ctrl)
        if braking:
            acts = np.zeros((L, 2))
        else:
            acts = plan(wm, s, t, L, prev=None)
        traj = [s.copy()]
        for i in range(min(L, T - t)):
            f_now = fault if t >= CP else "none"
            if ctrl == "oracle" and not braking and fault != "none" and t >= CP:
                braking = True                     # switch exactly at the fault, mid-block
                alarm = t
                break
            s_new = env.step(acts[i], f_now)
            costs[t] = stage_cost(s_new, acts[i], t + 1)
            if cus is not None or collect:
                one_step.append(one_step_z(wm, s, acts[i], s_new))
            if cus is not None and not braking and cus.step(one_step[-1]):
                braking = True
                alarm = t
                s = s_new
                t += 1
                break
            s = s_new
            traj.append(s.copy())
            t += 1
            # ---- ATLAS: horizon h completes a chain element every h steps of the block;
            # the actions were committed at block start, so each element's predictive is
            # conditioned on actions known at its issue time. The gate is checked every step.
            if (ctrl.startswith("ATLAS") or collect) and not braking and L > 1:
                k = len(traj) - 1                      # steps executed in this block
                for h in LADDER:
                    if h <= L and k % h == 0:
                        z, _ = predictive_stat(wm, traj[k - h], acts[k - h:k], traj[k], n=200)
                        z = float(np.clip(z, Z_LO, Z_HI))
                        stats[h].append(z)
                        if ctrl.startswith("ATLAS"):
                            frontier[h].update(z)
                if ctrl.startswith("ATLAS"):
                    h_star = 0
                    for h in LADDER:
                        if frontier[h].rejected:
                            break
                        h_star = h
                    if h_star <= 1:                    # abort the block, enter safe mode
                        braking = True
                        alarm = t - 1
                        break
    if ctrl == "oracle":
        alarm = CP if fault != "none" else None
    out = dict(pre=float(costs[:CP].mean()), post=float(costs[CP:].mean()),
               alarm=alarm)
    if collect:
        out["stats"] = stats
        out["one_step"] = one_step
    return out


def calibrate(mi):
    """Nominal closed-loop episodes of the fixed-16 controller -> tolerances and CUSUM."""
    wm = model(mi)
    zs = {h: [] for h in LADDER}
    os_all = []
    for j in range(N_CAL):
        r = run_episode("fixed-16", "none", wm, seed=9000 + 50 * mi + j, collect=True)
        for h in LADDER:
            zs[h] += r["stats"][h]
        os_all.append(np.array(r["one_step"]))
    ref = np.concatenate(os_all)
    m0, s0 = float(ref.mean()), float(ref.std())
    # CUSUM threshold: smallest threshold with <=5% false alarms over nominal episodes
    best = None
    for th in np.linspace(2, 80, 30):
        fa = 0
        for arr in os_all:
            S = 0.0
            for v in arr:
                S = max(0.0, S + (v - m0 - 0.25 * s0))
                if S >= th:
                    fa += 1
                    break
        if fa / len(os_all) <= ALPHA:
            best = th
            break
    return zs, (best if best is not None else 80.0), m0, s0


_CAL = {}


def job(args):
    ctrl, fault, seed = args
    mi = seed % 5
    zs, th, m0, s0 = _CAL[mi]
    k = float(ctrl.split("k=")[1]) if "k=" in ctrl else 0.0
    eps = {h: float(np.mean(zs[h]) + k * np.std(zs[h])) for h in LADDER if zs[h]}
    wm = model(mi)
    # CUSUM runs on the standardised stream centred at its calibration mean
    cusum = (th, m0, s0) if ctrl == "CUSUM-gated" else None
    r = run_episode(ctrl, fault, wm, seed=seed, eps=eps, cusum=cusum)
    r.update(ctrl=ctrl, fault=fault, seed=seed)
    return r


def null_job(args):
    ctrl, seed = args
    return job((ctrl, "none", seed))["alarm"] is not None


def main():
    os.makedirs(RESULTS, exist_ok=True)
    jobs = [(c, f, 500 + sd) for f in FAULTS for sd in range(N_SEEDS) for c in CONTROLLERS]
    with Pool(4) as pool:
        for mi, c in zip(range(5), pool.map(calibrate, range(5))):
            _CAL[mi] = c
    print("calibration sizes (model 0):", {h: len(v) for h, v in _CAL[0][0].items()},
          "cusum", [round(float(x), 3) for x in _CAL[0][1:]])
    with Pool(4) as pool:                    # fork inherits _CAL
        res = pool.map(job, jobs, chunksize=4)
    summary = {}
    print(f"{'fault':<6}{'controller':<13}{'cost pre':>10}{'cost post':>11}{'trigger<CP':>12}{'detect':>8}{'delay':>8}")
    for f in FAULTS:
        for c in CONTROLLERS:
            rs = [r for r in res if r["fault"] == f and r["ctrl"] == c]
            pre = np.array([r["pre"] for r in rs]); post = np.array([r["post"] for r in rs])
            early = np.mean([r["alarm"] is not None and r["alarm"] < CP for r in rs])
            det = [r["alarm"] - CP for r in rs if r["alarm"] is not None and r["alarm"] >= CP]
            rng = np.random.default_rng(0)
            ci = lambda x: tuple(np.percentile([rng.choice(x, len(x)).mean() for _ in range(1000)],
                                               [2.5, 97.5]))
            summary[f"{f}|{c}"] = dict(pre=float(pre.mean()), pre_ci=ci(pre),
                                        post=float(post.mean()), post_ci=ci(post),
                                        early=float(early), detect=len(det) / len(rs),
                                        delay=float(np.mean(det)) if det else None,
                                        n=len(rs))
            s = summary[f"{f}|{c}"]
            print(f"{f:<6}{c:<13}{s['pre']:>10.3f}{s['post']:>11.3f}{s['early']:>12.2f}"
                  f"{s['detect']:>8.2f}{(s['delay'] if s['delay'] is not None else float('nan')):>8.1f}")
    # ---- false triggers over many independent fault-free episodes
    null = {}
    gates = [f"ATLAS-gated k={k}" for k in K_LIST] + ["CUSUM-gated"]
    with Pool(4) as pool:
        for g in gates:
            hits = pool.map(null_job, [(g, 70000 + i) for i in range(N_NULL)], chunksize=2)
            null[g] = float(np.mean(hits))
            print(f"fault-free false triggers, {g}: {null[g]:.3f} over {N_NULL} episodes")
    with open(os.path.join(RESULTS, "exp10_gated_control.json"), "w") as fh:
        json.dump(dict(summary=summary, null=null, n_null=N_NULL, T=T, cp=CP, n_seeds=N_SEEDS,
                       k_main=K_MAIN), fh, indent=2)


if __name__ == "__main__":
    main()
