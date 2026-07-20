# ATLAS — Related Work & Novelty Ledger

*Literature sweep, first verified pass. Citations use the keys in
[`references.bib`](references.bib); authors/years/titles were checked against
primary sources. A handful of exact volume/page fields are flagged there for a final
check at submission. This supersedes the earlier UNVERIFIED scaffold.*

ATLAS sits at the intersection of four literatures. For each we state what exists and
the precise gap ATLAS fills; the sweep **confirmed** that no prior work brings
anytime-valid sequential testing to world-model faithfulness.

## 1. Anytime-valid inference / testing by betting

**Exists.** Ville's inequality for nonnegative supermartingales
[ville1939]; test (super)martingales and e-values/e-processes, and the
game-theoretic foundations [shafer2019game, vovk2021evalues,
ramdas2023savi]; testing by betting [shafer2021testing]; safe testing and
growth-rate-optimal (GRO/GROW) e-variables [gruenwald2024safe]; betting
confidence sequences for bounded means [waudbysmith2024betting] and
time-uniform confidence sequences [howard2021timeuniform].

**Gap ATLAS fills.** This machinery has not been instantiated for **world-model
faithfulness** on **action-conditioned, policy-endogenous** streams, nor as a
**horizon-indexed family** with a frontier estimator. ATLAS's per-horizon e-process
(construction §2) and the betting CS driving the simulation lemma are direct
specializations of [waudbysmith2024betting, gruenwald2024safe]; the contribution is
the world-model construction on endogenous data (**N1**) and the frontier (**N2**).

## 2. Conformal prediction for time series / multi-step

**Exists.** Conformal prediction [vovk2005algorithmic]; validity under
covariate shift [tibshirani2019covariate]; adaptive conformal inference under
distribution shift [gibbs2021adaptive] and for arbitrary online shifts
[gibbs2024conformal]; a growing set of multi-step / multi-horizon conformal
time-series methods (2024–2026).

**Gap ATLAS fills.** These target (approximately) exchangeable or bounded-drift
**single-step** prediction and provide **marginal coverage**, not an **anytime-valid,
per-horizon certificate** with a Ville-type never-revoke guarantee, and they do not
handle **policy-endogenous** data (the data-generating policy is a function of the
model under test). ATLAS never invokes exchangeability — only a conditional-mean null
under the deployment filtration (setup §5) — which is what survives endogeneity.

## 3. World-model / simulator evaluation

**Exists.** World models and latent simulators [ha2018worldmodels,
hafner2023dreamerv3, assran2023ijepa, bardes2024vjepa]; the dominant metrics are
one-off and offline — FVD [unterthiner2018fvd], PSNR/SSIM/LPIPS, and return
gaps — and recent work notes these do not predict whether a representation is
action-relevant.

**Gap ATLAS fills.** All of these are offline and static: **no running certificate**,
no statistically valid **trust horizon**, and empirically **blind** to the moment a
sim-to-real gap opens (our headline comparison, `figures/` and `results/`). ATLAS
supplies exactly the online, guaranteed object this literature lacks (**N2**).

## 4. Model-based RL theory — the simulation lemma

**Exists.** The classical simulation lemma bounding value gap by average one-step
model error [kearns2002near], its use in model-based policy optimization
[janner2019mbpo], and recent tightness analyses [lobel2024tightness]. The value
loss carries a coefficient that grows with the effective horizon.

**Gap ATLAS fills.** All versions are **fixed-$n$** and assume the one-step error is
**known/estimated offline**. ATLAS's **anytime-valid simulation lemma** (Theorem 2)
drives the horizon bound from an **online betting confidence sequence**
[waudbysmith2024betting] on the certified one-step gap, holding uniformly over time
(**N3**). The sweep found no time-uniform / e-process-driven simulation lemma.

## 5. Sequential change detection (the drift alternative)

**Exists.** CUSUM / Shiryaev–Roberts / GLR; the Lorden/Lai delay lower bounds
[lorden1971, lai1998information]; and the nonparametric **e-detector** framework
[shin2024edetectors], whose Shiryaev–Roberts-style e-detectors ATLAS uses for the
drift setting.

**Gap ATLAS fills.** Not previously applied to **world-model faithfulness drift** with
a **score-divergence null** and a horizon frontier; ATLAS's contribution is the
instantiation and the coupling of the [shin2024edetectors]-style delay to the
trust-horizon collapse (Theorem 3), validated empirically (`results/exp3_delay.png`).

## Nearest neighbours found (adjacent, not overlapping)

The sweep surfaced anytime-valid work on **agent-behaviour** monitoring (detecting
strategic deviations in multi-agent systems) and **LLM-evaluation drift attribution**
— both audit *what an agent does or how it is judged*, whereas ATLAS audits *what the
agent's world model believes*. This matches the intended portfolio position with
SENTRY (behaviour) and confirms ATLAS occupies distinct ground.

## Novelty ledger (the three theorem-shaped holes) — status

| # | Claim | Where | Nearest prior art | Verdict |
|---|-------|-------|-------------------|---------|
| N1 | E-processes over **action-conditioned, policy-endogenous** streams (conditional, no exchangeability) | setup §5, Thm 1 | betting CS/safe testing [waudbysmith2024betting, gruenwald2024safe]; conformal shift [gibbs2024conformal] | **open** — none target endogenous WM data |
| N2 | **Horizon-indexed** e-process family + **frontier** $h^\*(t)$ | constr. §4–5 | e-values combine by averaging [vovk2021evalues]; WM eval [unterthiner2018fvd] | **open** — no valid trust-horizon object |
| N3 | **Anytime-valid simulation lemma** (online CS → time-uniform value/divergence-gap bound) | Thm 2 | simulation lemma [kearns2002near, lobel2024tightness]; betting CS [waudbysmith2024betting] | **open** — all prior bounds fixed-$n$ |

## Remaining sweep tasks
1. Final check of exact volume/page fields in `references.bib` at submission.
2. Add the strongest 2025–2026 multi-step conformal time-series baseline for the
   empirical comparison table.
3. Track any concurrent work on anytime-valid RL/world-model monitoring before camera-ready.
