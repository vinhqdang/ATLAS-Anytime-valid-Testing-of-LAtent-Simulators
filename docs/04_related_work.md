# ATLAS — Related Work & Novelty Ledger  *(citations UNVERIFIED — to lock in the lit sweep)*

> **Status.** This is a *scaffold* for the literature sweep, not a verified
> bibliography. Every author/year/venue below is a placeholder to be confirmed
> against primary sources before it enters any paper. Do **not** cite from this file
> as-is. The purpose here is to state, per neighboring area, (i) what exists and
> (ii) the precise gap ATLAS claims — so the sweep can confirm or refute each gap.

## Neighboring areas and the claimed gap

### 1. Anytime-valid inference / testing by betting
- **Exists:** Ville's inequality; test (super)martingales and e-values/e-processes;
  testing-by-betting and the betting/confidence-sequence duality; empirical-Bernstein
  and mixture (GRO) betting; e-detectors for sequential change detection.
- **Gap ATLAS claims:** none of this has been instantiated for **world-model
  faithfulness** on **action-conditioned, policy-endogenous** streams, nor as a
  **horizon-indexed family** with a frontier estimator.
- *To verify:* exact statements/attributions for Ville, WSR betting, mixture method,
  e-detector delay optimality; whether any e-process work already targets RL/dynamics
  models.

### 2. Conformal prediction for time series / multi-step
- **Exists:** split/full conformal; adaptive conformal inference under distribution
  shift; conformal for time series; multi-step conformal predictors.
- **Gap ATLAS claims:** these are built for (approx.) exchangeable or bounded-drift
  single-step prediction and degrade under **recursive multi-step rollouts** with
  **endogenous** data; they give marginal coverage, not an **anytime-valid, per-
  horizon certificate** with a Ville-type never-revoke guarantee.
- *To verify:* the strongest existing multi-step / rollout conformal method and its
  exact assumptions; any conformal work handling policy-dependent data.

### 3. World-model / simulator evaluation
- **Exists:** FVD, PSNR/SSIM, return-gap, and other one-off offline metrics; recent
  world-model benchmarks (video and latent).
- **Gap ATLAS claims:** all offline and static; **no running certificate**, no
  statistically valid **trust horizon**, and empirically **blind** to the moment a
  sim-to-real gap opens (the headline comparison).
- *To verify:* whether any world-model eval offers a sequential / online validity
  guarantee (expected: no).

### 4. Model-based RL theory — the simulation lemma
- **Exists:** the classical simulation lemma bounding value gap by average one-step
  model error (Kearns–Singh lineage; Lipschitz-MDP variants).
- **Gap ATLAS claims:** all versions are **fixed-$n$** and assume the one-step error
  is **known/estimated offline**; there is **no anytime-valid, empirically-certified**
  simulation lemma driven by an online confidence sequence (Theorem 2).
- *To verify:* survey simulation-lemma variants; confirm none is time-uniform /
  e-process-driven.

### 5. Sequential change detection
- **Exists:** CUSUM, Shiryaev–Roberts, GLR; Lorden/Lai delay lower bounds;
  nonparametric and e-detector approaches.
- **Gap ATLAS claims:** not previously applied to **world-model faithfulness drift**
  with a **score-divergence null** and horizon frontier; ATLAS's contribution is the
  instantiation + the coupling of delay to the trust-horizon collapse (Theorem 3).

## Novelty ledger (the three theorem-shaped holes)

| # | Claim | Where | Verify against |
|---|-------|-------|----------------|
| N1 | E-processes over **action-conditioned, policy-endogenous** streams (conditional, no exchangeability) | setup §5, Thm 1 | areas 1, 2 |
| N2 | **Horizon-indexed family** of dependent e-processes + **frontier** $h^\*(t)$ estimator (valid under arbitrary dependence via e-averaging) | constr. §4–5 | areas 1, 3 |
| N3 | **Anytime-valid simulation lemma** (online CS $\to$ time-uniform value/divergence-gap bound) | Thm 2 | area 4 |

## Portfolio positioning *(author's framing — to substantiate)*
Third pillar with **SENTRY** (agent-behavior monitoring) and **ORACLE** (sequential
causal discovery): *SENTRY audits what the agent does; ATLAS audits what the agent
believes.* Backups on the shelf: (a) interventional identifiability of latent world
models; (b) risk-controlled planning with conformal rollout sets.

## Sweep protocol (next pass)
1. For each N1–N3, run targeted queries; record the closest prior art with a verified
   citation and a one-line "why ATLAS differs."
2. Downgrade any gap claim that a real paper already fills; up-rank the surviving
   theorem-shaped holes.
3. Produce a verified `references.bib` and rewrite this file with confirmed citations
   only.
