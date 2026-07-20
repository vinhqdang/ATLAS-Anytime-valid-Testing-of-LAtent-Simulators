# ATLAS — Simulation-Study Results

A ground-truth validation of the three theorems on a controllable linear-Gaussian
latent simulator (`atlas/simulator.py`). This is the setting where the true dynamics
are known, so ATLAS's guarantees can be checked against ground truth — something no
real-world world model permits (there, "truth" is unobserved). Reproduce with:

```bash
pip install -r requirements.txt
python -m experiments.run_all      # writes the three figures + summary.json here
```

## Experiment 1 — Theorem 1: uniform validity (`exp1_validity.png`)

Under the null (no dynamics drift) the model is *exactly* faithful (`eps=0`). Across
400 independent deployments ATLAS's false-revocation rate stays at or below the
level, uniformly over all rounds:

| horizon | false-revocation rate | bound (α/H = 0.01) |
|--------:|----------------------:|-------------------:|
| h=1     | 0.0025                | 0.01               |
| h=2..20 | 0.0000                | 0.01               |
| **any** | **0.0025**            | **α = 0.05**       |

The e-process is conservative, as Ville's inequality guarantees. Long horizons use
**non-overlapping windows** to keep the residuals i.i.d. within a horizon — the fix
for the recursive-multi-step failure of naive multi-step CP.

## Experiment 2 — frontier collapse & the simulation lemma (`exp2_frontier.png`)

A dynamics drift is injected mid-deployment. ATLAS's certified one-step gap
`δ̂₁(t)` (an anytime-valid confidence sequence) stays low while faithful and rises
right when the drift opens; the Theorem-2 trust horizon
`h*(t) = max{h : C(h,γ)·φ(δ̂₁(t)) ≤ budget}` **collapses from ~38 to ~7 within a few
rounds of the change**, while an offline metric frozen at its pre-deployment value
stays at 38 — blind, and dangerously overconfident.

## Experiment 3 — Theorem 3: detection delay (`exp3_delay.png`)

Sweeping the drift margin and measuring detection delay (mixture e-process, 100
seeds) confirms the Theorem-3 law **delay ≈ (log(1/α)+log m)/D\***, where `D*` is the
empirically-estimated GRO growth rate:

| margin m | D*    | delay (measured) | (log 1/α + log m)/D* |
|---------:|------:|-----------------:|---------------------:|
| 0.10     | 0.008 | 645              | 725                  |
| 0.44     | 0.053 | 118              | 113                  |
| 1.40     | 0.170 | 37               | 35                   |

Mean relative error **~6%**; `delay·D* ≈ 6` is nearly constant (`= log(1/α)+log
n_grid`, the mixture-betting penalty). Delay is `∝ 1/D*`; here `D* ∝ m` because the
bet is capped at `λ_max` (the `∝ m²` / slope-`-2` regime needs `m ≲ 0.07`).

## Notes / honest caveats

- The behaviour policy is **exogenous exploration** so the action-conditioned
  predictive is exactly calibrated under the null (a genuine null to test). The fully
  endogenous policy-in-the-loop regime is the target of the real-world-model phase;
  the conditional e-process is valid there for the same reason (no exchangeability).
- Numbers above are from the committed configs; exact values shift slightly with
  `N_RUNS`/seeds. `summary.json` records the run that produced the figures.
