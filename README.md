# ATLAS — Anytime-valid Testing of LAtent Simulators

**Sequential certification of world-model faithfulness via e-process betting.**

A world model (latent simulator) is only useful if its imagined rollouts stay
faithful to reality — but faithfulness is not static. It **decays with horizon**
(compounding error) and **drifts over deployment time** (non-stationarity, the
sim-to-real gap, and the self-referential feedback loop of model-based RL, where an
improving policy steers the agent into states the model never saw). Current practice
scores world models with one-off offline metrics (FVD, PSNR, return gap). None of
them gives a deployed model a **running, statistically valid certificate**:

> *As of now, this model's rollouts are trustworthy up to horizon $h^\*(t)$ at error
> level $\varepsilon$* — with a guarantee that holds **uniformly over all time**.

That certificate is exactly what an **e-process** delivers and offline metrics
cannot. ATLAS builds one.

## The idea in one paragraph

At each deployment step the model emits an action-conditioned $h$-step predictive
distribution; reality later reveals the truth. ATLAS scores each horizon with a
proper scoring rule (energy / CRPS / latent Mahalanobis — hence **representation-
agnostic**), bets against the null "*this horizon is faithful within $\varepsilon_h$*"
with a test martingale, and tracks the **faithfulness frontier** $h^\*(t)$ = the
largest horizon whose e-process hasn't crossed $1/\alpha$. A planner can then legally
clip its imagination depth to $h^\*(t)$ — turning the certificate into an algorithm,
not just a monitor.

The construction is **conditional** (one bet at a time), so it needs **no
i.i.d./exchangeability** assumption — which is what lets it stay valid on the
endogenous, policy-dependent data stream that breaks conformal methods.

## The three theorems

1. **Uniform validity.** ATLAS never revokes a truly faithful horizon except w.p.
   $\le \alpha$, uniformly over all time (Ville).
2. **Anytime-valid simulation lemma.** The running certified one-step error yields a
   time-uniform bound on the $h$-step value / divergence gap — the bridge to the
   model-based-RL audience. *(No anytime-valid simulation lemma currently exists.)*
3. **Detection delay.** After a real dynamics shift at unknown $\tau$, $h^\*(t)$
   collapses within $O(\log(1/\alpha)/\Delta^2)$ rounds — order-optimal via GRO
   betting.

See [`docs/03_theorems.md`](docs/03_theorems.md) for formal statements and proof
sketches.

## Repository layout

```
docs/                   Theory notes (earlier drafts; the paper supersedes them)
atlas/                  Reference implementation
  scores.py             Proper scoring rules (energy, CRPS, Gaussian NLL, Mahalanobis)
  eprocess.py           Mixture-betting e-process, betting CS (constant mean), e-detector
  cs.py                 Confidence sequence for the running average of conditional means,
                        and its windowed form (the certificate)
  frontier.py           Horizon-indexed e-processes, revocation frontier h*(t)
  simulator.py          Linear-Gaussian latent world model (drift injection)
  closedloop.py         Closed-loop world model with state-dependent noise and planners
  fastsim.py            Vectorised multi-horizon streams and e-process paths
experiments/
  exp1_validity.py      Uniform validity (no false revocation)
  exp2_frontier.py      Frontier collapse and the windowed certified gap
  exp3_delay.py         Detection delay vs (log 1/alpha + log m)/D*
  exp4_baselines.py     Competitor monitors on the simulator
  exp5_horizon_delay.py Horizon-resolved delay law; interleaving negative result
  exp6_endogeneity.py   Closed-loop planners: validity and power vs competitors
  exp7_ablations.py     Bet, tolerance, drift shape, statistic, noise tails
  exp8_real_replication.py  Replicated real-video study (30 splits) with competitors
  exp9_simlemma.py      Certified total-variation bound vs closed-form truth
  exp10_gated_control.py Planner with a neural dynamics model gated by the frontier
  real_data/            Moving MNIST / KTH loaders and the PCA+linear world model
  gpu/                  Neural world-model monitors (single illustrative runs)
tests/                  Ville control, CS coverage (time-varying means), power, scores
results/                JSON outputs and figures of the simulator experiments
paper/                  Manuscript (Neural Networks / elsarticle) and tables
```

## Results (see paper/ for the full account)

- **Simulator with ground truth.** No false revocation; measured delay matches the
  horizon-resolved prediction within 16%; the certified TV bound is never violated.
- **Closed-loop endogeneity.** On a stream steered by a model-based planner, raw-error
  monitors calibrated on exogenous data inflate false alarms (CUSUM and a conformal
  martingale reach 1.00 when the planner seeks high-noise states); ATLAS stays at or
  below 0.013. Power depends on where the planner steers.
- **Gated planning.** A planner with a neural dynamics model, gated by the frontier, under
  three faults: lowest mean post-fault cost at margins k = 0.25-0.5, no false trigger in 100
  fault-free episodes; slower than CUSUM on the abrupt fault; braking hurts on the other two.
- **Real video, replicated.** Moving MNIST speed shift and two KTH shifts, with linear,
  conv-autoencoder and frozen-encoder world models, against CUSUM, ADWIN, KSWIN and a
  conformal martingale. False-alarm control needs a tolerance margin; a tuned CUSUM is
  competitive on the linear model; the frozen-encoder model reaches 0 false alarms with
  detection 0.9-1.0 at k = 1 (10 repetitions).
- **Negative results.** Pooling overlapping horizon chains does not recover the discarded
  data; a constant tolerance calibrated on pooled data voids the conditional guarantee.

Run the simulator suite with `python -m experiments.run_all` and the others with
`python -m experiments.expN_...`; the real-video study needs
`python -m experiments.real_data.download`.

## Positioning

Third pillar alongside **SENTRY** (agent-behavior monitoring) and **ORACLE**
(sequential causal discovery): *SENTRY audits what the agent does; ATLAS audits what
the agent believes.*

## License

See [`LICENSE`](LICENSE).
