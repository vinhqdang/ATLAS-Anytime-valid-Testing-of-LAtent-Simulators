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
docs/                   Theory: problem setup, e-process construction, the 3 theorems
atlas/                  Reference implementation
  scores.py             Proper scoring rules (energy, CRPS, Gaussian NLL, Mahalanobis)
  eprocess.py           Mixture-betting e-process, betting confidence sequence, e-detector
  frontier.py           Horizon-indexed e-processes, faithfulness frontier h*(t)
  simulator.py          Controllable linear-Gaussian latent world model (drift injection)
experiments/
  exp1_validity.py      Theorem 1 — uniform validity (no false revocation)
  exp2_frontier.py      Frontier collapse + anytime-valid simulation lemma (Theorem 2)
  exp3_delay.py         Theorem 3 — detection delay ~ (log 1/alpha + log m)/D*
  real_data/            ATLAS on real image-sequence datasets (Moving MNIST, KTH)
tests/                  Ville false-alarm control, CS coverage, power, proper scores
results/                Synthetic-study figures + summary
figures/                Real-data figures (sample frames + frontier collapse)
```

## Results

- **Synthetic ground-truth study** (`results/`): all three theorems validated —
  false-revocation 0.0025 ≤ α; `h*(t)` collapses 38→7 at a drift while an offline
  metric stays blind; detection delay matches `(log 1/α + log m)/D*` to ~6%. Run:
  `python -m experiments.run_all`.
- **Real-world data** (`figures/`): ATLAS monitors a learned latent world model on
  **Moving MNIST** (native → 2× speed) and **KTH Actions** (walking → running);
  `h*(t)` collapses right when the real shift opens. Run:
  `python -m experiments.real_data.download && python -m experiments.real_data.exp_mnist`.

## Planned experiments (GPU phase)

- **DreamerV3** on DMC / Atari with an injected mid-deployment dynamics shift
  (mass/friction perturbation): show $h^\*(t)$ collapsing exactly when the gap opens
  while FVD / offline metrics stay blind.
- A **JEPA-style latent model**, scored purely in latent space (no decoding). The
  PCA+linear latent WM in `experiments/real_data/latent_wm.py` is the CPU stand-in;
  a Dreamer/JEPA model drops in behind the identical interface.
- An **LLM-agent "textual world model"** setting (connects to the SENTRY line of work).

## Roadmap / status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Formal problem setup + three theorem statements | **done** |
| 3 | `atlas/` reference implementation (e-process, frontier, scores) | **done** |
| 4a | Synthetic ground-truth validation of Theorems 1–3 | **done** |
| 4b | Real-data experiments (Moving MNIST, KTH Actions) | **done** |
| 1 | Verified literature sweep → `references.bib`, rewrite `04_related_work.md` | next |
| 2 | Full proofs | pending |
| 4c | DreamerV3 / JEPA / LLM-WM experiments (GPU) | pending |
| 5 | Paper draft (target: ICLR 2027 World-Model Workshop → main track / AISTATS 2027 / TMLR) | pending |

## Positioning

Third pillar alongside **SENTRY** (agent-behavior monitoring) and **ORACLE**
(sequential causal discovery): *SENTRY audits what the agent does; ATLAS audits what
the agent believes.*

## License

See [`LICENSE`](LICENSE).
