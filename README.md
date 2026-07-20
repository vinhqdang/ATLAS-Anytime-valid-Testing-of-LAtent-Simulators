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
docs/
  01_problem_setup.md   Deployment protocol, faithfulness, scores, the endogeneity problem
  02_construction.md    E-process betting construction + the faithfulness frontier h*(t)
  03_theorems.md        The three theorems (uniform validity, sim-lemma, detection delay)
  04_related_work.md    Novelty ledger + lit-sweep protocol (citations UNVERIFIED)
  notation.md           Symbol table
```
*(An `atlas/` reference implementation and `experiments/` follow — see roadmap.)*

## Planned experiments

- **DreamerV3** on DMC / Atari with an injected mid-deployment dynamics shift
  (mass/friction perturbation): show $h^\*(t)$ collapsing exactly when the gap opens
  while FVD / offline metrics stay blind.
- A **JEPA-style latent model**, scored purely in latent space (no decoding).
- An **LLM-agent "textual world model"** setting (connects to the SENTRY line of work).

## Roadmap / status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Formal problem setup + three theorem statements | **done** (this commit) |
| 1 | Verified literature sweep → `references.bib`, rewrite `04_related_work.md` | next |
| 2 | Full proofs | pending |
| 3 | `atlas/` reference implementation (e-process, frontier, scores) | pending |
| 4 | DreamerV3 / JEPA / LLM-WM experiments | pending |
| 5 | Paper draft (target: ICLR 2027 World-Model Workshop → main track / AISTATS 2027 / TMLR) | pending |

## Positioning

Third pillar alongside **SENTRY** (agent-behavior monitoring) and **ORACLE**
(sequential causal discovery): *SENTRY audits what the agent does; ATLAS audits what
the agent believes.*

## License

See [`LICENSE`](LICENSE).
