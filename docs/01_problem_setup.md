# ATLAS — Formal Problem Setup

**ATLAS: Anytime-valid Testing of LAtent Simulators**
Sequential certification of world-model faithfulness via e-process betting.

This document fixes notation and states the formal problem. It is deliberately
representation-agnostic: the same construction covers pixel-space video predictors
(Cosmos/Genie-style), latent dynamics models (Dreamer), and JEPA-style latent
simulators. Companion documents:

- [`02_construction.md`](02_construction.md) — the e-process construction and the faithfulness frontier.
- [`03_theorems.md`](03_theorems.md) — the three theorems (uniform validity, anytime-valid simulation lemma, detection delay).
- [`notation.md`](notation.md) — symbol table.

---

## 1. Deployment protocol

A **world model** (latent simulator) $M$ is deployed online alongside a policy /
planner $\pi$. Time proceeds in rounds $t = 1, 2, \dots$. We deliberately do **not**
assume the data stream is i.i.d. or exchangeable; on the contrary, the central
technical difficulty (Section 5) is that the stream is *endogenous* — the policy
that plans inside $M$ changes which slice of reality is subsequently observed.

At round $t$:

1. The agent occupies a true (possibly latent) state $x_t \in \mathcal{X}$, with
   observation $o_t \in \mathcal{O}$.
2. The planner selects an action sequence $a_{t:t+H-1} = (a_t, \dots, a_{t+H-1})$
   up to a maximum horizon $H$, using $M$ for imagined rollouts.
3. For each horizon $h \in \{1, \dots, H\}$, $M$ emits an **action-conditioned
   $h$-step predictive distribution**
   $$
   P^{M}_{t,h}(\cdot) \;=\; P^{M}\!\big(\, \cdot \mid x_t,\, a_{t:t+h-1} \,\big)
   \;\in\; \Delta(\mathcal{Y}),
   $$
   a distribution over a target space $\mathcal{Y}$ (observations, states, or a
   learned latent code $z_{t+h}$; see Section 4).
4. The agent executes (a prefix of) the plan; reality reveals the realized
   $h$-step outcome $Y^{(h)}_t \in \mathcal{Y}$ (e.g. $Y^{(h)}_t = z_{t+h}$ or
   $o_{t+h}$). This resolves the prediction made at time $t$ for horizon $h$.

A prediction *issued* at time $s$ for horizon $h$ is *resolved* at time $s+h$.
To keep every constructed process adapted to the information it may legally use,
we index each per-horizon stream by its resolution order (Section 3).

## 2. Faithfulness, and why it is not static

$M$ is useful only if its imagined rollouts stay faithful to reality. Faithfulness
fails along two axes that one-off offline metrics (FVD, PSNR, return gap) cannot
track:

- **Horizon decay.** Recursive rollouts compound one-step error; a model faithful
  at $h=1$ can be useless at $h=20$. Faithfulness is therefore *horizon-indexed*.
- **Deployment drift.** Non-stationarity, sim-to-real gap, and the self-referential
  feedback loop of model-based RL (an improving $\pi$ pushes the agent into states
  $M$ never trained on) make faithfulness *time-varying*.

ATLAS's object of interest is a **running, statistically valid certificate**:

> *As of round $t$, this model's rollouts are trustworthy up to horizon $h^\*(t)$
> at tolerance $\varepsilon$*,

with a guarantee that holds **uniformly over all time** (anytime-valid), not just
at a pre-specified stopping round.

## 3. Nonconformity scores

Fix a **negatively-oriented, strictly proper scoring rule** $S : \Delta(\mathcal{Y})
\times \mathcal{Y} \to [0, B]$ (smaller is better; we assume a bounded range $B$,
attained e.g. by a bounded target space or by clipping — boundedness is used only
for the empirical-Bernstein e-variables and can be relaxed to a sub-$\psi$ tail
condition). Canonical choices:

| Model family        | Target $\mathcal{Y}$      | Score $S$                                   |
|---------------------|---------------------------|---------------------------------------------|
| Pixel/video         | frames / patches          | Energy score, CRPS (per-channel, aggregated)|
| Dreamer latent      | latent state $z$          | Gaussian log-score, CRPS                     |
| JEPA latent         | predicted embedding       | Mahalanobis / energy score in latent metric |
| LLM textual WM      | token / semantic embed.   | Semantic energy score, log-loss             |

The **latent-space Mahalanobis / energy score** is what makes ATLAS representation-
agnostic: it scores the predicted *embedding* against the realized embedding under
the model's own latent metric, so no pixel reconstruction is required for JEPA-style
models that never decode.

For horizon $h$, let $k$ index the resolution rounds (i.e. the $k$-th time a
horizon-$h$ prediction is resolved). Define the per-round score
$$
S^{(h)}_k \;=\; S\!\big(P^{M}_{s_k, h},\, Y^{(h)}_{s_k}\big),
$$
where $s_k$ is the issue-time of the $k$-th resolved horizon-$h$ prediction.

## 4. Score divergence and the tolerance $\varepsilon_h$

Let $Q_{s,h} = \mathrm{Law}\big(Y^{(h)}_s \mid \mathcal{F}_{s}\big)$ be the true
conditional law of the outcome given the information at issue-time. For a strictly
proper $S$, the **score divergence**
$$
d_S(Q, P) \;=\; \mathbb{E}_{Y\sim Q}\!\big[S(P, Y)\big] - \mathbb{E}_{Y\sim Q}\!\big[S(Q, Y)\big] \;\ge\; 0,
$$
with equality iff $P = Q$. For the log-score $d_S = \mathrm{KL}(Q\Vert P)$; for the
energy score $d_S$ is (a power of) an integral-probability / MMD-type metric; for a
Gaussian/Mahalanobis score $d_S$ controls a squared Wasserstein-2 distance. These
identities are what let Theorem 2 convert a certified *score* gap into a *dynamics*
gap.

**$\varepsilon$-faithfulness (null hypothesis).** For horizon $h$ and tolerance
$\varepsilon_h \ge 0$:
$$
H_0^{(h)} : \qquad d_S\big(Q_{s,h},\, P^{M}_{s,h}\big) \;\le\; \varepsilon_h
\quad \text{for every issue-time } s .
$$
$\varepsilon_h = 0$ recovers exact conditional calibration; $\varepsilon_h > 0$
tolerates a fixed, application-chosen amount of model error at horizon $h$. In
practice $H_0^{(h)}$ is operationalized against a **predictable calibrated reference**
$R_k$ (a competitor a faithful model must not lose to by more than $\varepsilon_h$);
see [`02_construction.md`](02_construction.md) §2. The choice of $R_k$ is what pins
down the precise meaning of "faithful," and is stated as an explicit modeling knob
rather than hidden.

## 5. The endogeneity problem (why existing CP breaks)

The data stream is **policy-dependent**: the distribution of $x_{t+1}, x_{t+2},
\dots$ depends on the actions the planner chose by imagining rollouts inside $M$.
As $\pi$ improves, it steers the agent into regions where $M$'s errors are largest —
precisely the states we most need to test. Concretely:

- The outcomes $\{Y^{(h)}_t\}$ are **neither i.i.d. nor exchangeable**; standard
  split/full conformal prediction and their time-series variants assume one or the
  other (or a bounded-drift relaxation) and lose coverage here.
- The mechanism generating the data (the policy) is itself a function of the object
  under test (the model), so any fixed reference distribution is moving.

ATLAS resolves this by never invoking exchangeability. We build the test around a
filtration and require only **conditional** ("one bet at a time") validity:

$$
\mathcal{F}_k \;=\; \sigma\Big(\text{everything observable up to the resolution of the } k\text{-th horizon-}h \text{ prediction}\Big),
$$
including all past states, actions, planner internals, and realized outcomes. Every
e-variable is required to satisfy $\mathbb{E}[\,e_k \mid \mathcal{F}_{k-1}\,] \le 1$
under $H_0^{(h)}$. Because this is a *conditional* statement, it holds for **any**
predictable policy and **any** predictable betting rule — the endogenous,
self-referential data stream is handled for free. This conditional martingale
construction is the technical core that distinguishes ATLAS from conformal
approaches that need an exchangeability crutch.

## 6. What ATLAS outputs

At every round $t$, ATLAS reports:

1. A wealth process $W^{(h)}_k$ (test supermartingale / e-process) per horizon.
2. The **faithfulness frontier**
   $$
   h^\*(t) \;=\; \max\Big\{\, h \in \{1,\dots,H\} : W^{(h)}_{k(t)} < 1/\alpha \,\Big\},
   $$
   the largest horizon not yet certified-untrustworthy at level $\alpha$.
3. An anytime-valid **confidence sequence** on the one-step faithfulness gap
   $\delta_1(t)$, feeding the simulation-lemma bound (Theorem 2).

The frontier turns the certificate into an *algorithm*: a planner may legally clip
its imagination depth to $h^\*(t)$, so ATLAS is a controller, not merely a monitor.
