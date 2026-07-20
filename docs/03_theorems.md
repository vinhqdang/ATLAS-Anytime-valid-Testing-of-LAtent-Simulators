# ATLAS — The Three Theorems

Prerequisites: [`01_problem_setup.md`](01_problem_setup.md),
[`02_construction.md`](02_construction.md).

These are the three theorem-shaped holes ATLAS fills:

1. **Theorem 1 — Uniform validity / no false revocation.** ATLAS never revokes a
   truly faithful horizon, except with probability $\le \alpha$, *uniformly over all
   time*.
2. **Theorem 2 — Anytime-valid simulation lemma.** The running certified one-step
   error yields a time-uniform bound on the $h$-step value / divergence gap. (To our
   knowledge no anytime-valid simulation lemma exists; this is the bridge to the
   model-based-RL audience.)
3. **Theorem 3 — Detection delay under drift.** When faithfulness breaks at an
   unknown changepoint, ATLAS detects it with a delay that matches the information-
   theoretic optimum up to constants, via GRO betting.

Statements are given at working rigor with proof sketches; full proofs are deferred
to the paper. Each statement flags exactly which standing assumptions it uses.

---

## Standing assumptions

- **(A1) Bounded score.** $S \in [0,B]$, hence $Z_k \in [-B,B]$. (Relaxable to a
  sub-$\psi$ conditional tail; only Thm 2–3 rates depend on it.)
- **(A2) Predictability.** Bets $\lambda_k$ and reference $R_k$ are
  $\mathcal{F}_{k-1}$-measurable. No i.i.d./exchangeability assumption is made on the
  data stream (this is the point — see setup §5).
- **(A3) Proper score / divergence link.** $S$ is strictly proper with a known
  modulus $\phi$ relating its divergence to a probability metric $\rho$:
  $\rho\big(Q,P\big) \le \phi\big(d_S(Q,P)\big)$ (e.g. Pinsker $\rho=\mathrm{TV}$,
  $\phi(u)=\sqrt{u/2}$ for the log-score; energy score $\to$ MMD/IPM; Gaussian score
  $\to W_2$). Used only by Theorem 2.
- **(A4) Rollout regularity (for Thm 2).** Either (i) $(\gamma,R_{\max})$-discounted
  returns, or (ii) $L$-Lipschitz true dynamics in $\rho$, so that one-step model
  error propagates through an $h$-step rollout with a known amplification factor
  $C(h)$.

---

## Theorem 1 — Uniform validity (no false revocation)

> **Theorem 1.** Fix horizon $h$ and $\alpha \in (0,1)$. Under $H_0^{(h)}$ and (A1)–(A2),
> the ATLAS wealth $W^{(h)}$ of [`02_construction.md`](02_construction.md) §2 satisfies
> $$
> \Pr\Big(\, \exists\, K \ge 1 : W^{(h)}_K \ge 1/\alpha \,\Big) \;\le\; \alpha .
> $$
> Equivalently, the revocation time $\tau_h = \inf\{K : W^{(h)}_K \ge 1/\alpha\}$
> obeys $\Pr(\tau_h < \infty \mid H_0^{(h)}) \le \alpha$: the probability that ATLAS
> **ever** revokes a truly $\varepsilon_h$-faithful horizon is at most $\alpha$,
> uniformly over all rounds and all (predictable) policies and betting rules.
>
> **Simultaneous frontier control.** Let $\mathcal{H}_0 = \{h : H_0^{(h)} \text{ holds}\}$
> be the set of genuinely faithful horizons. Running horizon $h$ at level $\alpha/H$
> (union bound) gives
> $$
> \Pr\Big(\, \exists\, t,\ \exists\, h \in \mathcal{H}_0 : h \text{ revoked by round } t \,\Big) \;\le\; \alpha,
> $$
> i.e. with probability $\ge 1-\alpha$ the frontier $h^\*(t)$ **never dips below**
> $\min \mathcal{H}_0^{c}\!-\!1$… more precisely, never falsely excludes any faithful
> horizon, for all $t$ simultaneously. The averaged e-process of §5 gives the same at
> $O(1)$ rather than $O(\log H)$ cost when a single pooled certificate suffices.

**Proof sketch.** Under $H_0^{(h)}$, (A2) makes each $e_k = 1+\lambda_k(Z_k-\varepsilon_h)$
a conditional e-variable ($\mathbb{E}[e_k\mid\mathcal F_{k-1}]\le 1$; §2), and (A1)+the
$\lambda_{\max}$ cap keep $e_k \ge 0$. Hence $(W^{(h)}_K)$ is a nonnegative
supermartingale with $W_0=1$. **Ville's inequality** gives $\Pr(\sup_K W_K \ge 1/\alpha)
\le \mathbb{E}[W_0]\,\alpha = \alpha$. No exchangeability enters — only the conditional
mean bound $(\star)$, which is exactly what survives policy endogeneity. Simultaneity
is a union bound over $h$; the averaged-e-process version uses that a mean of
e-processes is an e-process. $\qquad\blacksquare$

*Remark (composite null).* $H_0^{(h)}$ is composite (any conditional law with $d_S \le
\varepsilon_h$). Validity holds because $\mathbb{E}[e_k\mid\mathcal F_{k-1}]\le1$ is
required for *every* member; $W^{(h)}$ is an e-process for the composite null, not
merely a martingale under one law.

---

## Theorem 2 — Anytime-valid simulation lemma

The classical simulation lemma bounds a *value* gap by an *average one-step model
error* — but at a fixed, pre-committed sample size and with the true error assumed
known. ATLAS produces a **time-uniform, empirical** version: the very e-process used
for testing also emits a running upper bound on the one-step faithfulness gap, which
then propagates through the rollout.

**Step 1 — E-process $\to$ confidence sequence on the one-step gap.** Instantiate the
construction at $h=1$. Treating $\varepsilon$ as a free parameter, the set of
tolerances not yet rejected,
$$
\mathrm{CS}_K \;=\; \Big\{\, \varepsilon \ge 0 : W^{(1)}_K(\varepsilon) < 1/\alpha \,\Big\},
$$
is, by Theorem 1 applied uniformly in $\varepsilon$, a **$(1-\alpha)$ confidence
sequence** for the running one-step gap $\delta_1(K) = \frac1K\sum_{k\le K}
\mathbb{E}[Z_k\mid\mathcal F_{k-1}]$. With method-of-mixtures / empirical-Bernstein
betting its upper edge $\hat\delta_1(K) = \sup \mathrm{CS}_K$ obeys, with probability
$\ge 1-\alpha$ *simultaneously for all $K$*,
$$
\delta_1(K) \;\le\; \hat\delta_1(K) \;=\; \bar Z_K + O\!\left(\sqrt{\frac{\hat\sigma_K^2 \log(1/\alpha)}{K}} + \frac{B\log(1/\alpha)}{K}\right),
$$
the variance-adaptive (empirical-Bernstein) rate.

**Step 2 — One-step gap $\to$ $h$-step gap.** Combine with (A3)–(A4).

> **Theorem 2 (anytime-valid simulation lemma).** Under (A1)–(A4), with probability
> $\ge 1-\alpha$, **for all rounds $t$ and all horizons $h \le H$ simultaneously**,
> the gap between the imagined $h$-step return (or predictive law) under $M$ and the
> true one under real dynamics is bounded by the running certified one-step error:
> $$
> \big|\,V^{\pi}_{M,h}(t) - V^{\pi}_{\mathrm{real},h}(t)\,\big|
> \;\le\; C(h,\gamma,L)\,\cdot\, \phi\!\big(\hat\delta_1(t)\big),
> $$
> where $\phi$ is the score-to-metric modulus (A3) and
> $$
> C(h,\gamma,L) \;=\;
> \begin{cases}
> \dfrac{\gamma\,(1-\gamma^{h})}{(1-\gamma)^2}\,R_{\max}, & \text{(discounted return form)}\\[2ex]
> R_{\max}\displaystyle\sum_{i=1}^{h} L^{\,i-1}, & \text{(Lipschitz-dynamics form).}
> \end{cases}
> $$
> Equivalently, for the *distributional* gap,
> $\rho\big(\text{law}^M_{t,h}, \text{law}^{\mathrm{real}}_{t,h}\big) \le
> \big(\sum_{i<h} L^i\big)\,\phi(\hat\delta_1(t))$.

**Proof sketch.** Step 1 is Theorem 1 read as a confidence sequence (the duality
between anytime-valid tests and confidence sequences), with the empirical-Bernstein
rate coming from mixture betting on a bounded score. Step 2 telescopes one-step error
along the rollout: writing the $h$-step law as an $h$-fold composition of one-step
kernels, the difference of compositions telescopes into a sum of one-step differences,
each amplified by the downstream Lipschitz/contraction factor; (A3) converts the
certified score gap $\hat\delta_1$ into the metric $\rho$ that the telescoping is
stated in. The union over $t$ and $h$ is free because Step 1 already holds uniformly
in $t$ and Step 2 is deterministic given $\hat\delta_1(t)$. $\qquad\blacksquare$

**Why this is the bridge.** Every quantity on the right is *observed and certified
online*: a model-based-RL practitioner reads off a live, statistically guaranteed
upper bound on how far imagined rollouts can drift from reality, and the frontier
$h^\*(t) = \max\{h : C(h,\gamma,L)\,\phi(\hat\delta_1(t)) \le \text{budget}\}$ falls
straight out of it. This is the object the world-model-eval literature currently
lacks.

---

## Theorem 3 — Detection delay under drift

Now the *alternative*: faithfulness is intact until an unknown changepoint $\tau$,
after which the one-step gap jumps by a detectable margin, $\mathbb{E}[Z_k \mid
\mathcal F_{k-1}] \ge \varepsilon_1 + \Delta$ for $k > \tau$, with post-change
conditional variance $\le \sigma^2$. A single e-process can be slow if its wealth was
depleted before $\tau$; ATLAS therefore uses an **e-detector** — the running maximum
over start times, $\bar W_K = \max_{1\le j \le K} \prod_{k=j}^{K} e_k$ (equivalently a
Shiryaev–Roberts-style sum of e-processes) — which resets statistical strength at
every candidate onset.

> **Theorem 3.** Under (A1)–(A2) and the drift alternative above, run the ATLAS
> e-detector with GRO (log-optimal, mixture/plug-in) betting and alarm at
> $T = \inf\{K : \bar W_K \ge 1/\alpha\}$.
>
> **(a) False-alarm control.** Under $H_0^{(1)}$ (no change), the e-detector controls
> the false-alarm probability / average run length: $\Pr(T < \infty \mid H_0) \le
> \alpha$ for the Ville-style variant, and $\mathbb{E}[T \mid H_0] \ge 1/\alpha$ for
> the SR-style ARL variant.
>
> **(b) Delay bound.** With probability $\ge 1-\alpha$ the detection delay after the
> true changepoint satisfies
> $$
> T - \tau \;\le\; \frac{\log(1/\alpha)}{D^\*} \,\big(1 + o(1)\big),
> \qquad
> D^\* \;=\; \sup_{\lambda \in [0,\lambda_{\max}]} \mathbb{E}_{\mathrm{post}}\!\big[\log\big(1+\lambda(Z-\varepsilon_1)\big)\big],
> $$
> the post-change **expected log-e-value** (growth rate). For the bounded score, GRO
> betting attains
> $$
> D^\* \;\ge\; \frac{\Delta^2}{2\,(\sigma^2 + B\Delta/3)},
> \qquad\text{hence}\qquad
> T - \tau \;\lesssim\; \frac{2(\sigma^2 + B\Delta/3)\,\log(1/\alpha)}{\Delta^2},
> $$
> a variance-adaptive (empirical-Bernstein) delay. This matches the Lorden/Lai lower
> bound for sequential change detection up to constants, so ATLAS is **delay-optimal**
> in order.

**Proof sketch.** (a) Each product $\prod_{k=j}^{K} e_k$ started at $j$ is a test
supermartingale under $H_0$; the e-detector's guarantee follows from the maximal /
stitched inequality for the running maximum (Ville variant) or the supermartingale
property of the SR sum (ARL variant). (b) After $\tau$, the summand started at
$j=\tau{+}1$ grows in expectation at rate $D^\*$ per round (law of large numbers for
the log-wealth under the post-change law, with predictable GRO plug-ins converging to
the optimizer); crossing $\log(1/\alpha)$ then takes $\approx \log(1/\alpha)/D^\*$
rounds. The bounded-score bound on $D^\*$ is the standard $\log(1+x)$ second-order
(Bernstein) expansion optimized over $\lambda$; matching to Lorden/Lai is the
change-detection lower bound. $\qquad\blacksquare$

**Reading.** The trust horizon $h^\*(t)$ **collapses within $O(\log(1/\alpha)/\Delta^2)$
rounds of a real dynamics shift** — the quantitative form of the headline experiment
(inject a mass/friction perturbation mid-deployment; $h^\*(t)$ drops while FVD / offline
metrics stay blind).

---

## Dependency map

| Result | Uses | Delivers |
|--------|------|----------|
| Thm 1  | Ville + conditional e-variable (A1–A2) | anytime false-revocation $\le\alpha$ |
| Thm 2  | Thm 1 as CS + proper-score modulus + rollout regularity (A1–A4) | online value/divergence-gap certificate |
| Thm 3  | e-detector + GRO growth rate (A1–A2) | order-optimal detection delay |

All three avoid any i.i.d./exchangeability assumption; validity rests solely on the
conditional-mean null $(\star)$ under the deployment filtration, which is what makes
ATLAS sound on endogenous, policy-dependent world-model data.
