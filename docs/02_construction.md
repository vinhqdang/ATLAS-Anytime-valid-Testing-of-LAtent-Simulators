# ATLAS — E-process Construction and the Faithfulness Frontier

Prerequisite: [`01_problem_setup.md`](01_problem_setup.md). All expectations are
conditional on the filtration $(\mathcal{F}_k)$ defined there.

---

## 1. E-variables, test supermartingales, e-processes

Recall the objects we build on:

- An **e-variable** for $H_0$ is a random variable $e \ge 0$ with
  $\mathbb{E}_{H_0}[e] \le 1$.
- A **test supermartingale** is a process $W_0 = 1$, $W_k = \prod_{j\le k} e_j$
  with each $e_j$ a *conditional* e-variable: $\mathbb{E}[e_j \mid \mathcal{F}_{j-1}]
  \le 1$ under $H_0$. Then $(W_k)$ is a nonnegative supermartingale under $H_0$.
- An **e-process** is a nonnegative process dominated by a test supermartingale
  under every distribution in $H_0$ (composite nulls). Ville's inequality applies
  to all of these and is the engine of anytime validity.

## 2. The per-horizon betting e-variable

Fix horizon $h$. Let $R_k$ be a **predictable calibrated reference** predictor
($\mathcal{F}_{k-1}$-measurable) — e.g. a marginal climatology, a conformalized
baseline, or a frozen earlier checkpoint. Define the per-round **score advantage of
the reference over the model**:
$$
Z_k \;=\; S\!\big(P^{M}_{s_k,h}, Y^{(h)}_{s_k}\big) \;-\; S\!\big(R_k, Y^{(h)}_{s_k}\big)
\;\in\; [-B,\, B].
$$
Under $\varepsilon_h$-faithfulness, the model must not be worse than the reference by
more than the tolerance:
$$
H_0^{(h)}: \qquad \mathbb{E}\big[\, Z_k \mid \mathcal{F}_{k-1}\,\big] \;\le\; \varepsilon_h
\quad \text{for all } k. \tag{$\star$}
$$
(This is the operational form of the divergence null of §4 in the setup: choosing
$R_k$ to be a *calibrated* competitor makes $(\star)$ equivalent to a bound on
$d_S(Q,P^M)$ up to the reference's own irreducible score; with $R_k$ the
Bayes-optimal predictor, $\mathbb{E}[Z_k\mid\mathcal F_{k-1}] = d_S(Q_{s_k,h},
P^M_{s_k,h})$ exactly.)

**Betting e-variable (Waudby-Smith–Ramdas form).** With a predictable bet
$\lambda_k \in [0, \lambda_{\max}]$, $\lambda_{\max} = 1/(B - \varepsilon_h)$,
$$
\boxed{\;e_k \;=\; 1 + \lambda_k\,\big(Z_k - \varepsilon_h\big)\;}
\qquad\Longrightarrow\qquad
\mathbb{E}\big[e_k \mid \mathcal{F}_{k-1}\big] \;=\; 1 + \lambda_k\big(\mathbb{E}[Z_k\mid\mathcal F_{k-1}] - \varepsilon_h\big) \;\le\; 1
$$
under $(\star)$. Nonnegativity holds because $Z_k - \varepsilon_h \ge -(B+\varepsilon_h)$
and $\lambda_k \le 1/(B-\varepsilon_h)$ keep $e_k \ge 0$ on the range of $Z_k$
(clip $\lambda_{\max}$ to $1/(B+\varepsilon_h)$ for the two-sided-safe variant). The
wealth
$$
W^{(h)}_K \;=\; \prod_{k=1}^{K} e_k \;=\; \prod_{k=1}^{K}\big(1 + \lambda_k(Z_k-\varepsilon_h)\big)
$$
is a test supermartingale under $H_0^{(h)}$: an e-process for the composite,
non-stationary, policy-dependent null.

**Log-optimal / GRO betting.** Choosing $\lambda_k$ to maximize expected log-wealth
gives the growth-rate-optimal (GRO) bet. Since the post-change law is unknown, we use
predictable plug-ins:

- **aGRAPA / mixture** — a running-mean plug-in $\hat\lambda_k \approx
  (\bar Z_{k-1}-\varepsilon_h)_+ / \widehat{\mathrm{Var}}$, truncated to
  $[0,\lambda_{\max}]$; or
- **method of mixtures** — $W^{(h)}_K = \int_0^{\lambda_{\max}} \prod_k (1+\lambda(Z_k-\varepsilon_h))\,\mathrm{d}\nu(\lambda)$
  for a prior $\nu$, which is parameter-free and yields the empirical-Bernstein rate
  used in Theorems 2–3.

Both are predictable, so validity is untouched; they affect only power / delay.

## 3. A calibration-native instantiation (PIT betting)

For scalar or low-dimensional targets an equivalent construction bets directly on
calibration. Let $F_{s,h}$ be the model's predictive CDF and $U_k = F_{s_k,h}(Y^{(h)}_{s_k})$
the PIT. Under exact conditional calibration $U_k \mid \mathcal{F}_{k-1} \sim
\mathrm{Unif}[0,1]$, so any $\mathcal{F}_{k-1}$-measurable **betting density**
$g_k \ge 0$ with $\int_0^1 g_k = 1$ gives an e-variable $e_k = g_k(U_k)$
($\mathbb{E}[e_k\mid\mathcal F_{k-1}] = \int_0^1 g_k = 1$). Wealth accrues exactly
when PITs are non-uniform (mis-calibration). The $\varepsilon_h$-tolerant null relaxes
uniformity to a density band $g^{\mathrm{true}} \in [1-\varepsilon_h, 1+\varepsilon_h]$;
the resulting e-variable is $e_k = g_k(U_k)/(1+\varepsilon_h)$ over $g_k$ supported on
the band. This is the natural score for models that expose a calibrated CDF.

## 4. The faithfulness frontier $h^\*(t)$

Run one e-process $W^{(h)}$ per horizon $h \in \{1,\dots,H\}$. Declare horizon $h$
**revoked** at the first round its wealth crosses the threshold:
$$
\tau_h \;=\; \inf\{\, k : W^{(h)}_k \ge 1/\alpha \,\}, \qquad
\text{horizon } h \text{ certified at round } t \iff k(t) < \tau_h .
$$
The **faithfulness frontier** is
$$
h^\*(t) \;=\; \max\Big\{\, h : W^{(h)}_{k(t)} < 1/\alpha \,\Big\}
\quad (\text{with } h^\*(t)=0 \text{ if even } h{=}1 \text{ is revoked}).
$$

**Staircase structure.** Because error compounds, $d_S$ is (typically) non-decreasing
in $h$, so larger-$h$ e-processes cross first and $h^\*(t)$ is a downward staircase in
$t$ as drift sets in. ATLAS does **not** assume monotonicity for validity — each
horizon is tested independently — but monotonicity, when it holds, can be *imposed*
to share statistical strength (isotonic smoothing of $\{W^{(h)}\}$, an e-process
transform that preserves validity).

## 5. Combining horizons

Two distinct aggregation needs, two valid operators:

- **"Is the model faithful at *any* / *all* horizons?"** — the arithmetic mean
  $\bar e_k = \frac1H \sum_h e^{(h)}_k$ is an e-variable under arbitrary dependence
  across horizons (an average of e-values is an e-value), so $\prod_k \bar e_k$ is a
  single valid e-process for the pooled null. No independence between horizons is
  needed — critical, since the horizon streams share the same underlying trajectory.
- **The frontier $h^\*(t)$** does *not* pool; it reports the max certified horizon.
  For *simultaneous* validity of the whole frontier (Theorem 1), either run each
  horizon at level $\alpha/H$ (union bound, $+\log H$ cost) or drive the frontier off
  the averaged e-process; both are stated in [`03_theorems.md`](03_theorems.md).

## 6. From certificate to controller

Because $h^\*(t)$ is anytime-valid, a planner may **clip imagination depth** to
$h^\*(t)$ at every round with the guarantee that it never (except w.p. $\le \alpha$
over the whole run) plans on a horizon that is truly untrustworthy. This is the
step that turns ATLAS from a passive monitor into an intervention:
`plan_horizon(t) = min(H_desired, h*(t))`.
