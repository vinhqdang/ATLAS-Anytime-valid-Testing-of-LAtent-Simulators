# ATLAS — Notation

| Symbol | Meaning |
|--------|---------|
| $t$ | deployment round (real time) |
| $h,\ H$ | rollout horizon; maximum horizon |
| $k,\ s_k$ | resolution index for a horizon; issue-time of the $k$-th resolved prediction |
| $\mathcal{X},\mathcal{O},\mathcal{Y}$ | state / observation / prediction-target spaces |
| $x_t, o_t, z_t$ | true state, observation, latent code at round $t$ |
| $a_{t:t+h-1}$ | action sequence conditioning the rollout |
| $M,\ \pi$ | world model (latent simulator); policy / planner |
| $P^M_{t,h}$ | model's action-conditioned $h$-step predictive distribution |
| $Q_{s,h}$ | true conditional law of the $h$-step outcome given $\mathcal F_s$ |
| $R_k$ | predictable calibrated reference predictor |
| $Y^{(h)}_t$ | realized $h$-step outcome resolving the round-$t$, horizon-$h$ prediction |
| $S,\ B$ | proper scoring rule (negatively oriented); its bound, $S\in[0,B]$ |
| $d_S(Q,P)$ | score divergence, $\ge 0$, $=0$ iff $P=Q$ |
| $\varepsilon_h$ | faithfulness tolerance at horizon $h$ |
| $\rho,\ \phi$ | probability metric; score-to-metric modulus, $\rho\le\phi(d_S)$ |
| $\mathcal{F}_k$ | deployment filtration (all info to the $k$-th resolution) |
| $Z_k$ | per-round score advantage of $R_k$ over $M$: $S(P^M)-S(R_k)$ |
| $\lambda_k,\ \lambda_{\max}$ | predictable betting fraction; its cap |
| $e_k$ | per-round e-variable, $\mathbb E[e_k\mid\mathcal F_{k-1}]\le1$ under $H_0$ |
| $W^{(h)}_K$ | wealth / test supermartingale (e-process) for horizon $h$ |
| $\bar W_K$ | e-detector (running max / SR sum over start times) |
| $U_k$ | probability integral transform $F_{s_k,h}(Y^{(h)}_{s_k})$ |
| $\alpha$ | error level; reject when wealth $\ge 1/\alpha$ |
| $\tau_h,\ h^\*(t)$ | revocation time for horizon $h$; faithfulness frontier |
| $\delta_1(K),\ \hat\delta_1$ | running one-step gap; its anytime upper confidence edge |
| $\tau,\ \Delta,\ \sigma^2$ | changepoint; post-change margin; post-change variance |
| $D^\*$ | GRO post-change growth rate (expected log-e-value) |
| $C(h,\gamma,L)$ | rollout amplification factor (discount / Lipschitz form) |
| $V^\pi_{M,h}$ | $h$-step value of $\pi$ under model $M$ |
