# ATLAS — Real-World Data Experiments

ATLAS applied to **real image-sequence datasets**, monitoring a learned latent world
model (PCA + linear latent dynamics — a lightweight but genuine latent simulator;
see `experiments/real_data/latent_wm.py`) under a real distribution shift. The same
`atlas/` e-process code certifies faithfulness here as in the synthetic study — this
doubles as an end-to-end test of ATLAS on real data. A Dreamer/JEPA latent model
would drop in behind the identical interface (the GPU phase).

Reproduce:

```bash
pip install -r requirements.txt
python -m experiments.real_data.download     # ~1.4 GB into data/ (git-ignored)
python -m experiments.real_data.exp_mnist
python -m experiments.real_data.exp_kth
python -m experiments.real_data.exp_kth_domain
```

## Datasets & shift types

| # | dataset | source | shift | kind | figures |
|---|---------|--------|-------|------|---------|
| 1 | Moving MNIST | U. Toronto video benchmark | native → 2× speed | dynamics | `mnist_frames.png`, `mnist_frontier.png` |
| 2 | KTH Actions | KTH CVAP (real human video) | **walking → running** | dynamics (gait) | `kth_frames.png`, `kth_frontier.png` |
| 3 | KTH Actions | KTH CVAP (real human video) | **outdoor → indoor** (same action) | domain / appearance | `kth_domain_frames.png`, `kth_domain_frontier.png` |

`*_frames.png` are sample frames from each regime (manuscript images). Three
distinct shift types across two datasets exercise ATLAS on real data.

## What the experiments show

For each experiment the latent WM is fit on the in-distribution regime; ATLAS's
tolerance `eps_h` is calibrated on held-out in-distribution clips (on the same
**clipped** calibration statistic the e-process consumes — real-video excess is
heavy-tailed); then deployment runs in-distribution and switches to the shifted
regime at a known point.

- **Moving MNIST (dynamics)** — the per-horizon wealth is flat during the
  native-rate period (no depletion, valid), `h*(t)=8`; after the 2× speed shift,
  horizons h=1,2,3 cross the threshold and **`h*(t)` collapses to 0**, while an
  offline metric frozen at its pre-deployment value stays at 8 (blind).
- **KTH walking → running (dynamics)** — `h*(t)=3` throughout walking (no false
  revocation), then **collapses to 0** once running begins (all of h=1,2,3 revoked).
- **KTH outdoor → indoor (domain)** — a subtler shift: `h*(t)=3` outdoors, then a
  **partial collapse to 2** indoors (only the longest horizon h=3 is revoked). The
  graded response (full collapse for strong dynamics shifts, partial for a subtle
  domain shift) is itself informative.

## Honest caveats

- The PCA + linear-dynamics WM is intentionally simple (CPU-only, no training loop).
  On real video it is only moderately calibrated, and long horizons (h≥5 on KTH) are
  not reliably certifiable with it — an honest limitation that motivates the
  GPU-based Dreamer/JEPA phase, where ATLAS's guarantees are unchanged.
- KTH in-distribution deploy clips are shuffled so subject-to-subject appearance
  differences do not form a spurious trend across the (crude, not perfectly
  stationary) model.
- `eps_h` is calibrated to the in-distribution mean excess so the pre-shift wealth is
  (approximately) a martingale — the null is "as faithful as in-distribution", and
  ATLAS certifies *departures* from it.
