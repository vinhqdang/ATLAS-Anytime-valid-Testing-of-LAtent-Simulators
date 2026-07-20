# ATLAS — GPU Phase (neural world model)

The CPU study monitors a *linear* latent world model. This phase swaps in a **learned
neural** latent world model — a convolutional autoencoder + latent-dynamics network
(`neural_wm.py`, Dreamer/JEPA-lite) — behind the **same** `encode/predict/
excess_stream` interface, so the ATLAS monitoring code is unchanged. A full DreamerV3
or a JEPA model drops in the same way; ATLAS's guarantees do not depend on the WM.

```bash
python -m experiments.gpu.run_neural --dataset mnist   # or --dataset kth
python -m experiments.gpu.run_neural --dataset mnist --smoke   # tiny CPU test
```

The model auto-selects CUDA when available (`torch.cuda.is_available()`).

## Running on a GPU

There is **no official Google Colab CLI** for driving Colab's GPU from an external
machine — Colab GPUs are only reachable from inside a Colab notebook session. So the
GPU run is launched from a notebook or a GPU host, not remote-controlled from here.
Three practical paths:

### 1. Google Colab (free GPU) — recommended
Open a new Colab notebook (Runtime → Change runtime type → **GPU**) and run:

```python
!git clone https://github.com/vinhqdang/ATLAS-Anytime-valid-Testing-of-LAtent-Simulators.git
%cd ATLAS-Anytime-valid-Testing-of-LAtent-Simulators
!pip install -q -r requirements.txt torch
!python -m experiments.real_data.download
!python -m experiments.gpu.run_neural --dataset mnist
!python -m experiments.gpu.run_neural --dataset kth
from google.colab import files          # download the figures
import glob
for f in glob.glob('figures/neural_*_full.png'): files.download(f)
```

### 2. Kaggle Notebooks (free GPU)
Same commands in a GPU-enabled Kaggle notebook; commit the output figures.

### 3. Any GPU host / cloud VM
`gcloud`/AWS support device-flow auth (a URL + code you approve). On the VM:
`git clone … && pip install -r requirements.txt torch && python -m experiments.gpu.run_neural …`.

## Scaling up to DreamerV3 / JEPA / DMC / Atari

`neural_wm.py` is the reference neural WM. To use a stronger model, implement the
same three methods and point the experiment at it:

- `encode(seq) -> (T, d)` latent codes,
- `predict(z0, h) -> (mu, Sigma)` Gaussian h-step predictive,
- `excess_stream(seq, horizons) -> [ {h: Z}, ... ]` calibration excess.

For DMC/Atari with an injected mid-deployment dynamics shift (mass/friction), roll
out DreamerV3's RSSM to get `(mu, Sigma)` per horizon and feed the same frontier.
This is the last planned experiment (README roadmap phase 4c).

## Outputs

`figures/neural_<dataset>_full.png` — per-horizon wealth and the `h*(t)` collapse for
the neural WM (analogous to the linear-WM figures in `figures/`, `--smoke` variants
are git-ignored).

## GPU run findings (Colab T4, official google-colab-cli)

All runs executed on a real Colab **Tesla T4** via the official CLI
(`colab new --gpu T4`, `colab exec`, `colab download`).

- **Moving MNIST, conv-AE neural WM — clean (`figures/neural_mnist_gpu.png`).** Wealth
  flat during the native-rate period (valid, no false revocation); `h*(t)` collapses
  3→0 after the 2× speed shift. ATLAS's guarantees hold behind a *learned* WM.
- **KTH walking→running, JEPA WM — clean (`figures/neural_kth_jepa_gpu.png`).** The
  JEPA-style model (frozen ImageNet ResNet-18 encoder + learned latent dynamics,
  `jepa_wm.py`) keeps `h*(t)=3` through the entire walking period (valid null) and
  **collapses to 0** once running begins (all of h=1,2,3 revoked). The frozen
  pretrained encoder generalizes across subjects, so the in-distribution walking
  excess is low and stable (≈0.1 at h=1) while running is clearly separated (≈1.1) —
  exactly the property the small conv-AE lacked.
- **KTH with the small conv-AE WM — WM-limited.** The conv-AE overfits the training
  subjects; its held-out walking excess is variable at the clip level and overlaps
  running, so no tolerance cleanly separates them. This is a **world-model quality**
  limitation, resolved by the stronger JEPA representation above (and the low-capacity
  linear WM, which generalizes uniformly, is also clean: `figures/kth_frontier.png`).

Takeaway: the GPU pipeline works end-to-end via the official CLI, and a
representation that generalizes across subjects (JEPA's frozen encoder) is what makes
ATLAS's certificate clean on real video. A full DreamerV3 RSSM drops in behind the
same `encode/predict/excess_stream` interface.

### Gotcha: the Colab kernel persists across `colab exec` calls

Each `colab exec` shares one long-lived Python kernel, so `import` returns **cached**
modules — re-shipping edited code has no effect unless you purge them first. The VM
bootstrap therefore runs, after extracting the code and before importing:

```python
for m in [k for k in list(sys.modules) if k.split('.')[0] in ("experiments","atlas")]:
    del sys.modules[m]
```

(Or `colab stop && colab new` for a fresh kernel.) Missing this silently reruns stale
code — worth knowing for any iterative use of the CLI.
