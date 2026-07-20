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
