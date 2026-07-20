"""A learned *neural* latent world model (Dreamer/JEPA-lite), GPU-ready.

Same ``encode`` / ``predict`` / ``excess_stream`` interface as
``experiments.real_data.latent_wm.LatentLinearWM``, so it drops straight into the
existing ATLAS real-data experiments — swap the WM, keep the monitoring code. This
is the CPU/GPU stand-in for a full DreamerV3/JEPA latent model; ATLAS's guarantees
are unchanged behind the interface.

Design (stable and simple, avoids JEPA collapse):
  1. a convolutional autoencoder is trained on frames (reconstruction) to get a
     meaningful latent code,
  2. the encoder is frozen and a latent dynamics net ``f`` (MLP) is trained to
     predict the next latent (JEPA-style latent prediction),
  3. per-horizon predictive covariances are estimated empirically from training
     residuals, giving a Gaussian h-step predictive ``N(f^h(z), Sigma_h)``.

Runs on CPU (small config, for smoke tests) or GPU (scale up ``latent_dim`` /
``epochs`` / data). Device is auto-selected.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ConvAE(nn.Module):
    def __init__(self, img_size=32, latent_dim=32, ch=1):
        super().__init__()
        h = img_size // 8
        self.h = h
        self.enc = nn.Sequential(
            nn.Conv2d(ch, 16, 4, 2, 1), nn.ReLU(),      # /2
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),      # /4
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),      # /8
        )
        self.to_z = nn.Linear(64 * h * h, latent_dim)
        self.from_z = nn.Linear(latent_dim, 64 * h * h)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, ch, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.to_z(self.enc(x).flatten(1))

    def decode(self, z):
        return self.dec(self.from_z(z).view(-1, 64, self.h, self.h))


class _Dyn(nn.Module):
    def __init__(self, latent_dim=32, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z):
        return z + self.net(z)          # residual dynamics (stable)


class NeuralLatentWM:
    def __init__(self, latent_dim=32, img_size=32, epochs_ae=8, epochs_dyn=8,
                 batch=128, lr=1e-3, seed=0):
        self.d = latent_dim
        self.img = img_size
        self.epochs_ae = epochs_ae
        self.epochs_dyn = epochs_dyn
        self.batch = batch
        self.lr = lr
        self.dev = _device()
        torch.manual_seed(seed)

    # ------------------------------------------------------------------ fit
    def fit(self, sequences, horizons=(1, 2, 3)):
        frames = np.concatenate([s.reshape(-1, 1, self.img, self.img)
                                 for s in sequences], axis=0)
        X = torch.tensor(frames, dtype=torch.float32)
        self.ae = _ConvAE(self.img, self.d).to(self.dev)
        opt = torch.optim.Adam(self.ae.parameters(), lr=self.lr)
        # 1) train autoencoder (reconstruction)
        for _ in range(self.epochs_ae):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch):
                xb = X[perm[i:i + self.batch]].to(self.dev)
                opt.zero_grad()
                loss = ((self.ae.decode(self.ae.encode(xb)) - xb) ** 2).mean()
                loss.backward(); opt.step()
        # 2) freeze encoder, encode sequences, train latent dynamics
        for p in self.ae.parameters():
            p.requires_grad_(False)
        Zs = [self.encode(s) for s in sequences]
        Z0 = np.concatenate([z[:-1] for z in Zs], 0)
        Z1 = np.concatenate([z[1:] for z in Zs], 0)
        self.dyn = _Dyn(self.d).to(self.dev)
        opt = torch.optim.Adam(self.dyn.parameters(), lr=self.lr)
        z0 = torch.tensor(Z0, dtype=torch.float32)
        z1 = torch.tensor(Z1, dtype=torch.float32)
        for _ in range(self.epochs_dyn):
            perm = torch.randperm(len(z0))
            for i in range(0, len(z0), self.batch):
                idx = perm[i:i + self.batch]
                a, b = z0[idx].to(self.dev), z1[idx].to(self.dev)
                opt.zero_grad()
                loss = ((self.dyn(a) - b) ** 2).mean()
                loss.backward(); opt.step()
        # 3) empirical per-horizon predictive covariance from training residuals
        self.Sigma_h = {}
        for h in horizons:
            res = []
            for z in Zs:
                if len(z) <= h:
                    continue
                mu = self._rollout(torch.tensor(z[:-h], dtype=torch.float32), h)
                res.append(z[h:] - mu)
            R = np.concatenate(res, 0)
            self.Sigma_h[h] = np.cov(R.T) + 1e-4 * np.eye(self.d)
        return self

    # -------------------------------------------------------------- helpers
    @torch.no_grad()
    def encode(self, seq):
        x = torch.tensor(seq.reshape(-1, 1, self.img, self.img),
                         dtype=torch.float32).to(self.dev)
        return self.ae.encode(x).cpu().numpy()

    @torch.no_grad()
    def _rollout(self, z, h):
        z = z.to(self.dev)
        for _ in range(h):
            z = self.dyn(z)
        return z.cpu().numpy()

    def predict(self, z0, h):
        mu = self._rollout(torch.tensor(z0[None], dtype=torch.float32), h)[0]
        return mu, self.Sigma_h[h]

    # ----------------------------------------------------- calibration excess
    def excess_stream(self, seq, horizons, B=8.0):
        z = self.encode(seq)
        T = z.shape[0] - 1
        by_time = [dict() for _ in range(T + 1)]
        d = self.d
        for h in horizons:
            for s in range(0, len(z) - h, h):
                mu, Sigma = self.predict(z[s], h)
                diff = z[s + h] - mu
                mahal2 = float(diff @ np.linalg.solve(Sigma, diff))
                by_time[s + h][h] = float(np.clip(mahal2 / d - 1.0, -1.0, B))
        return by_time

    def raw_excess(self, seqs, horizons, B=None):
        acc = {h: [] for h in horizons}
        d = self.d
        for seq in seqs:
            z = self.encode(seq)
            for h in horizons:
                for s in range(0, len(z) - h, h):
                    mu, Sigma = self.predict(z[s], h)
                    diff = z[s + h] - mu
                    e = float(diff @ np.linalg.solve(Sigma, diff)) / d - 1.0
                    acc[h].append(np.clip(e, -1.0, B) if B is not None else e)
        return {h: np.array(v) for h, v in acc.items()}
