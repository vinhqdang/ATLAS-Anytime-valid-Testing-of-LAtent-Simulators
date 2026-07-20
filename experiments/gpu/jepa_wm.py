"""JEPA-style latent world model — frozen pretrained encoder + learned dynamics.

The small conv-autoencoder (`neural_wm.py`) overfits the training subjects on real
video, so its in-distribution excess is non-stationary across subjects. A JEPA-style
model fixes this: the representation is a **frozen, pretrained image encoder**
(ImageNet ResNet-18 features, PCA-reduced) that generalizes across subjects out of
the box, so the in-distribution (walking) latent statistics are stable; only a small
latent **dynamics** net is learned. Predictions then fail specifically when the
*motion* changes (running), not because the encoder overfit.

Same `encode/predict/excess_stream` interface as the other world models, so it drops
straight into `experiments.gpu.run_neural`. GPU strongly recommended (ResNet forward
per frame). This is the "stronger WM" step of roadmap phase 4c.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Dyn(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))

    def forward(self, z):
        return z + self.net(z)


class JEPALatentWM:
    def __init__(self, latent_dim=32, epochs_dyn=40, batch=128, lr=1e-3, seed=0):
        self.d = latent_dim
        self.epochs_dyn = epochs_dyn
        self.batch = batch
        self.lr = lr
        self.dev = _device()
        torch.manual_seed(seed)
        from torchvision import models
        try:
            w = models.ResNet18_Weights.IMAGENET1K_V1
            net = models.resnet18(weights=w)
        except Exception:
            net = models.resnet18(weights="IMAGENET1K_V1")
        # frozen features up to layer3 (256-d, keeps some locality) + global pool
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3,
            nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(self.dev).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    # ------------------------------------------------------------ features
    @torch.no_grad()
    def _feats(self, seq):
        x = torch.tensor(seq, dtype=torch.float32, device=self.dev)      # (T,H,W)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)                            # gray->3ch
        x = torch.nn.functional.interpolate(x, size=96, mode="bilinear",
                                            align_corners=False)
        x = (x - self._mean) / self._std
        out = []
        for i in range(0, len(x), 64):
            out.append(self.backbone(x[i:i + 64]).cpu().numpy())
        return np.concatenate(out, 0)                                    # (T, F)

    # ------------------------------------------------------------------ fit
    def fit(self, sequences, horizons=(1, 2, 3)):
        F = np.concatenate([self._feats(s) for s in sequences], 0)
        self._fmean = F.mean(0)
        Fc = F - self._fmean
        # PCA on frozen features -> latent
        U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
        self.components_ = Vt[: self.d]
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
        self.Sigma_h = {}
        for h in horizons:
            res = []
            for z in Zs:
                if len(z) <= h:
                    continue
                res.append(z[h:] - self._rollout(np.asarray(z[:-h]), h))
            R = np.concatenate(res, 0)
            self.Sigma_h[h] = np.cov(R.T) + 1e-4 * np.eye(self.d)
        return self

    # ----------------------------------------------------- covariance recalibration
    def recalibrate(self, sequences, horizons, shrink=0.2):
        """Re-estimate the per-horizon predictive covariance on *held-out*
        in-distribution residuals so the calibration excess is centred near zero
        out-of-sample; shrinks toward a diagonal target for stability."""
        for h in horizons:
            res = []
            for s in sequences:
                z = self.encode(s)
                if len(z) <= h:
                    continue
                res.append(z[h:] - self._rollout(np.asarray(z[:-h]), h))
            if not res:
                continue
            R = np.concatenate(res, 0)
            emp = np.cov(R.T)
            diag = np.diag(np.diag(emp))
            self.Sigma_h[h] = ((1 - shrink) * emp + shrink * diag
                               + 1e-4 * np.eye(self.d))
        return self

    # -------------------------------------------------------------- helpers
    def encode(self, seq):
        return (self._feats(seq) - self._fmean) @ self.components_.T

    @torch.no_grad()
    def _rollout(self, z, h):
        z = torch.tensor(z, dtype=torch.float32, device=self.dev)
        for _ in range(h):
            z = self.dyn(z)
        return z.cpu().numpy()

    def predict(self, z0, h):
        mu = self._rollout(z0[None], h)[0]
        return mu, self.Sigma_h[h]

    def excess_stream(self, seq, horizons, B=8.0):
        z = self.encode(seq)
        by_time = [dict() for _ in range(z.shape[0])]
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
