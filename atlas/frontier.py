"""The faithfulness frontier h*(t): horizon-indexed e-processes.

Runs one :class:`~atlas.eprocess.EProcess` per horizon and reports the largest
horizon not yet revoked, exactly as in ``docs/02_construction.md`` §4:

    h*(t) = max { h in {1..H} : W^(h)_{k(t)} < 1/alpha }.

For *simultaneous* frontier validity over all faithful horizons (Theorem 1,
simultaneous form) each horizon is tested at ``alpha / H`` (union bound), which is
the default; pass ``simultaneous=False`` to test each at ``alpha`` marginally.
"""

from __future__ import annotations

import numpy as np

from .eprocess import EProcess, BettingCS

__all__ = ["FaithfulnessFrontier", "amplification", "trust_horizon_from_gap"]


def amplification(h, gamma, R_max=1.0):
    """Simulation-lemma rollout amplification ``C(h) = gamma (1-gamma^h)/(1-gamma)^2 R_max``.

    Increasing in ``h`` by construction (docs/03_theorems.md, Theorem 2), so a fixed
    certified one-step gap yields a larger certified value gap at longer horizons.
    """
    return gamma * (1 - gamma ** h) / (1 - gamma) ** 2 * R_max


def trust_horizon_from_gap(delta1, gamma, budget, R_max=1.0, phi=np.sqrt,
                           h_grid=None):
    """Theorem-2 trust horizon: largest ``h`` with ``C(h) phi(delta1) <= budget``.

    This is the anytime-valid simulation lemma turned into a controller: given the
    certified one-step gap ``delta1`` (the upper edge of the BettingCS), a planner
    may legally imagine up to this horizon while keeping the certified value gap
    within ``budget``. As ``delta1`` jumps at a dynamics shift, the trust horizon
    collapses.
    """
    if h_grid is None:
        h_grid = np.arange(1, 101)
    val = phi(max(delta1, 0.0))
    ok = [h for h in h_grid if amplification(h, gamma, R_max) * val <= budget]
    return int(max(ok)) if ok else 0


class FaithfulnessFrontier:
    """Manage per-horizon e-processes and track the trust horizon h*(t).

    Parameters
    ----------
    horizons : sequence of int
        The horizons to certify, e.g. ``[1, 2, 5, 10, 20]``.
    eps : float or dict
        Tolerance; a scalar (same for all horizons) or a ``{h: eps_h}`` map.
    z_lo, z_hi : float
        Bounds on the per-round score advantage ``Z`` (assumption A1).
    alpha : float
        Target level for the whole frontier.
    simultaneous : bool
        If True (default) each horizon runs at ``alpha/len(horizons)`` so the whole
        frontier is valid simultaneously (Theorem 1, simultaneous form).
    track_cs : bool
        If True, also maintain a :class:`BettingCS` per horizon for the
        anytime-valid one-step-gap bound (Theorem 2 feeds off horizon 1).
    """

    def __init__(self, horizons, eps, z_lo, z_hi, alpha=0.05,
                 simultaneous=True, track_cs=False, monotone=True):
        self.horizons = list(horizons)
        self.monotone = monotone
        H = len(self.horizons)
        self.alpha = float(alpha)
        self.per_alpha = alpha / H if simultaneous else alpha
        if np.isscalar(eps):
            self.eps = {h: float(eps) for h in self.horizons}
        else:
            self.eps = {h: float(eps[h]) for h in self.horizons}
        self.ep = {
            h: EProcess(self.eps[h], z_lo, z_hi, alpha=self.per_alpha)
            for h in self.horizons
        }
        self.track_cs = track_cs
        if track_cs:
            self.cs = {
                h: BettingCS(z_lo, z_hi, alpha=self.per_alpha)
                for h in self.horizons
            }
        self.t = 0
        self.h_star_trace = []

    def update(self, z_by_h):
        """Feed the per-horizon score advantages for one resolved round.

        ``z_by_h`` maps horizon -> scalar ``Z``. Horizons absent from the map are
        left unchanged this round (e.g. not yet resolved).
        """
        for h, z in z_by_h.items():
            self.ep[h].update(z)
            if self.track_cs:
                self.cs[h].update(z)
        self.t += 1
        self.h_star_trace.append(self.h_star)
        return self.h_star

    @property
    def h_star(self):
        """The trust horizon.

        With ``monotone=True`` (default): the largest ``h`` such that *all* horizons
        up to ``h`` are still certified (trust up to the first broken horizon) — the
        physically correct notion for clipping imagination depth, since a broken
        short horizon implies broken longer ones. With ``monotone=False``: the
        largest individual horizon not yet revoked.
        """
        if self.monotone:
            h = 0
            for hh in sorted(self.horizons):
                if self.ep[hh].rejected:
                    break
                h = hh
            return h
        alive = [h for h in self.horizons if not self.ep[h].rejected]
        return max(alive) if alive else 0

    @property
    def revoked(self):
        """Set of currently-revoked horizons."""
        return {h for h in self.horizons if self.ep[h].rejected}

    def wealth(self, h):
        return self.ep[h].wealth

    def log_wealth_trace(self, h):
        return np.asarray(self.ep[h].history)

    def one_step_gap_bound(self):
        """Certified anytime upper bound ``hat_delta_1`` (needs ``track_cs=True``)."""
        if not self.track_cs:
            raise RuntimeError("construct with track_cs=True to use the simulation lemma")
        h1 = min(self.horizons)
        return self.cs[h1].upper

    def value_gap_bound(self, h, gamma, R_max, phi):
        """Theorem-2 anytime-valid simulation-lemma bound on the h-step value gap.

        C(h,gamma) * phi(hat_delta_1), with C the discounted amplification factor
        ``gamma (1-gamma^h)/(1-gamma)^2 * R_max``.
        """
        delta1 = self.one_step_gap_bound()
        C = gamma * (1 - gamma ** h) / (1 - gamma) ** 2 * R_max
        return C * phi(max(delta1, 0.0))
