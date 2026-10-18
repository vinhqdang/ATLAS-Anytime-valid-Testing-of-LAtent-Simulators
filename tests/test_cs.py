"""Coverage of the running-average-of-conditional-means confidence sequence."""

import numpy as np

from atlas.cs import AvgCondMeanCS, WindowedCS


def _stream(rng, T, kind):
    """Bounded stream in [0,1] whose conditional mean varies adaptively over time."""
    mus = np.empty(T)
    xs = np.empty(T)
    prev = 0.5
    for k in range(T):
        if kind == "trend":
            mu = 0.2 + 0.6 * k / T
        elif kind == "switch":
            mu = 0.2 if k < T // 2 else 0.7
        else:                       # adaptive: mean depends on the previous outcome
            mu = 0.25 + 0.5 * prev
        mus[k] = mu
        xs[k] = float(rng.random() < mu)
        prev = xs[k]
    return mus, xs


def test_cs_covers_running_average_of_conditional_means():
    rng = np.random.default_rng(0)
    alpha, T, N = 0.1, 400, 300
    for kind in ("trend", "switch", "adaptive"):
        bad = 0
        for _ in range(N):
            mus, xs = _stream(rng, T, kind)
            cs = AvgCondMeanCS(0.0, 1.0, alpha=alpha)
            run = np.cumsum(mus) / np.arange(1, T + 1)
            viol = False
            for k in range(T):
                cs.update(xs[k])
                if run[k] > cs.upper + 1e-12:
                    viol = True
                    break
            bad += viol
        assert bad / N <= alpha, (kind, bad / N)


def test_windowed_cs_tracks_a_shift():
    rng = np.random.default_rng(1)
    T, w = 600, 100
    mus, xs = _stream(rng, T, "switch")
    cs = WindowedCS(0.0, 1.0, window=w, alpha=0.1)
    ups = []
    for k in range(T):
        cs.update(xs[k])
        ups.append(cs.upper)
    assert ups[T // 2 - 1] < 0.65         # before the shift (true mean 0.2)
    assert ups[-1] > 0.9                  # after the shift (true mean 0.7)
    assert ups[-1] >= xs[-w:].mean()      # an upper bound on the window average
