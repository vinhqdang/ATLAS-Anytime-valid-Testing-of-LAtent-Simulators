"""Competitor faithfulness-drift monitors for head-to-head comparison with ATLAS.

All monitors consume the same per-round score stream ``Z_k`` (the one-step
calibration excess) and emit an alarm time. They represent the families that recent
(2024--2026) work on online distribution-shift monitoring draws on:

* ``FixedThreshold``  -- an offline metric: alarm when a rolling-mean score exceeds a
  threshold fixed before deployment (no online guarantee).
* ``SlidingZTest``    -- a sliding-window two-sample mean-shift test against a
  reference (the online-shift-detection / sliding-window family); repeated looks
  inflate its false-alarm rate.
* ``CUSUM``           -- the classical cumulative-sum change detector.
* ATLAS (``atlas.eprocess.EProcess``) -- the anytime-valid e-process.

Unlike ATLAS, whose false-alarm probability is controlled by construction at every
round (Ville), the baselines require a tuned threshold and only control the
false-alarm rate at the horizon/setting they were tuned for; under longer monitoring
(more looks) they break. ``exp4_baselines.py`` quantifies this.
"""

from __future__ import annotations

import numpy as np


class FixedThreshold:
    def __init__(self, thresh, window=20):
        self.thresh, self.window = thresh, window
        self.buf = []

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if len(self.buf) < self.window:        # only test on a full window
            return False
        return np.mean(self.buf) >= self.thresh


class SlidingZTest:
    def __init__(self, ref_mean, ref_sd, thresh, window=20):
        self.m0, self.s0, self.thresh, self.window = ref_mean, ref_sd, thresh, window
        self.buf = []

    def step(self, z):
        self.buf.append(z)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if len(self.buf) < self.window:
            return False
        zstat = (np.mean(self.buf) - self.m0) / (self.s0 / np.sqrt(self.window) + 1e-12)
        return zstat >= self.thresh


class CUSUM:
    def __init__(self, ref_mean, slack, thresh):
        self.m0, self.slack, self.thresh = ref_mean, slack, thresh
        self.S = 0.0

    def step(self, z):
        self.S = max(0.0, self.S + (z - self.m0 - self.slack))
        return self.S >= self.thresh


def run_monitor(monitor, stream, cp=None):
    """Return the first alarm index over ``stream`` (or ``None``). If ``cp`` is given,
    return the post-change delay (alarm - cp), ignoring pre-cp alarms as false."""
    for t, z in enumerate(stream):
        if monitor.step(z):
            if cp is None:
                return t                      # null: first (false) alarm
            if t >= cp:
                return t - cp                 # detection delay
    return None
