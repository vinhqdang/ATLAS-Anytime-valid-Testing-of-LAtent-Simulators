"""Real-data experiment 2 — KTH Actions, walking -> running (gait DYNAMICS shift).

A latent world model is fit on *walking* video and deployed; the subject then
switches to *running* (faster leg dynamics the walking-trained model cannot track).
ATLAS's frontier ``h*(t)`` holds while walking and collapses once running begins.

Run: ``python -m experiments.real_data.exp_kth``
"""

from __future__ import annotations

from experiments.real_data.datasets import load_kth
from experiments.real_data.kth_common import run_shift

HORIZONS = [1, 2, 3]
ALPHA = 0.05
LATENT = 32
SIZE = 32
SEQ = 40


def main():
    print("=" * 66)
    print("Real-data Exp 2 — KTH Actions (walking -> running: dynamics shift)")
    print("=" * 66)
    walking = load_kth("walking", size=SIZE, max_videos=60, seq_len=SEQ)
    running = load_kth("running", size=SIZE, max_videos=40, seq_len=SEQ)
    print(f"loaded {len(walking)} walking, {len(running)} running sequences")
    return run_shift(walking, running, horizons=HORIZONS, latent=LATENT,
                     eps_mult=0.5, alpha=ALPHA, label_id="walking",
                     label_od="running", shift_label="walking->running",
                     tag="kth",
                     title="ATLAS per-horizon wealth — KTH (walking -> running)")


if __name__ == "__main__":
    main()
