"""Real-data experiment 3 — KTH Actions, outdoor -> indoor (DOMAIN shift).

A distinct real-world failure mode: same *action* (walking), but the recording
condition changes. A world model fit on outdoor walking (scenario d1) is deployed
and the feed switches to indoor walking (scenario d4) — an appearance / domain shift
rather than a dynamics shift. ATLAS's frontier ``h*(t)`` collapses when the domain
changes, showing the certificate is not specific to dynamics drift.

Run: ``python -m experiments.real_data.exp_kth_domain``
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
    print("Real-data Exp 3 — KTH Actions (outdoor -> indoor: domain shift)")
    print("=" * 66)
    outdoor = load_kth("walking", size=SIZE, max_videos=60, seq_len=SEQ, scenario=1)
    indoor = load_kth("walking", size=SIZE, max_videos=60, seq_len=SEQ, scenario=4)
    print(f"loaded {len(outdoor)} outdoor (d1), {len(indoor)} indoor (d4) sequences")
    return run_shift(outdoor, indoor, horizons=HORIZONS, latent=LATENT,
                     eps_mult=0.05, alpha=ALPHA, label_id="outdoor",
                     label_od="indoor", shift_label="outdoor->indoor",
                     tag="kth_domain",
                     title="ATLAS per-horizon wealth — KTH (outdoor -> indoor)")


if __name__ == "__main__":
    main()
