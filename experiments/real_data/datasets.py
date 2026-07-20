"""Loaders for the real-world sequence datasets used in the ATLAS experiments.

Two datasets, both yielding image sequences (for manuscript figures) and a natural
distribution shift on which to exercise ATLAS:

* Moving MNIST — the standard video-prediction benchmark. Native frame rate is the
  in-distribution regime; temporally subsampling frames (2x playback speed) is a
  controllable *dynamics shift* a native-rate model cannot track.
* KTH Actions — real human-action videos. A model fit on ``walking`` is deployed
  and the subject switches to ``running`` (a genuine real-world dynamics shift).

Frames are converted to grayscale and downscaled; sequences are returned as
float arrays in [0, 1].
"""

from __future__ import annotations

import os
import glob
import zipfile
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# ------------------------------------------------------------------ Moving MNIST
def load_moving_mnist(path=None, n=None):
    """Return Moving MNIST as ``(N, T, H, W)`` float in [0, 1].

    The Toronto ``.npy`` has shape ``(T=20, N=10000, 64, 64)`` uint8.
    """
    path = path or os.path.join(DATA, "mnist_test_seq.npy")
    arr = np.load(path)                       # (20, 10000, 64, 64)
    arr = np.transpose(arr, (1, 0, 2, 3))     # (N, T, H, W)
    if n is not None:
        arr = arr[:n]
    return arr.astype(np.float32) / 255.0


def moving_mnist_speed_shift(frames, factor=2):
    """Return a speed-shifted copy by temporal subsampling (every ``factor`` frames).

    Doubling playback speed changes the effective dynamics; a native-rate world
    model becomes unfaithful. Output length is ``T // factor``.
    """
    return frames[:, ::factor, :, :]


# ------------------------------------------------------------------ KTH Actions
def _read_avi_gray(path, size=32, max_frames=60):
    import imageio.v3 as iio
    frames = []
    try:
        for i, fr in enumerate(iio.imiter(path, plugin="FFMPEG")):
            if i >= max_frames:
                break
            g = fr.mean(axis=2) if fr.ndim == 3 else fr    # to grayscale
            # center-crop to a square that is a multiple of `size`
            h, w = g.shape
            s = (min(h, w) // size) * size
            g = g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            # block-mean downscale (anti-aliased) to size x size
            g = g.reshape(size, s // size, size, s // size).mean(axis=(1, 3))
            frames.append(g.astype(np.float32) / 255.0)
    except Exception as e:                                  # pragma: no cover
        print(f"  [warn] failed to read {os.path.basename(path)}: {e}")
    return np.array(frames) if frames else None


def load_kth(action, zip_path=None, size=32, max_videos=40, max_frames=60,
             seq_len=40):
    """Load KTH ``action`` videos as a list of ``(seq_len, size, size)`` sequences.

    Decodes the ``.avi`` files inside the action zip, downscales to grayscale, and
    chops each into fixed-length sequences.
    """
    zip_path = zip_path or os.path.join(DATA, f"kth_{action}.zip")
    seqs = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".avi")]
        names = sorted(names)[:max_videos]
        tmp = os.path.join(DATA, "_kth_tmp")
        os.makedirs(tmp, exist_ok=True)
        for n in names:
            out = os.path.join(tmp, os.path.basename(n))
            with zf.open(n) as src, open(out, "wb") as dst:
                dst.write(src.read())
            vid = _read_avi_gray(out, size=size, max_frames=max_frames)
            os.remove(out)
            if vid is None or len(vid) < seq_len:
                continue
            # one non-overlapping sequence per video (keeps them independent)
            seqs.append(vid[:seq_len])
    return seqs


def sample_frames_figure(seqs_a, seqs_b, out_path, label_a, label_b, k=6):
    """Save a manuscript figure: a row of frames from regime A over regime B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, k, figsize=(1.5 * k, 3.2))

    def most_dynamic(seqs):
        if not isinstance(seqs, list):
            return seqs
        # pick the clip with the largest frame-to-frame motion (person visible)
        motion = [np.abs(np.diff(s, axis=0)).mean() for s in seqs]
        return seqs[int(np.argmax(motion))]

    sa = most_dynamic(seqs_a)
    sb = most_dynamic(seqs_b)
    ia = np.linspace(0, len(sa) - 1, k).astype(int)
    ib = np.linspace(0, len(sb) - 1, k).astype(int)
    for j in range(k):
        axes[0, j].imshow(sa[ia[j]], cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(sb[ib[j]], cmap="gray"); axes[1, j].axis("off")
        axes[0, j].set_title(f"t={ia[j]}", fontsize=8)
    axes[0, 0].set_ylabel(label_a); axes[1, 0].set_ylabel(label_b)
    fig.suptitle(f"{label_a}  (top)   vs   {label_b}  (bottom)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
