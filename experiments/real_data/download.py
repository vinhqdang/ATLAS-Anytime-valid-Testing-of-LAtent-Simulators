"""Download the real-world datasets used by the ATLAS real-data experiments.

    python -m experiments.real_data.download

Datasets (public):
  * Moving MNIST  (U. Toronto)          -> data/mnist_test_seq.npy   (~780 MB)
  * KTH Actions   (KTH CVAP), walking+running -> data/kth_*.zip       (~390 MB)

Data is written to ``data/`` (git-ignored). Re-running skips files already present.
"""

from __future__ import annotations

import os
import urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

FILES = {
    "mnist_test_seq.npy":
        "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
    "kth_walking.zip": "https://www.csc.kth.se/cvap/actions/walking.zip",
    "kth_running.zip": "https://www.csc.kth.se/cvap/actions/running.zip",
}


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(DATA, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[skip] {name} already present ({os.path.getsize(dst)//1_000_000} MB)")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"       done ({os.path.getsize(dst)//1_000_000} MB)")
    print("all datasets ready in", os.path.abspath(DATA))


if __name__ == "__main__":
    main()
