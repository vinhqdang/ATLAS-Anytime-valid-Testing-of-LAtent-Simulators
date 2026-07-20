"""Run the full ATLAS simulation-study suite and write a results summary.

    python -m experiments.run_all
"""

from __future__ import annotations

import os
import json

from experiments import exp1_validity, exp2_frontier, exp3_delay

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("\n### Running ATLAS simulation study (Theorems 1-3) ###\n")
    r1 = exp1_validity.main()
    print()
    r2 = exp2_frontier.main()
    print()
    r3 = exp3_delay.main()

    summary = {"exp1_validity": r1, "exp2_frontier": r2, "exp3_delay": r3}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {os.path.abspath(os.path.join(RESULTS, 'summary.json'))}")


if __name__ == "__main__":
    main()
