"""ATLAS — Anytime-valid Testing of LAtent Simulators.

Sequential certification of world-model faithfulness via e-process betting.

Public API
----------
scores      : proper scoring rules (energy, CRPS, Gaussian NLL, Mahalanobis)
eprocess    : one-sided testing-by-betting e-process + two-sided betting CS + e-detector
frontier    : horizon-indexed e-processes and the faithfulness frontier h*(t)
simulator   : a controllable linear-Gaussian latent world model with drift injection

The code implements the constructions in ``docs/`` verbatim:
  - the conditional betting e-variable  e_k = 1 + lambda_k (Z_k - eps)   (02_construction.md, Thm 1)
  - the confidence-sequence dual used by the anytime-valid simulation lemma (Thm 2)
  - the change-detection e-detector                                       (Thm 3)
"""

from . import scores, eprocess, frontier, simulator
from .eprocess import EProcess, BettingCS, EDetector
from .frontier import FaithfulnessFrontier

__all__ = [
    "scores",
    "eprocess",
    "frontier",
    "simulator",
    "EProcess",
    "BettingCS",
    "EDetector",
    "FaithfulnessFrontier",
]

__version__ = "0.1.0"
