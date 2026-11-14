"""Stationary block bootstrap of real daily log returns (Politis & Romano, 1994).

Each simulated day either continues the current block (next real day) or, with probability
1/mean_block, jumps to a uniformly random real day. Blocks therefore have geometric lengths with mean
`mean_block`, which keeps volatility clustering, crashes and fat tails up to that time scale.
"""
from __future__ import annotations

import numpy as np


def stationary_bootstrap(source: np.ndarray, n_paths: int, length: int, mean_block: float,
                         rng: np.random.Generator) -> np.ndarray:
    n = len(source)
    p_new = 1.0 / mean_block
    idx = np.empty((n_paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    jump = rng.random((n_paths, length)) < p_new
    fresh = rng.integers(0, n, (n_paths, length))
    for t in range(1, length):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return source[idx]
