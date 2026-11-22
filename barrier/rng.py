"""Reproducible, independent random streams keyed by a tuple of labels."""
from __future__ import annotations

import numpy as np

from . import config as C


def _hash(k) -> int:
    # Polynomial hash: order-sensitive, so ("12",) and ("21",) get different seeds.
    return sum(ord(ch) * 131 ** i for i, ch in enumerate(str(k))) % 2 ** 31


def rng_for(*key) -> np.random.Generator:
    """Independent stream per (dataset, purpose, ...) key."""
    return np.random.default_rng([C.SEED] + [_hash(k) for k in key])
