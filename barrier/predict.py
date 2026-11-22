"""Exact capped forecasts for every monthly rolling fit (shared by Q3 and Q4).

theory.exit_distribution gives the full joint (exit day, exit return) distribution; a Forecast keeps
only what the questions use (outcome probabilities, the daily exit-day distribution, the stop-fill
distribution, and the compressed outcome sample Kelly needs), so the cache stays small.
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import data as D
from . import theory as T
from .models import calibration_grid

MODELS = ("RandomWalk", "GBM", "Merton")


@dataclass
class Forecast:
    p_win: float
    p_ruin: float
    p_timeout: float
    tau_pmf_: np.ndarray        # P(exit on day t), t = 1..cap (float32)
    ruin_ret: np.ndarray        # simple returns of stopped trades (support)
    ruin_prob: np.ndarray       # their probabilities, normalised to 1
    compressed: object          # kelly.Outcomes

    def tau_pmf(self) -> np.ndarray:
        return self.tau_pmf_.astype(float)

    @property
    def tau_mean(self) -> float:
        return float((self.tau_pmf() * np.arange(1, len(self.tau_pmf_) + 1)).sum())

    def ret_ruin(self):
        return self.ruin_ret, self.ruin_prob

    def ret_mean_ruin(self) -> float:
        return float((self.ruin_ret * self.ruin_prob).sum())

    def ret_quantile_ruin(self, q: float) -> float:
        o = np.argsort(self.ruin_ret)
        return float(self.ruin_ret[o][np.searchsorted(np.cumsum(self.ruin_prob[o]), q)])


def to_forecast(xd: T.ExitDistribution) -> Forecast:
    from .kelly import compress
    r, p = xd.ret_ruin()
    rk = np.round(r, 6)
    u, inv = np.unique(rk, return_inverse=True)
    pr = np.bincount(inv.ravel(), weights=p)
    keep = pr > 1e-12
    return Forecast(xd.p_win, xd.p_ruin, xd.p_timeout, xd.tau_pmf().astype(np.float32),
                    u[keep], pr[keep] / pr[keep].sum(), compress(xd))


def _solve(model):
    return to_forecast(T.exit_distribution(model, cap=C.CAP_DAYS, n=C.GRID_N))


def _cache_path(ds):
    return C.CACHE_DIR / f"forecast_v2_{ds.key}_w{C.ROLLING_DAYS}_n{C.GRID_N}_cap{C.CAP_DAYS}.pkl"


def load_grid_and_predictions(ds, workers: int = 8, refresh: bool = False):
    """(full, test, grid, preds) where preds[model][k] is the Forecast of monthly fit k."""
    full, _, test, _ = D.load(ds)
    grid = calibration_grid(D.log_returns(full), test.index)
    path = _cache_path(ds)
    if path.exists() and not refresh:
        preds = pickle.loads(path.read_bytes())
        if all(len(preds[m]) == len(grid.fits) for m in MODELS):
            return full, test, grid, preds
    jobs = [(m, k) for m in MODELS for k in range(len(grid.fits))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_solve, [grid.fits[k][m] for m, k in jobs], chunksize=4))
    preds = {m: [None] * len(grid.fits) for m in MODELS}
    for (m, k), fc in zip(jobs, res):
        preds[m][k] = fc
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(preds))
    return full, test, grid, preds
