"""Kelly sizing for one round of the barrier game.

A round puts notional f*W into the index at entry and leaves (1-f)*W in cash; it is held until the
barrier exit (or the cap). Per round, wealth is multiplied by

    M(f) = 1 + f R + carry(f, T) - f c - f (1 + R) c
    carry = (1 - f) r_f T/252            if f <= 1  (cash earns r_f)
          = -(f - 1)(r_f + s) T/252      if f > 1   (borrowing costs r_f + spread s)

with R the exit return, T the exit day and c the one-way trading cost. Kelly maximises
g(f) = E[ln M(f)]; growth per year is (g + ln(1 + r_f/252)) / ((E[T] + 1)/252), the +1 being the
cash day between an exit and the next entry.

Outcomes with probability below PROB_FLOOR per trade are ignored: under log utility any
positive-probability wipe-out gives g = -inf, and the grid's open-ended overshoot bin would otherwise
decide the answer. The ruin cliff (smallest f that a 1-in-a-million loss would wipe out) is reported
separately.
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import config as C
from . import theory as T
from .models import GBMModel, MertonModel

PROB_FLOOR = 1e-9
CLIFF_PROB = 1e-6
DAY_BIN = 10                     # exit days are grouped in 10-day bins for speed (carry is ~linear)


@dataclass
class Outcomes:
    """Compressed (R, T, prob) sample of one round."""
    R: np.ndarray
    T: np.ndarray
    p: np.ndarray
    outcome: np.ndarray

    @property
    def mean_T(self) -> float:
        return float((self.p * self.T).sum())


def compress(xd, floor: float = PROB_FLOOR) -> Outcomes:
    """Outcomes sample of an ExitDistribution (a predict.Forecast already carries one)."""
    if hasattr(xd, "compressed"):
        return xd.compressed
    keep = xd.prob >= floor
    day, lr, pr = xd.day[keep], xd.log_ret[keep], xd.prob[keep]
    oc = xd.outcome[keep]
    dbin = (day - 1) // DAY_BIN
    key = np.stack([oc.astype(np.int64), dbin, np.round(lr * 1e6).astype(np.int64)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    p = np.bincount(inv, weights=pr)
    Tm = np.bincount(inv, weights=pr * day) / p
    R = np.expm1(uniq[:, 2] / 1e6)
    return Outcomes(R=R, T=Tm, p=p / p.sum(), outcome=uniq[:, 0].astype(np.int8))


def multiplier(o: Outcomes, f: float, rf: float, spread: float = C.BORROW_SPREAD,
               cost: float = C.TRADE_COST) -> np.ndarray:
    yrs = o.T / C.TRADING_DAYS
    carry = (1 - f) * rf * yrs if f <= 1 else -(f - 1) * (rf + spread) * yrs
    return 1 + f * o.R + carry - f * cost - f * (1 + o.R) * cost


def growth(o: Outcomes, f: float, rf: float, **kw) -> float:
    """Expected log growth per round, g(f)."""
    m = multiplier(o, f, rf, **kw)
    return float((o.p * np.log(m)).sum()) if (m > 0).all() else -np.inf


def growth_per_year(o: Outcomes, f: float, rf: float, **kw) -> float:
    g = growth(o, f, rf, **kw)
    return (g + np.log1p(rf / C.TRADING_DAYS)) * C.TRADING_DAYS / (o.mean_T + 1)


def growth_curve(o: Outcomes, fs, rf: float, **kw) -> np.ndarray:
    return np.array([growth_per_year(o, f, rf, **kw) for f in fs])


def feasible_fmax(R, T, rf: float, spread: float = C.BORROW_SPREAD, cost: float = C.TRADE_COST,
                  axis=None):
    """Largest f with M(f) > 0 for every outcome (the wipe-out edge).

    For f <= 1, M > 0 whenever R > -1. For f > 1, M(f) = 1 + (r_f+s) y + f k with
    k = R - (r_f+s) y - c (2 + R), so outcomes with k < 0 bound f by (1 + (r_f+s) y) / -k."""
    y = np.asarray(T) / C.TRADING_DAYS
    k = np.asarray(R) - (rf + spread) * y - cost * (2 + np.asarray(R))
    with np.errstate(divide="ignore"):
        lim = np.where(k < 0, (1 + (rf + spread) * y) / -np.where(k < 0, k, -1.0), np.inf)
    return np.maximum(np.min(lim, axis=axis), 1.0)


def kelly_fraction(o: Outcomes, rf: float, fmax: float = C.F_GRID_MAX, tol: float = 1e-3, **kw) -> float:
    """argmax_f g(f) on [0, min(fmax, wipe-out edge)] by golden-section search.

    g is concave and finite below the wipe-out edge, so the search is restricted to that range
    (otherwise it can converge onto the edge and step past it into g = -inf)."""
    gr = (np.sqrt(5) - 1) / 2
    lo, hi = 0.0, float(min(fmax, feasible_fmax(o.R, o.T, rf, **kw) * (1 - 1e-9)))
    g = lambda f: growth(o, f, rf, **kw)
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    g1, g2 = g(x1), g(x2)
    while hi - lo > tol:
        if g1 >= g2:
            hi, x2, g2 = x2, x1, g1
            x1 = hi - gr * (hi - lo)
            g1 = g(x1)
        else:
            lo, x1, g1 = x1, x2, g2
            x2 = lo + gr * (hi - lo)
            g2 = g(x2)
    f = 0.5 * (lo + hi)
    return 0.0 if g(0.0) >= g(f) else float(f)


def cash_breakeven(o: Outcomes, rf: float, f_star: float, fmax: float = 4 * C.F_GRID_MAX,
                   **kw) -> float | None:
    """Smallest f > f* whose growth falls back to the all-cash growth (None if beyond fmax)."""
    g0 = growth(o, 0.0, rf, **kw)
    fs = np.arange(max(f_star, 0.0) + 0.01, fmax, 0.01)
    for f in fs:
        if growth(o, f, rf, **kw) <= g0:
            return float(f)
    return None


def ruin_cliff(xd: T.ExitDistribution, prob: float = CLIFF_PROB) -> float:
    """Leverage at which the worst loss with probability >= `prob` per trade wipes out the account."""
    r, p = xd.ret_ruin()
    pr = p * xd.p_ruin
    o = np.argsort(r)
    tail = np.cumsum(pr[o])
    worst = r[o][np.searchsorted(tail, prob)]
    return float(1 / -worst) if worst < 0 else np.inf


def swap_ruin_fills(xd: T.ExitDistribution, fills: np.ndarray) -> Outcomes:
    """Same exit timing and outcome probabilities, but stopped trades lose according to `fills`
    (an empirical sample of simple returns) instead of the model's overshoot distribution."""
    o = compress(xd)
    ruin = o.outcome == T.RUIN
    vals, cnt = np.unique(np.round(fills, 4), return_counts=True)
    q = cnt / cnt.sum()
    R = np.concatenate([o.R[~ruin], np.repeat(vals[None, :], ruin.sum(), 0).ravel()])
    Tm = np.concatenate([o.T[~ruin], np.repeat(o.T[ruin], len(vals))])
    p = np.concatenate([o.p[~ruin], (o.p[ruin][:, None] * q[None, :]).ravel()])
    oc = np.concatenate([o.outcome[~ruin], np.full(ruin.sum() * len(vals), T.RUIN, np.int8)])
    return Outcomes(R=R, T=Tm, p=p, outcome=oc)


# ------------------------------------------------------------------ lookup tables for Q5 traders
MU_AXIS = np.round(np.arange(-0.20, 0.5001, 0.025), 4)
SIGMA_AXIS = np.round(np.arange(0.06, 0.5001, 0.02), 4)
LAM_AXIS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 22.0])
TABLE_GRID_N = 600


def _gbm_from(mu, sigma):
    return GBMModel(nu=mu - 0.5 * sigma ** 2, sigma=sigma, n_obs=C.ROLLING_DAYS)


def merton_from(mu, sigma, lam, mu_J, sigma_J):
    """Merton model with total annual arithmetic drift `mu` and total daily variance sigma^2 dt
    (moment-matched like MertonModel.calibrate), for given jump intensity and jump sizes."""
    dt = C.DT
    var_d = sigma ** 2 * dt - lam * dt * (mu_J ** 2 + sigma_J ** 2)
    if var_d <= 0:
        return None
    sigma_d = np.sqrt(var_d / dt)
    # log mean per day: nu dt = (mu - sigma^2/2) dt ; nu_d dt + lam dt mu_J = nu dt
    nu = mu - 0.5 * sigma ** 2
    nu_d = nu - lam * mu_J
    return MertonModel(nu_d=nu_d, sigma_d=sigma_d, lam=lam, mu_J=mu_J, sigma_J=sigma_J)


def _table_solve(model):
    if model is None:
        return None
    return compress(T.exit_distribution(model, n=TABLE_GRID_N))


@dataclass
class KellyTable:
    """f*(mu, sigma) for GBM and f*(mu, sigma, lambda) for Merton (jump sizes fixed), per r_f."""
    rf: float
    gbm: np.ndarray            # (mu, sigma)
    merton: np.ndarray         # (mu, sigma, lam); NaN where the jumps exceed the variance
    mu_J: float
    sigma_J: float

    def __post_init__(self):
        self._g = RegularGridInterpolator((MU_AXIS, SIGMA_AXIS), self.gbm, bounds_error=False,
                                          fill_value=None)
        mer = self.merton.copy()
        # fill infeasible cells with the GBM value at the same (mu, sigma) (jumps -> 0 limit)
        bad = np.isnan(mer)
        mer[bad] = np.broadcast_to(self.gbm[:, :, None], mer.shape)[bad]
        self._m = RegularGridInterpolator((MU_AXIS, SIGMA_AXIS, LAM_AXIS), mer, bounds_error=False,
                                          fill_value=None)

    @staticmethod
    def _clip(x, axis):
        return np.clip(x, axis[0], axis[-1])

    def f_gbm(self, mu, sigma) -> np.ndarray:
        pts = np.column_stack([self._clip(np.atleast_1d(mu), MU_AXIS),
                               self._clip(np.atleast_1d(sigma), SIGMA_AXIS)])
        return np.maximum(self._g(pts), 0.0)

    def f_merton(self, mu, sigma, lam) -> np.ndarray:
        pts = np.column_stack([self._clip(np.atleast_1d(mu), MU_AXIS),
                               self._clip(np.atleast_1d(sigma), SIGMA_AXIS),
                               self._clip(np.atleast_1d(lam), LAM_AXIS)])
        return np.maximum(self._m(pts), 0.0)


def build_kelly_table(rf: float, mu_J: float, sigma_J: float, workers: int = 8) -> KellyTable:
    gbm_models = [_gbm_from(m, s) for m in MU_AXIS for s in SIGMA_AXIS]
    mer_models = [merton_from(m, s, l, mu_J, sigma_J) for m in MU_AXIS for s in SIGMA_AXIS
                  for l in LAM_AXIS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        og = list(ex.map(_table_solve, gbm_models, chunksize=8))
        om = list(ex.map(_table_solve, mer_models, chunksize=8))
    fg = np.array([kelly_fraction(o, rf) for o in og]).reshape(len(MU_AXIS), len(SIGMA_AXIS))
    fm = np.array([np.nan if o is None else kelly_fraction(o, rf) for o in om]).reshape(
        len(MU_AXIS), len(SIGMA_AXIS), len(LAM_AXIS))
    return KellyTable(rf=rf, gbm=fg, merton=fm, mu_J=mu_J, sigma_J=sigma_J)


def load_or_build_table(ds, mu_J: float, sigma_J: float, refresh: bool = False) -> KellyTable:
    path = C.CACHE_DIR / (f"kelly_table_v2_{ds.key}_rf{ds.rf}_s{C.BORROW_SPREAD}_c{C.TRADE_COST}"
                          f"_J{mu_J:.4f}_{sigma_J:.4f}.pkl")
    if path.exists() and not refresh:
        t = pickle.loads(path.read_bytes())
        return KellyTable(rf=t["rf"], gbm=t["gbm"], merton=t["merton"], mu_J=t["mu_J"], sigma_J=t["sigma_J"])
    t = build_kelly_table(ds.rf, mu_J, sigma_J)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"rf": t.rf, "gbm": t.gbm, "merton": t.merton, "mu_J": t.mu_J,
                                   "sigma_J": t.sigma_J}))
    return t
