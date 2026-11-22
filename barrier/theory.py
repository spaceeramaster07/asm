"""Step 2: exact closed-form first-passage results.

All functions work in log space X_t = ln(S_t/S0), lower barrier a < 0 < b upper barrier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as C

WIN, RUIN, TIMEOUT = 1, 0, -1   # same codes as engine.py (kept here to avoid a circular import)


# ---------------------------------------------------------------- Random walk (lattice)
def rw_first_passage(p: float, A: int, B: int) -> tuple[float, float]:
    """Gambler's ruin for a +/-1 walk starting at 0, absorbed at -A (ruin) or +B (win).

    Returns (p_win, E[tau] in steps).

    Corrected formula, with r = q/p:
        p_win = (1 - r^A) / (1 - r^(A+B))
    (The draft had q/p and p/q swapped; with A=B=1, p=0.6 it returned 0.4 instead of 0.6.)
    E[tau] from Wald's identity: E[S_tau] = (p - q) E[tau].
    """
    q = 1.0 - p
    if abs(p - q) < 1e-12:
        return A / (A + B), float(A * B)
    r = q / p
    p_win = (1 - r ** A) / (1 - r ** (A + B))
    e_tau = (B * p_win - A * (1 - p_win)) / (p - q)
    return p_win, e_tau


def rw_first_passage_linear_system(p: float, A: int, B: int) -> tuple[float, float]:
    """Brute-force check: solve the absorbing Markov chain directly (independent of the formula)."""
    n = A + B - 1                      # interior states -A+1 .. B-1
    q = 1 - p
    M = np.eye(n)
    rhs_win = np.zeros(n)
    rhs_time = np.ones(n)
    for i in range(n):
        if i + 1 < n:
            M[i, i + 1] -= p
        else:
            rhs_win[i] += p            # stepping up from B-1 hits B
        if i - 1 >= 0:
            M[i, i - 1] -= q
    P = np.linalg.solve(M, rhs_win)
    T = np.linalg.solve(M, rhs_time)
    k0 = A - 1                          # index of state 0
    return float(P[k0]), float(T[k0])


# ---------------------------------------------------------------- Brownian motion with drift
def gbm_first_passage(nu: float, sigma: float, a: float = C.A_LOG, b: float = C.B_LOG
                      ) -> tuple[float, float]:
    """Continuous-time exit of X_t = nu t + sigma W_t from (a, b).

    nu, sigma are annualised. Returns (p_win, E[tau] in years).
    Uses the exponential martingale exp(gamma X), gamma = -2 nu / sigma^2, and Wald.
    """
    if abs(nu) < 1e-12:
        return -a / (b - a), -a * b / sigma ** 2
    g = -2.0 * nu / sigma ** 2
    # p = (1 - e^{g a}) / (e^{g b} - e^{g a}), written with expm1 for stability
    p_win = -math.expm1(g * a) / (math.exp(g * b) - math.exp(g * a))
    e_tau = (b * p_win + a * (1 - p_win)) / nu
    return p_win, e_tau


def bgk_barriers(sigma: float, a: float = C.A_LOG, b: float = C.B_LOG,
                 dt: float = C.DT) -> tuple[float, float]:
    """Broadie-Glasserman-Kou shift for monitoring once per dt: push barriers outward
    by beta * sigma * sqrt(dt). This is an approximation (derived for one barrier,
    error o(sqrt(dt))), not an exact result - Q1 should measure its residual bias."""
    shift = C.BETA_BGK * sigma * math.sqrt(dt)
    return a - shift, b + shift


def gbm_first_passage_daily(nu: float, sigma: float, a: float = C.A_LOG,
                            b: float = C.B_LOG) -> tuple[float, float]:
    """BGK-corrected prediction for GBM checked only at daily closes. (p_win, E[tau] years)."""
    a2, b2 = bgk_barriers(sigma, a, b)
    return gbm_first_passage(nu, sigma, a2, b2)


def expected_daily_overshoot(sigma: float, dt: float = C.DT) -> float:
    """Asymptotic mean overshoot (log units) of a Gaussian random walk past a distant barrier.
    Same beta constant as BGK: E[overshoot] ~ beta * sigma * sqrt(dt)."""
    return C.BETA_BGK * sigma * math.sqrt(dt)


# ---------------------------------------------------------------- Markov-chain grid solver
@dataclass
class ExitDistribution:
    """Joint distribution of (exit day, exit log return) for one trade, as weighted support points.

    outcome: WIN / RUIN / TIMEOUT per support point (engine constants); day in 1..cap."""
    day: np.ndarray
    log_ret: np.ndarray
    outcome: np.ndarray
    prob: np.ndarray
    cap: int | None

    def _mask(self, o):
        return self.outcome == o

    @property
    def p_win(self) -> float:
        return float(self.prob[self._mask(WIN)].sum())

    @property
    def p_ruin(self) -> float:
        return float(self.prob[self._mask(RUIN)].sum())

    @property
    def p_timeout(self) -> float:
        return float(self.prob[self._mask(TIMEOUT)].sum())

    @property
    def tau_mean(self) -> float:
        return float((self.prob * self.day).sum())

    def tau_pmf(self) -> np.ndarray:
        """P(exit on day t), t = 1..cap (timeouts count as exiting on day cap)."""
        return np.bincount(self.day, weights=self.prob, minlength=self.cap + 1)[1:]

    def ret_ruin(self) -> tuple[np.ndarray, np.ndarray]:
        """(simple returns, probabilities normalised to 1) of ruin exits."""
        m = self._mask(RUIN)
        return np.expm1(self.log_ret[m]), self.prob[m] / max(self.prob[m].sum(), 1e-300)

    def ret_mean_ruin(self) -> float:
        r, p = self.ret_ruin()
        return float((r * p).sum())

    def ret_quantile_ruin(self, q: float) -> float:
        r, p = self.ret_ruin()
        o = np.argsort(r)
        return float(r[o][np.searchsorted(np.cumsum(p[o]), q)])

    def summary(self) -> dict:
        return {"p_win": self.p_win, "p_ruin": self.p_ruin, "p_timeout": self.p_timeout,
                "tau_mean": self.tau_mean, "ret_mean_ruin": self.ret_mean_ruin(),
                "ret_p01_ruin": self.ret_quantile_ruin(0.01)}


def _start_weights(x: np.ndarray) -> np.ndarray:
    """Put unit mass at log price 0 by splitting it linearly between the two nearest cell centres."""
    v = np.zeros(len(x))
    j = int(np.searchsorted(x, 0.0))
    w = (0.0 - x[j - 1]) / (x[j] - x[j - 1])
    v[j - 1], v[j] = 1 - w, w
    return v


def grid_first_passage(daily_cdf, cap: int | None = None, a: float = C.A_LOG, b: float = C.B_LOG,
                       n: int = C.GRID_N, overshoot_bin: float = 0.0025,
                       overshoot_max: float = 0.40):
    """Exact (up to the grid) first passage for a model checked once a day at the close.

    The log price lives on n cells of width h covering (a, b); one day moves it by an i.i.d. increment
    with CDF `daily_cdf`. Cell-to-cell probabilities are CDF differences, so no probability mass is lost.
    A step that lands at or beyond a barrier exits there; how far beyond (the overshoot) is recorded in
    bins of `overshoot_bin` log units.

    cap=None: returns (p_win, E[tau] in days) from one linear solve.
    cap=int : returns an ExitDistribution (time-stepped `cap` days; open trades are TIMEOUTs).
    """
    from scipy.linalg import toeplitz

    h = (b - a) / n
    x = a + (np.arange(n) + 0.5) * h
    d = np.arange(-(n - 1), n)
    K = daily_cdf((d + 0.5) * h) - daily_cdf((d - 0.5) * h)     # P(move by d cells)
    Q = toeplitz(K[n - 1::-1], K[n - 1:])                       # Q[i, j] = K[j - i]
    up = 1 - daily_cdf(b - x)                                   # P(exit at b from cell i)
    dn = daily_cdf(a - x)                                       # P(exit at a from cell i)
    v0 = _start_weights(x)

    if cap is None:
        M = np.eye(n) - Q
        p = np.linalg.solve(M, up)
        t = np.linalg.solve(M, np.ones(n))
        return float(v0 @ p), float(v0 @ t)

    nb = int(round(overshoot_max / overshoot_bin))
    k = np.arange(nb + 1) * overshoot_bin
    # P(land in overshoot bin j beyond the barrier); last bin is open-ended
    F_dn = daily_cdf((a - x)[:, None] - k[None, :])             # P(y <= a - x - k)
    O_dn = F_dn[:, :-1] - F_dn[:, 1:]
    O_dn[:, -1] += F_dn[:, -1]
    F_up = 1 - daily_cdf((b - x)[:, None] + k[None, :])         # P(y > b - x + k)
    O_up = F_up[:, :-1] - F_up[:, 1:]
    O_up[:, -1] += F_up[:, -1]
    mid = (np.arange(nb) + 0.5) * overshoot_bin

    win_p = np.empty((cap, nb))
    ruin_p = np.empty((cap, nb))
    v = v0
    for t in range(cap):
        win_p[t] = v @ O_up
        ruin_p[t] = v @ O_dn
        v = v @ Q
    days = np.arange(1, cap + 1)
    parts = [
        (np.repeat(days, nb), np.tile(b + mid, cap), np.full(cap * nb, WIN), win_p.ravel()),
        (np.repeat(days, nb), np.tile(a - mid, cap), np.full(cap * nb, RUIN), ruin_p.ravel()),
        (np.full(n, cap), x, np.full(n, TIMEOUT), v),
    ]
    day, lr, oc, pr = (np.concatenate(z) for z in zip(*parts))
    keep = pr > 1e-14
    return ExitDistribution(day[keep].astype(np.int64), lr[keep], oc[keep].astype(np.int8),
                            pr[keep], cap)


def lattice_exit_distribution(p: float, A: int, B: int, dx: float, cap: int) -> ExitDistribution:
    """Exact capped exit distribution of the +/-dx random walk absorbed at -A and +B steps."""
    n = A + B - 1                       # interior states -A+1 .. B-1
    v = np.zeros(n)
    v[A - 1] = 1.0                      # state 0
    win_p, ruin_p = np.empty(cap), np.empty(cap)
    for t in range(cap):
        win_p[t] = v[-1] * p
        ruin_p[t] = v[0] * (1 - p)
        nv = np.zeros(n)
        nv[1:] += v[:-1] * p
        nv[:-1] += v[1:] * (1 - p)
        v = nv
    days = np.arange(1, cap + 1)
    states = (np.arange(n) - (A - 1)) * dx
    day = np.concatenate([days, days, np.full(n, cap)])
    lr = np.concatenate([np.full(cap, B * dx), np.full(cap, -A * dx), states])
    oc = np.concatenate([np.full(cap, WIN), np.full(cap, RUIN), np.full(n, TIMEOUT)])
    pr = np.concatenate([win_p, ruin_p, v])
    keep = pr > 1e-14
    return ExitDistribution(day[keep].astype(np.int64), lr[keep], oc[keep].astype(np.int8),
                            pr[keep], cap)


def exit_distribution(model, cap: int = C.CAP_DAYS, n: int = C.GRID_N) -> ExitDistribution:
    """Capped exit distribution for any of the three models."""
    if model.name == "RandomWalk":
        return lattice_exit_distribution(model.p, model.A, model.B, model.dx, cap)
    return grid_first_passage(model.daily_cdf, cap=cap, n=n)


def price_martingale_p_win(a: float = C.A_LOG, b: float = C.B_LOG) -> float:
    """Zero *arithmetic* drift (mu = 0): S_t is a martingale, so p_win = (S0 - L)/(U - L).
    For +20%/-10% this is exactly 1/3, versus 36.6% for zero *log* drift (nu = 0)."""
    return (1 - math.exp(a)) / (math.exp(b) - math.exp(a))
