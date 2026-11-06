"""Step 2: exact closed-form first-passage results.

All functions work in log space X_t = ln(S_t/S0), lower barrier a < 0 < b upper barrier.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C


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


def price_martingale_p_win(a: float = C.A_LOG, b: float = C.B_LOG) -> float:
    """Zero *arithmetic* drift (mu = 0): S_t is a martingale, so p_win = (S0 - L)/(U - L).
    For +20%/-10% this is exactly 1/3, versus 36.6% for zero *log* drift (nu = 0)."""
    return (1 - math.exp(a)) / (math.exp(b) - math.exp(a))
