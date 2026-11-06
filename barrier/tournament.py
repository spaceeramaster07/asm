"""Step 5 of the plan (Engine B): multi-round betting careers, marked to market every day.

Each round the trader puts a notional N_k into the index at the episode's entry price and
keeps the rest (W - N_k) in cash at r_f (negative cash = borrowing at r_f). The position is
held, not rebalanced, until the episode hits +20% / -10% or the time cap:

    W_t = W_{k-1} + N_k (P_t/P_0 - 1) + (W_{k-1} - N_k) r_f t/252

Wealth is tracked every trading day so drawdowns include intra-episode dips. If wealth
touches zero the trader is bankrupt and stays at zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .engine import RUIN, TIMEOUT, WIN, sim_barriers


# ------------------------------------------------------------------ Kelly
def growth_curve(R: np.ndarray, T: np.ndarray, rf: float, fs: np.ndarray) -> np.ndarray:
    """Expected log growth per episode g(f) = E[ln(1 + f R + (1 - f) r_f T/252)] over an
    empirical (R, T) sample. -inf where any episode would wipe the trader out."""
    cash = rf * T / C.TRADING_DAYS
    out = np.empty(len(fs))
    for i, f in enumerate(fs):
        x = 1 + f * R + (1 - f) * cash
        out[i] = np.log(x).mean() if (x > 0).all() else -np.inf
    return out


def kelly_fraction(R, T, rf, fmax: float = C.Q7_UNCAPPED_MAX, step: float = 0.005) -> float:
    fs = np.arange(0, fmax + step / 2, step)
    g = growth_curve(R, T, rf, fs)
    return float(fs[int(np.argmax(g))])


def two_point_growth(p: float, fs: np.ndarray, G: float = 0.20, L: float = 0.10) -> np.ndarray:
    """Textbook g(f) = p ln(1 + fG) + (1 - p) ln(1 - fL): exact +20/-10, no cash, no overshoot."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return p * np.log1p(fs * G) + (1 - p) * np.log1p(-fs * L)


# ------------------------------------------------------------------ strategies
@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str          # "fraction" | "flat" | "martingale"
    param: float = 0.0
    cap: float = C.LEVERAGE_CAP


def standard_strategies(f_kelly: float) -> list[Strategy]:
    return [
        Strategy("Flat Rs50", "flat", C.FLAT_BET),
        Strategy("Fixed 50%", "fraction", C.FIXED_FRACTION),
        Strategy("Full Kelly", "fraction", min(f_kelly, C.LEVERAGE_CAP)),
        Strategy("Half Kelly", "fraction", min(0.5 * f_kelly, C.LEVERAGE_CAP)),
        Strategy("Martingale", "martingale", C.MARTINGALE_START),
        # In a continuous price path, re-entering 100% at each exit price == holding.
        Strategy("Buy & Hold", "fraction", 1.0),
    ]


class Book:
    """Wealth, peak and drawdown state for several strategies x many careers."""

    def __init__(self, strategies: list[Strategy], n: int, K: int):
        self.S = strategies
        self.n, self.K = n, K
        s = len(strategies)
        self.W = np.full((s, n), C.W0)
        self.peak = np.full((s, n), C.W0)
        self.mdd = np.zeros((s, n))
        self.bankrupt = np.zeros((s, n), bool)
        self.mart_f = np.full((s, n), C.MARTINGALE_START)
        self.hist = np.full((s, n, K + 1), np.nan)
        self.hist[:, :, 0] = C.W0
        self.days = np.zeros(n)
        self.k = 0

    def notional(self, i: int) -> np.ndarray:
        st, W = self.S[i], self.W[i]
        if st.kind == "fraction":
            N = st.param * W
        elif st.kind == "flat":
            N = np.minimum(st.param, st.cap * W)
        else:
            N = self.mart_f[i] * W
        return np.where(self.bankrupt[i], 0.0, N)

    def play(self, P: np.ndarray, tau: np.ndarray, outcome: np.ndarray, rf: float,
             rows: np.ndarray | None = None) -> None:
        """One round. P[r, t-1] = price ratio on day t of the episode (t = 1..L); columns
        beyond each row's tau are ignored. `rows` selects which careers play (default all)."""
        rows = np.arange(self.n) if rows is None else rows
        L = P.shape[1]
        t = np.arange(1, L + 1)
        valid = t[None, :] <= tau[:, None]
        last = tau - 1
        r_idx = np.arange(len(rows))
        for i, st in enumerate(self.S):
            W = self.W[i, rows]
            N = self.notional(i)[rows]
            path = W[:, None] + N[:, None] * (P - 1) + (W - N)[:, None] * rf * t / C.TRADING_DAYS
            path = np.where(valid, path, np.nan)
            broke = (np.nanmin(path, axis=1) <= 0) | self.bankrupt[i, rows]
            run_peak = np.fmax.accumulate(np.where(valid, path, -np.inf), axis=1)
            run_peak = np.maximum(run_peak, self.peak[i, rows][:, None])
            dd = np.nanmax(np.where(valid, 1 - path / run_peak, np.nan), axis=1)
            W_end = np.where(broke, 0.0, path[r_idx, last])
            self.mdd[i, rows] = np.where(broke, 1.0, np.maximum(self.mdd[i, rows], dd))
            self.peak[i, rows] = np.maximum(self.peak[i, rows], np.nanmax(path, axis=1))
            self.W[i, rows] = W_end
            self.bankrupt[i, rows] = broke
            if st.kind == "martingale":
                f = self.mart_f[i, rows]
                f = np.where(outcome == RUIN, np.minimum(2 * f, st.cap), f)
                f = np.where(outcome == WIN, C.MARTINGALE_START, f)   # TIMEOUT: unchanged
                self.mart_f[i, rows] = f
            self.hist[i, rows, self.k + 1] = W_end
        self.days[rows] += tau

    def next_round(self) -> None:
        self.k += 1

    def cash_day(self, rf: float, rows: np.ndarray | None = None) -> None:
        """A day spent fully in cash (the gap between a real exit and the next entry)."""
        rows = np.arange(self.n) if rows is None else rows
        self.W[:, rows] *= 1 + rf / C.TRADING_DAYS
        self.days[rows] += 1


def simulate_careers(model, strategies, n: int, K: int, rf: float, rng,
                     cap: int = C.CAP_DAYS) -> Book:
    """n independent careers of K rounds in a model world (same paths for every strategy)."""
    a, b = sim_barriers(model)
    book = Book(strategies, n, K)
    for _ in range(K):
        X = np.cumsum(model.increments(rng, (n, cap)), axis=1)
        hit = (X >= b) | (X <= a)
        any_hit = hit.any(axis=1)
        j = np.where(any_hit, hit.argmax(axis=1), cap - 1)
        tau = j + 1
        xe = X[np.arange(n), j]
        outcome = np.where(any_hit, np.where(xe >= b, WIN, RUIN), TIMEOUT)
        book.play(np.exp(X), tau, outcome, rf)
        book.next_round()
    return book


def real_career(close: np.ndarray, chain, strategies, rf: float) -> Book:
    """One real career along a non-overlapping chain of trades (Close-checked exits)."""
    K = len(chain)
    book = Book(strategies, 1, K)
    prev_exit = None
    for tr in chain.itertuples():
        if prev_exit is not None:
            for _ in range(tr.entry_idx - prev_exit):
                book.cash_day(rf)
        P = close[tr.entry_idx + 1: tr.exit_idx + 1][None, :] / close[tr.entry_idx]
        book.play(P, np.array([tr.tau]), np.array([tr.outcome]), rf)
        book.next_round()
        prev_exit = tr.exit_idx
    return book


def buy_and_hold_real(close: np.ndarray, start: int, end: int) -> dict:
    """True buy-and-hold over the same calendar span as a real chain (no exits, no gaps)."""
    w = C.W0 * close[start: end + 1] / close[start]
    peak = np.maximum.accumulate(w)
    return {"W": float(w[-1]), "mdd": float(np.max(1 - w / peak)), "days": int(end - start)}


def metrics(book: Book) -> dict:
    """Leaderboard metrics per strategy."""
    out = {}
    yrs = np.maximum(book.days, 1) / C.TRADING_DAYS
    for i, st in enumerate(book.S):
        W = book.W[i]
        cagr = np.where(W > 0, (np.maximum(W, 1e-12) / C.W0) ** (1 / yrs) - 1, -1.0)
        out[st.name] = {
            "f": st.param if st.kind != "flat" else None,
            "W_median": float(np.median(W)), "W_mean": float(np.mean(W)),
            "W_p05": float(np.percentile(W, 5)), "W_p95": float(np.percentile(W, 95)),
            "cagr_median": float(np.median(cagr)),
            "mdd_median": float(np.median(book.mdd[i])),
            "p_mdd_gt_50": float(np.mean(book.mdd[i] > 0.5)),
            "p_lose_half": float(np.mean(W < 0.5 * C.W0)),
            "p_bankrupt": float(np.mean(book.bankrupt[i])),
            "years_median": float(np.median(yrs)),
        }
    return out
