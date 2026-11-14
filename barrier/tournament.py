"""Q5 engine: betting careers on continuous daily price paths, with a trader who only sees the past.

A career is one daily log-return path r[0..L-1] (log price p[t] = r[0] + ... + r[t]). The first `burn`
days are history the trader can see; then trades run back to back: enter at the close of day e, exit
at the first close at +20% / -10% (or after 504 days), re-enter at the close of the next day. Before
each entry the trader re-estimates from r[e-W+1 .. e] only, and every strategy picks its bet fraction.

Each round, a strategy with wealth W puts notional N in the index and (W - N) in cash:
    equity_t = W - N c + N (P_t - 1) + (W - N) rate t/252,   rate = r_f (cash) or r_f + spread (borrowed)
    exit:    equity_tau - N P_tau c
Levered positions are force-closed at the close when equity < MARGIN_MAINT x position value (the rest
of the round is then in cash); equity <= 0 is bankruptcy (absorbing). Wealth is marked to market daily,
so drawdowns include intra-trade dips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .engine import RUIN, TIMEOUT, WIN


# ------------------------------------------------------------------ strategies
@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str          # gbm_kelly | merton_kelly | emp_kelly | shrink_kelly | vol_target | dd_control
    mult: float = 1.0  # | flat | fixed | martingale | anti_martingale


REFERENCE = "GBM Half Kelly"


def standard_strategies() -> list[Strategy]:
    names = {0.5: "GBM Half Kelly", 1.0: "GBM Kelly"}
    s = [Strategy(names.get(m, f"GBM Kelly {m:.2f}x"), "gbm_kelly", m) for m in C.KELLY_SPECTRUM]
    s += [
        Strategy("Merton Kelly", "merton_kelly", 1.0),
        Strategy("Merton Half Kelly", "merton_kelly", 0.5),
        Strategy("Empirical Kelly", "emp_kelly", 1.0),
        Strategy("Empirical Half Kelly", "emp_kelly", 0.5),
        Strategy("Shrinkage Kelly", "shrink_kelly", 1.0),
        Strategy("Vol-targeted Half Kelly", "vol_target", 0.5),
        Strategy("Drawdown-controlled Half Kelly", "dd_control", 0.5),
        Strategy("Flat 50", "flat"),
        Strategy("Fixed 50%", "fixed", C.FIXED_FRACTION),
        Strategy("Martingale", "martingale"),
        Strategy("Anti-martingale", "anti_martingale"),
    ]
    return s


# ------------------------------------------------------------------ trader's estimates
def window_estimates(r: np.ndarray, e: np.ndarray, window: int = C.ROLLING_DAYS) -> dict:
    """Rolling estimates for each row at its entry day e (uses r[e-window+1 .. e] only)."""
    rows = np.arange(len(e))[:, None]
    x = r[rows, e[:, None] + np.arange(-window + 1, 1)[None, :]]
    m = x.mean(1)
    v = x.var(1, ddof=1)
    nu = m / C.DT
    sigma = np.sqrt(v / C.DT)
    mu = nu + 0.5 * sigma ** 2
    se = sigma / np.sqrt(window * C.DT)
    short = x[:, -C.VOL_TARGET_DAYS:]
    sigma_short = np.sqrt(short.var(1, ddof=1) / C.DT)
    # Merton jump filter (iterated k-sigma), vectorised across rows
    mask = np.ones_like(x, bool)
    for _ in range(100):
        cnt = mask.sum(1)
        mm = (x * mask).sum(1) / cnt
        ss = np.sqrt((((x - mm[:, None]) ** 2) * mask).sum(1) / (cnt - 1))
        new = np.abs(x - mm[:, None]) <= C.JUMP_K * ss[:, None]
        if np.array_equal(new, mask):
            break
        mask = new
    lam = (~mask).sum(1) / (window * C.DT)
    return {"mu": mu, "sigma": sigma, "se": se, "sigma_short": sigma_short, "lam": lam}


def first_exits(logp: np.ndarray, starts: np.ndarray, cap: int = C.CAP_DAYS, chunk: int = 64,
                step: int = 63) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barrier outcome of a trade entered at each start day (every row shares the start days).

    Returns (exit_idx, simple return, outcome), each (rows, len(starts)). Starts need s + cap < L."""
    B = logp.shape[0]
    S = len(starts)
    exit_idx = np.empty((B, S), np.int64)
    ret = np.empty((B, S))
    oc = np.empty((B, S), np.int8)
    a, b = C.A_LOG, C.B_LOG
    for j0 in range(0, S, chunk):
        st = starts[j0:j0 + chunk]
        base = logp[:, st]
        open_ = np.ones((B, len(st)), bool)
        tau = np.full((B, len(st)), cap)
        xe = logp[:, st + cap] - base
        o = np.full((B, len(st)), TIMEOUT, np.int8)
        for d0 in range(0, cap, step):
            d = np.arange(d0 + 1, min(d0 + step, cap) + 1)
            X = logp[:, st[:, None] + d[None, :]] - base[:, :, None]
            hit = (X >= b) | (X <= a)
            new = hit.any(2) & open_
            if new.any():
                f = hit.argmax(2)
                xv = np.take_along_axis(X, f[:, :, None], 2)[:, :, 0]
                tau = np.where(new, d0 + f + 1, tau)
                xe = np.where(new, xv, xe)
                o = np.where(new, np.where(xv >= b, WIN, RUIN), o).astype(np.int8)
                open_ &= ~new
            if not open_.any():
                break
        exit_idx[:, j0:j0 + len(st)] = st[None, :] + tau
        ret[:, j0:j0 + len(st)] = np.expm1(xe)
        oc[:, j0:j0 + len(st)] = o
    return exit_idx, ret, oc


def empirical_kelly(R: np.ndarray, Tt: np.ndarray, ok: np.ndarray, rf: float,
                    fmax: float = C.F_GRID_MAX, iters: int = 40) -> np.ndarray:
    """Row-wise argmax_f mean_ok ln M(f) by bisection on the (decreasing) derivative.
    R, Tt, ok: (rows, samples). Rows with fewer than 20 samples bet 0."""
    from .kelly import feasible_fmax
    yrs = Tt / C.TRADING_DAYS
    cost = C.TRADE_COST
    edge = feasible_fmax(np.where(ok, R, 0.0), np.where(ok, Tt, 0.0), rf, axis=1)
    hi = np.minimum(fmax, edge * (1 - 1e-9))
    lo = np.zeros(len(R))

    def deriv(f):
        f = f[:, None]
        lev = f > 1
        carry = np.where(lev, -(f - 1) * (rf + C.BORROW_SPREAD) * yrs, (1 - f) * rf * yrs)
        dcarry = np.where(lev, -(rf + C.BORROW_SPREAD) * yrs, -rf * yrs)
        M = 1 + f * R + carry - f * cost - f * (1 + R) * cost
        dM = R + dcarry - cost - (1 + R) * cost
        return np.where(ok, dM / np.maximum(M, 1e-12), 0.0).sum(1)

    at0 = deriv(lo) > 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = deriv(mid) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    f = np.where(at0, 0.5 * (lo + hi), 0.0)
    return np.where(ok.sum(1) >= 20, f, 0.0)


# ------------------------------------------------------------------ careers
@dataclass
class CareerResult:
    names: list
    W: np.ndarray          # (S, B, K+1) wealth after each round (NaN once a career has stopped)
    mdd: np.ndarray        # (S, B, K+1) max drawdown so far
    bust: np.ndarray       # (S, B, K+1)
    days: np.ndarray       # (B, K+1) trading days elapsed
    bh_W: np.ndarray       # (B, K+1) buy & hold wealth on the same path
    bh_mdd: np.ndarray     # (B, K+1)
    f_gbm_raw: np.ndarray  # (B, K) GBM Kelly fraction before multiplier and cap
    rounds: np.ndarray     # (B,) rounds completed


def run_careers(r: np.ndarray, strategies: list[Strategy], table, rf: float, K: int,
                burn: int = C.ROLLING_DAYS, entry0: np.ndarray | None = None,
                cap: int = C.CAP_DAYS, emp_stride: int = 5) -> CareerResult:
    """Play K rounds on every row of `r` (rows = careers). `entry0` optionally gives each row's first
    entry day (default `burn`). A row stops early if its next trade would run past the data."""
    B, L = r.shape
    logp = np.cumsum(r, axis=1)
    S = len(strategies)
    e = np.full(B, burn, np.int64) if entry0 is None else entry0.astype(np.int64).copy()
    e_first = e.copy()
    active = np.ones(B, bool)

    # empirical Kelly inputs: outcomes of every emp_stride-th start day (filtered per round by date)
    starts = np.arange(max(0, int(e.min()) - burn), L - cap - 1, emp_stride)
    ex_idx, ex_R, _ = first_exits(logp, starts, cap)
    ex_T = (ex_idx - starts[None, :]).astype(float)

    W = np.full((S, B), C.W0)
    peak = W.copy()
    mdd = np.zeros((S, B))
    bust = np.zeros((S, B), bool)
    mart = np.full((S, B), C.MARTINGALE_START)
    hW = np.full((S, B, K + 1), np.nan)
    hM = np.full((S, B, K + 1), np.nan)
    hB = np.zeros((S, B, K + 1), bool)
    hW[:, :, 0], hM[:, :, 0] = C.W0, 0.0
    days = np.zeros((B, K + 1))
    bhW = np.full((B, K + 1), np.nan)
    bhM = np.full((B, K + 1), np.nan)
    bhW[:, 0], bhM[:, 0] = C.W0, 0.0
    f_gbm_raw = np.full((B, K), np.nan)
    rounds = np.zeros(B, np.int64)
    # buy & hold drawdown along each path from its first entry
    after = np.arange(L)[None, :] >= e_first[:, None]
    run_max = np.maximum.accumulate(np.where(after, logp, -np.inf), axis=1)
    with np.errstate(invalid="ignore"):
        bh_mdd_path = np.where(after, 1 - np.exp(logp - run_max), 0.0)
    bh_mdd_path = np.maximum.accumulate(bh_mdd_path, axis=1)
    del run_max, after

    t_ax = np.arange(1, cap + 1)
    names = [s.name for s in strategies]
    for k in range(K):
        rows = np.nonzero(active)[0]
        # trade window for each active row; censored if the data end first without an exit
        ee = e[rows]
        idx = np.minimum(ee[:, None] + t_ax[None, :], L - 1)
        avail = ee[:, None] + t_ax[None, :] <= L - 1
        X = logp[rows[:, None], idx] - logp[rows, ee][:, None]
        hit = ((X >= C.B_LOG) | (X <= C.A_LOG)) & avail
        any_hit = hit.any(1)
        full_window = avail[:, -1]
        ok = any_hit | full_window
        active[rows[~ok]] = False
        rows, ee, X, hit, any_hit = rows[ok], ee[ok], X[ok], hit[ok], any_hit[ok]
        if rows.size == 0:
            break
        j = np.where(any_hit, hit.argmax(1), cap - 1)
        tau = j + 1
        xe = X[np.arange(len(rows)), j]
        outcome = np.where(any_hit, np.where(xe >= C.B_LOG, WIN, RUIN), TIMEOUT)
        P = np.exp(X)
        valid = t_ax[None, :] <= tau[:, None]

        est = window_estimates(r[rows], ee)
        f_g = table.f_gbm(est["mu"], est["sigma"])
        f_gbm_raw[rows, k] = f_g
        need_emp = any(s.kind == "emp_kelly" for s in strategies)
        if need_emp:
            # each row's own window of start days: starts in [e - burn, e), resolved by day e
            width = -(-burn // emp_stride)
            j0 = np.searchsorted(starts, ee - burn)
            cols = np.minimum(j0[:, None] + np.arange(width)[None, :], len(starts) - 1)
            rr = rows[:, None]
            s_win = starts[cols]
            sel = (s_win >= (ee - burn)[:, None]) & (s_win < ee[:, None]) & (ex_idx[rr, cols] <= ee[:, None])
            f_emp = empirical_kelly(ex_R[rr, cols], ex_T[rr, cols], sel, rf)
        f_m = table.f_merton(est["mu"], est["sigma"], est["lam"]) if any(
            s.kind == "merton_kelly" for s in strategies) else None
        mu_shr = rf + (est["mu"] - rf) * np.maximum(0.0, 1 - est["se"] ** 2 /
                                                     np.maximum((est["mu"] - rf) ** 2, 1e-12))
        f_shr = table.f_gbm(mu_shr, est["sigma"])

        for i, st in enumerate(strategies):
            Wi = W[i, rows]
            if st.kind == "gbm_kelly":
                f = st.mult * f_g
            elif st.kind == "merton_kelly":
                f = st.mult * f_m
            elif st.kind == "emp_kelly":
                f = st.mult * f_emp
            elif st.kind == "shrink_kelly":
                f = st.mult * f_shr
            elif st.kind == "vol_target":
                f = st.mult * f_g * est["sigma"] / np.maximum(est["sigma_short"], 1e-6)
            elif st.kind == "dd_control":
                dd_now = 1 - Wi / np.maximum(peak[i, rows], 1e-12)
                f = st.mult * f_g * np.where(dd_now > C.DD_TRIGGER, 0.5, 1.0)
            elif st.kind == "fixed":
                f = np.full(len(rows), st.mult)
            elif st.kind in ("martingale", "anti_martingale"):
                f = mart[i, rows]
            elif st.kind == "flat":
                f = np.minimum(C.FLAT_BET / np.maximum(Wi, 1e-12), C.LEVERAGE_CAP)
            else:
                raise ValueError(st.kind)
            f = np.clip(np.nan_to_num(f), 0.0, C.LEVERAGE_CAP)
            N = np.where(bust[i, rows], 0.0, f * Wi)
            rate = np.where(Wi - N >= 0, rf, rf + C.BORROW_SPREAD)
            Eq = (Wi - N * C.TRADE_COST)[:, None] + N[:, None] * (P - 1) \
                + ((Wi - N) * rate)[:, None] * t_ax[None, :] / C.TRADING_DAYS
            lev = (N > Wi)[:, None]
            mc = lev & valid & (Eq < C.MARGIN_MAINT * N[:, None] * P)
            br = valid & (Eq <= 0)
            stop = mc | br
            has_stop = stop.any(1)
            d = np.where(has_stop, stop.argmax(1), tau - 1)          # 0-based closing day
            ridx = np.arange(len(rows))
            Ed = Eq[ridx, d]
            broke = has_stop & br[ridx, d]
            w_close = Ed - N * P[ridx, d] * C.TRADE_COST
            w_end = np.where(has_stop & ~broke,
                             w_close * (1 + rf * (tau - 1 - d) / C.TRADING_DAYS), w_close)
            w_end = np.where(broke | bust[i, rows], 0.0, np.maximum(w_end, 0.0))
            upto = t_ax[None, :] <= (d + 1)[:, None]
            path = np.where(upto, Eq, -np.inf)
            rp = np.maximum(np.maximum.accumulate(path, axis=1), peak[i, rows][:, None])
            dd = np.max(np.where(upto, 1 - np.maximum(Eq, 0) / rp, 0.0), axis=1)
            newly_bust = broke & ~bust[i, rows]
            mdd[i, rows] = np.where(newly_bust | bust[i, rows], 1.0, np.maximum(mdd[i, rows], dd))
            peak[i, rows] = np.maximum(peak[i, rows], np.maximum(path.max(1), w_end))
            bust[i, rows] |= broke
            # one cash day before the next entry
            W[i, rows] = w_end * (1 + rf / C.TRADING_DAYS)
            if st.kind == "martingale":
                m = mart[i, rows]
                m = np.where(outcome == RUIN, np.minimum(2 * m, C.LEVERAGE_CAP), m)
                mart[i, rows] = np.where(outcome == WIN, C.MARTINGALE_START, m)
            elif st.kind == "anti_martingale":
                m = mart[i, rows]
                m = np.where(outcome == WIN, np.minimum(2 * m, C.LEVERAGE_CAP), m)
                mart[i, rows] = np.where(outcome == RUIN, C.MARTINGALE_START, m)
            hW[i, rows, k + 1] = W[i, rows]
            hM[i, rows, k + 1] = mdd[i, rows]
            hB[i, rows, k + 1] = bust[i, rows]

        exit_day = ee + tau
        days[rows, k + 1] = days[rows, k] + tau + 1
        bhW[rows, k + 1] = C.W0 * (1 - C.TRADE_COST) ** 2 * np.exp(logp[rows, exit_day] - logp[rows, e_first[rows]])
        bhM[rows, k + 1] = bh_mdd_path[rows, exit_day]
        rounds[rows] += 1
        e[rows] = exit_day + 1
        active[rows] &= e[rows] < L - 1
    return CareerResult(names, hW, hM, hB, days, bhW, bhM, f_gbm_raw, rounds)


# ------------------------------------------------------------------ metrics
def metrics_at(res: CareerResult, k: int, ref: str = REFERENCE) -> dict:
    """Leaderboard after round k (careers that reached round k)."""
    reached = res.rounds >= k
    yrs = np.maximum(res.days[reached, k], 1) / C.TRADING_DAYS
    out = {}
    ref_i = res.names.index(ref)

    def logg(Wk):
        return np.log(np.maximum(Wk, 1e-6 * C.W0) / C.W0) / yrs

    g_ref = logg(res.W[ref_i, reached, k])
    series = [(n, res.W[i, reached, :k + 1], res.mdd[i, reached, k], res.bust[i, reached, k])
              for i, n in enumerate(res.names)]
    series.append(("Buy & Hold", res.bh_W[reached, :k + 1], res.bh_mdd[reached, k],
                   np.zeros(reached.sum(), bool)))
    for name, Wh, md, bu in series:
        Wk = Wh[:, -1]
        cagr = np.where(Wk > 0, (np.maximum(Wk, 1e-12) / C.W0) ** (1 / yrs) - 1, -1.0)
        lg = logg(Wk)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.diff(np.log(np.maximum(Wh, 1e-6 * C.W0)), axis=1)
            per_yr = k / yrs
            sharpe = lr.mean(1) / lr.std(1, ddof=1) * np.sqrt(per_yr)
        d = lg - g_ref
        n = len(d)
        out[name] = {
            "W_median": float(np.median(Wk)), "W_p05": float(np.percentile(Wk, 5)),
            "W_p95": float(np.percentile(Wk, 95)),
            "cagr_median": float(np.median(cagr)), "log_growth_mean": float(lg.mean()),
            "mdd_median": float(np.median(md)), "p_mdd_gt_50": float(np.mean(md > 0.5)),
            "p_bust": float(np.mean(bu)), "p_lose_half": float(np.mean(Wk < 0.5 * C.W0)),
            "sharpe_median": float(np.nanmedian(sharpe)),
            "calmar_median": float(np.median(np.where(md > 0, cagr / np.maximum(md, 1e-9), np.nan))),
            "vs_ref_log_growth": float(d.mean()),
            "vs_ref_ci": [float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(n)),
                          float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(n))] if n > 1 else [np.nan, np.nan],
            "n": int(n), "years_median": float(np.median(yrs)),
        }
    return out
