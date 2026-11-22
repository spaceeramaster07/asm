"""Checks of the Q5 engine on hand-built paths."""
import numpy as np
import pandas as pd
import pytest

from barrier import config as C
from barrier import scanner as S
from barrier import tournament as TN
from barrier.bootstrap import stationary_bootstrap
from barrier.engine import RUIN, WIN

BURN = C.ROLLING_DAYS


class ConstTable:
    """Kelly table stand-in: always recommends the same fraction."""

    def __init__(self, f):
        self.f = f

    def f_gbm(self, mu, sigma):
        return np.full(np.atleast_1d(mu).shape, self.f)

    def f_merton(self, mu, sigma, lam):
        return np.full(np.atleast_1d(mu).shape, self.f)


def path_with_trade(daily, n_days, rng_seed=0, length=None):
    """Noise for the burn-in, then a deterministic run of `daily` log returns."""
    rng = np.random.default_rng(rng_seed)
    L = length or BURN + 2 * C.CAP_DAYS + 10
    r = rng.normal(0, 0.01, L)
    r[BURN + 1:BURN + 1 + n_days] = daily
    r[BURN + 1 + n_days:] = 0.0
    return r


def test_fixed_fraction_win_arithmetic():
    # +20% reached on day 4 with four equal steps; fixed 50%: no leverage, cash earns r_f
    step = C.B_LOG / 4 + 1e-9
    r = path_with_trade(step, 4)[None, :]
    rf = 0.05
    res = TN.run_careers(r, [TN.Strategy("Fixed 50%", "fixed", 0.5)], ConstTable(0.0), rf, K=1)
    N, W = 50.0, 100.0
    P = np.exp(4 * step)
    eq = W - N * C.TRADE_COST + N * (P - 1) + (W - N) * rf * 4 / 252
    expected = (eq - N * P * C.TRADE_COST) * (1 + rf / 252)
    assert res.W[0, 0, 1] == pytest.approx(expected, rel=1e-12)


def test_margin_call_closes_levered_position():
    # 3x leverage; a one-day -12% gap: equity 100 - 3*100*0.12 = 64 < 25% of position 264 -> closed
    r = path_with_trade(np.log(0.88), 1)[None, :]
    res = TN.run_careers(r, [TN.Strategy("Fixed 3x", "fixed", 3.0)], ConstTable(0.0), 0.0, K=1)
    W, N, P = 100.0, 300.0, 0.88
    borrow = (W - N) * (0.0 + C.BORROW_SPREAD) * 1 / 252      # 200 borrowed for one day at r_f + spread
    eq = W - N * C.TRADE_COST + N * (P - 1) + borrow
    assert eq < C.MARGIN_MAINT * N * P
    assert res.W[0, 0, 1] == pytest.approx(eq - N * P * C.TRADE_COST, rel=1e-12)
    assert not res.bust[0, 0, 1]


def test_bankruptcy_is_absorbing():
    r = path_with_trade(np.log(0.60), 1)[None, :]      # -40% at 3x wipes the account out
    res = TN.run_careers(r, [TN.Strategy("Fixed 3x", "fixed", 3.0)], ConstTable(0.0), 0.0, K=3)
    assert res.bust[0, 0, 1] and res.W[0, 0, 1] == 0.0
    assert np.all(res.W[0, 0, 1:res.rounds[0] + 1] == 0.0)


def test_no_look_ahead_in_first_decision():
    rng = np.random.default_rng(1)
    r1 = rng.normal(0.0004, 0.01, BURN + 3000)
    r2 = r1.copy()
    r2[BURN + 1:] = rng.normal(-0.002, 0.03, len(r2) - BURN - 1)   # different future
    e = np.array([BURN, BURN])
    est = TN.window_estimates(np.vstack([r1, r2]), e)
    for k in est:
        assert est[k][0] == pytest.approx(est[k][1], rel=0, abs=0)


def test_first_exits_match_scanner():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0003, 0.012, 3000)
    logp = np.cumsum(r)[None, :]
    starts = np.arange(0, 3000 - C.CAP_DAYS - 1, 7)
    ex_idx, ret, oc = TN.first_exits(logp, starts)
    df = pd.DataFrame({"Close": np.exp(logp[0])}, index=pd.bdate_range("2000-01-03", periods=3000))
    sc = S.scan(df, mode="close")
    assert np.array_equal(ex_idx[0] - starts, sc.tau.to_numpy()[starts])
    assert np.array_equal(oc[0], sc.outcome.to_numpy()[starts])
    assert np.allclose(ret[0], sc.ret.to_numpy()[starts])


def test_empirical_kelly_matches_grid_search():
    rng = np.random.default_rng(3)
    R = np.where(rng.random((1, 400)) < 0.6, 0.2, -0.11)
    Tt = rng.integers(20, 300, (1, 400)).astype(float)
    ok = np.ones_like(R, bool)
    f = TN.empirical_kelly(R, Tt, ok, rf=0.03)[0]
    fs = np.linspace(0, 6, 6001)
    yrs = Tt[0] / 252

    def g(x):
        carry = (1 - x) * 0.03 * yrs if x <= 1 else -(x - 1) * (0.03 + C.BORROW_SPREAD) * yrs
        M = 1 + x * R[0] + carry - x * C.TRADE_COST - x * (1 + R[0]) * C.TRADE_COST
        return np.log(M).mean() if (M > 0).all() else -np.inf

    best = fs[np.argmax([g(x) for x in fs])]
    assert f == pytest.approx(best, abs=2e-3)


def test_bootstrap_preserves_moments_and_clustering():
    rng = np.random.default_rng(4)
    vol = np.exp(np.convolve(rng.normal(0, 0.3, 20000), np.ones(50) / 50, mode="same"))
    src = rng.normal(0, 0.01, 20000) * vol                       # clustered volatility
    bs = stationary_bootstrap(src, 4, 20000, 63, rng).ravel()
    assert bs.std() == pytest.approx(src.std(), rel=0.05)
    ac = lambda x: np.corrcoef(np.abs(x[:-1]), np.abs(x[1:]))[0, 1]
    assert ac(bs) == pytest.approx(ac(src), abs=0.03)


def test_martingale_doubles_after_ruin_and_resets_after_win():
    st = [TN.Strategy("Martingale", "martingale")]
    r = path_with_trade(np.log(0.89), 1)[None, :]
    res = TN.run_careers(r, st, ConstTable(0.0), 0.0, K=1)
    # first bet is MARTINGALE_START x W; loss -11% minus costs
    W, N, P = 100.0, C.MARTINGALE_START * 100.0, 0.89
    eq = W - N * C.TRADE_COST + N * (P - 1)
    assert res.W[0, 0, 1] == pytest.approx(eq - N * P * C.TRADE_COST, rel=1e-12)
    assert RUIN == 0 and WIN == 1
