"""Fast checks of the closed forms and the Markov-chain grid solver."""
from dataclasses import replace

import numpy as np
import pytest

from barrier import config as C
from barrier import engine as E
from barrier import theory as T
from barrier.models import GBMModel, MertonModel, RandomWalkModel
from barrier.rng import rng_for

GBM = GBMModel(nu=0.08, sigma=0.18, n_obs=1260)
MER = MertonModel(nu_d=0.10, sigma_d=0.15, lam=6.0, mu_J=-0.012, sigma_J=0.02)


def test_rw_formula_matches_linear_system():
    for p, A, B in [(0.52, 7, 12), (0.48, 9, 15), (0.5, 8, 14)]:
        f = T.rw_first_passage(p, A, B)
        ls = T.rw_first_passage_linear_system(p, A, B)
        assert f[0] == pytest.approx(ls[0], abs=1e-9)
        assert f[1] == pytest.approx(ls[1], abs=1e-6)


def test_zero_price_drift_gives_one_third():
    s = 0.2
    assert T.gbm_first_passage(-0.5 * s ** 2, s)[0] == pytest.approx(1 / 3, abs=1e-12)


def test_grid_converges():
    p1, t1 = T.grid_first_passage(GBM.daily_cdf, n=C.GRID_N)
    p2, t2 = T.grid_first_passage(GBM.daily_cdf, n=2 * C.GRID_N)
    assert abs(p1 - p2) < 1e-4 and abs(t1 - t2) < 0.05


def test_grid_close_to_bgk_for_gbm():
    p, t = T.grid_first_passage(GBM.daily_cdf)
    th = GBM.theory()
    assert p == pytest.approx(th["p_win_daily_bgk"], abs=2e-3)
    assert t == pytest.approx(th["E_tau_days_daily_bgk"], abs=2.0)


def test_merton_without_jumps_is_gbm():
    m0 = replace(MER, lam=0.0, nu_d=GBM.nu, sigma_d=GBM.sigma)
    assert T.grid_first_passage(m0.daily_cdf) == pytest.approx(T.grid_first_passage(GBM.daily_cdf),
                                                               abs=1e-12)


def test_capped_distribution_is_normalised_and_matches_uncapped_limit():
    xd = T.grid_first_passage(MER.daily_cdf, cap=3000)
    assert xd.prob.sum() == pytest.approx(1.0, abs=1e-9)
    p_unc, _ = T.grid_first_passage(MER.daily_cdf)
    assert xd.p_win == pytest.approx(p_unc, abs=1e-4)      # timeouts vanish for a long cap
    assert xd.p_timeout < 1e-3


def test_capped_grid_matches_simulation():
    xd = T.grid_first_passage(MER.daily_cdf, cap=C.CAP_DAYS)
    ep = E.apply_cap(E.simulate(MER, 100_000, rng_for("test", "merton")), C.CAP_DAYS)
    s = E.summarize(ep)
    assert abs(s["p_win"] - xd.p_win) < 4 * s["p_win_se"]
    assert abs(s["tau_mean"] - xd.tau_mean) < 4 * s["tau_se"]
    assert s["ret_mean_ruin"] == pytest.approx(xd.ret_mean_ruin(), abs=1e-3)


def test_lattice_distribution_matches_formula():
    rw = RandomWalkModel(dx=0.01, p=0.51, A=11, B=18, nu=0.0, sigma=0.0)
    xd = T.lattice_exit_distribution(rw.p, rw.A, rw.B, rw.dx, cap=20_000)
    p, t = T.rw_first_passage(rw.p, rw.A, rw.B)
    assert xd.p_win == pytest.approx(p, abs=1e-9)
    assert xd.tau_mean == pytest.approx(t, rel=1e-6)


def test_merton_density_integrates_to_one():
    y = np.linspace(-0.4, 0.4, 200_001)
    dens = np.exp(MER.daily_logpdf(y))
    assert np.trapezoid(dens, y) == pytest.approx(1.0, abs=1e-6)
    assert MER.daily_cdf(0.4) - MER.daily_cdf(-0.4) == pytest.approx(1.0, abs=1e-9)


def test_kelly_does_not_jump_to_zero_at_the_wipeout_edge():
    # regression: when g(f) rises right up to the wipe-out edge, golden section used to step past it
    from barrier import kelly as K
    fs = []
    for mu in (0.2646, 0.2771, 0.2896):
        o = K.compress(T.exit_distribution(K._gbm_from(mu, 0.106), n=K.TABLE_GRID_N))
        fs.append(K.kelly_fraction(o, 0.025))
    assert min(fs) > 5.5 and np.all(np.diff(fs) > 0)


def test_rng_streams_are_order_sensitive():
    a = rng_for("x", "12").random()
    b = rng_for("x", "21").random()
    assert a != b
