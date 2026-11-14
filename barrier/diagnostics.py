"""Q2: out-of-sample tests of the models on daily log returns.

Each day's return r_t is scored with the latest monthly rolling fit made *before* day t.
  - predictive log-likelihood log f_model(r_t)                     (higher = less surprised)
  - PIT-normalised residual z_t = Phi^-1(F_model(r_t))             (i.i.d. N(0,1) if the model is right)
Tests on z: Jarque-Bera, tail exceedance, Ljung-Box on z^2, ARCH-LM, Berkowitz LR.
The random walk puts all its probability on two points (+/-dx), so it has no density and is not scored.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import config as C

SCORED_MODELS = ("GBM", "Merton")


def score_days(grid, returns) -> dict:
    """Per-day log-likelihood and PIT residual for each scored model.

    `returns`: Series of daily log returns covering the scored days. Days without an earlier fit are
    dropped. Returns {"dates", model: {"loglik", "z"}}."""
    pos = grid.index_for(returns.index, strictly_before=True)
    keep = pos >= 0
    r, pos = returns.to_numpy()[keep], pos[keep]
    out = {"dates": returns.index[keep]}
    for name in SCORED_MODELS:
        ll, u = np.empty(len(r)), np.empty(len(r))
        for k in np.unique(pos):
            m = grid.fits[k][name]
            sel = pos == k
            ll[sel] = m.daily_logpdf(r[sel])
            u[sel] = m.daily_cdf(r[sel])
        z = stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        out[name] = {"loglik": ll, "z": z}
    return out


def block_bootstrap_mean_ci(d: np.ndarray, block: int = C.BOOT_BLOCK_DAYS, reps: int = C.BOOT_REPS,
                            rng=None, level: float = 0.95) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a (serially dependent) daily series."""
    rng = rng or np.random.default_rng(0)
    n = len(d)
    nb = -(-n // block)
    idx = ((rng.integers(0, n, (reps, nb))[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]
    m = d[idx].mean(1)
    a = (1 - level) / 2
    return float(np.percentile(m, 100 * a)), float(np.percentile(m, 100 * (1 - a)))


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    x = x - x.mean()
    n = len(x)
    den = (x * x).sum()
    q = n * (n + 2) * sum(((x[:-k] * x[k:]).sum() / den) ** 2 / (n - k) for k in range(1, lags + 1))
    return float(q), float(stats.chi2.sf(q, lags))


def arch_lm(z: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Engle's test: regress z_t^2 on its own lags; n R^2 ~ chi2(lags) under no ARCH effects."""
    y = z ** 2
    X = np.column_stack([np.ones(len(y) - lags)] + [y[lags - k:len(y) - k] for k in range(1, lags + 1)])
    yy = y[lags:]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    r2 = 1 - resid.var() / yy.var()
    stat = len(yy) * r2
    return float(stat), float(stats.chi2.sf(stat, lags))


def berkowitz(z: np.ndarray) -> tuple[float, float]:
    """Berkowitz (2001) LR test of z ~ i.i.d. N(0,1) against an AR(1) with free mean and variance."""
    y, x = z[1:], z[:-1]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = (y - X @ beta).var()
    l1 = stats.norm.logpdf(y, X @ beta, np.sqrt(s2)).sum()
    l0 = stats.norm.logpdf(y).sum()
    lr = 2 * (l1 - l0)
    return float(lr), float(stats.chi2.sf(lr, 3))


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    den = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / den for k in range(1, lags + 1)])


def residual_tests(z: np.ndarray) -> dict:
    jb = stats.jarque_bera(z)
    a = np.abs(z)
    lb_q, lb_p = ljung_box(z ** 2, 10)
    lm, lm_p = arch_lm(z, 5)
    bk, bk_p = berkowitz(z)
    return {
        "n": int(len(z)), "mean": float(z.mean()), "sd": float(z.std(ddof=1)),
        "skew": float(stats.skew(z)), "exkurt": float(stats.kurtosis(z)),
        "jb_stat": float(jb.statistic), "jb_p": float(jb.pvalue),
        "share_gt3": float(np.mean(a > 3)), "share_gt4": float(np.mean(a > 4)),
        "normal_share_gt3": float(2 * stats.norm.sf(3)), "normal_share_gt4": float(2 * stats.norm.sf(4)),
        "ljung_box_z2_q": lb_q, "ljung_box_z2_p": lb_p,
        "arch_lm_stat": lm, "arch_lm_p": lm_p,
        "berkowitz_lr": bk, "berkowitz_p": bk_p,
        "acf_abs_z_lag1": float(acf(a, 1)[0]),
    }
