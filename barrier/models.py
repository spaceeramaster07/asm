"""Step 1b: the three stochastic models, calibrated to daily log returns.

Design rule: all three models are calibrated to reproduce the *same* daily mean and variance
of log returns. Any difference between them in later steps then comes from the shape of the
return distribution (lattice vs Gaussian vs jumps), not from mismatched first two moments.

Each model exposes:
    calibrate(r)              -> model, from a Series of daily log returns
    increments(rng, shape)    -> array of simulated daily log-return increments
    moments()                 -> theoretical daily mean / var / skew / excess kurtosis
    theory()                  -> closed-form barrier results where they exist
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import theory as T


def sample_moments(r: pd.Series | np.ndarray) -> dict:
    r = np.asarray(r)
    return {
        "mean": float(r.mean()),
        "var": float(r.var(ddof=1)),
        "skew": float(stats.skew(r, bias=False)),
        "exkurt": float(stats.kurtosis(r, fisher=True, bias=False)),
        "n": int(len(r)),
    }


def _annualise(mean_d: float, var_d: float) -> tuple[float, float]:
    """Daily log mean/var -> (nu, sigma) annualised."""
    return mean_d / C.DT, math.sqrt(var_d / C.DT)


# =============================================================== Model B: GBM
@dataclass
class GBMModel:
    nu: float        # annual log drift  (mu - sigma^2/2)
    sigma: float     # annual volatility
    n_obs: int

    name = "GBM"

    @property
    def mu(self) -> float:
        return self.nu + 0.5 * self.sigma ** 2

    @property
    def nu_se(self) -> float:
        """Standard error of the annual drift estimate: sigma / sqrt(T_years)."""
        return self.sigma / math.sqrt(self.n_obs * C.DT)

    @classmethod
    def calibrate(cls, r: pd.Series) -> "GBMModel":
        m = sample_moments(r)
        nu, sigma = _annualise(m["mean"], m["var"])
        return cls(nu=nu, sigma=sigma, n_obs=m["n"])

    def increments(self, rng: np.random.Generator, shape) -> np.ndarray:
        return self.nu * C.DT + self.sigma * math.sqrt(C.DT) * rng.standard_normal(shape)

    def moments(self) -> dict:
        return {"mean": self.nu * C.DT, "var": self.sigma ** 2 * C.DT, "skew": 0.0, "exkurt": 0.0}

    def theory(self) -> dict:
        pc, tc = T.gbm_first_passage(self.nu, self.sigma)
        pd_, td = T.gbm_first_passage_daily(self.nu, self.sigma)
        ov = T.expected_daily_overshoot(self.sigma)
        return {
            "p_win_continuous": pc,
            "E_tau_days_continuous": tc * C.TRADING_DAYS,
            "p_win_daily_bgk": pd_,
            "E_tau_days_daily_bgk": td * C.TRADING_DAYS,
            "expected_stop_return_daily": math.exp(C.A_LOG - ov) - 1,
        }


# =============================================================== Model A: biased random walk
@dataclass
class RandomWalkModel:
    dx: float        # lattice step (log units)
    p: float         # P(up step)
    A: int           # ruin barrier in steps below 0
    B: int           # win barrier in steps above 0
    nu: float
    sigma: float

    name = "RandomWalk"

    @classmethod
    def calibrate(cls, r: pd.Series) -> "RandomWalkModel":
        """Exact mean AND variance match of the daily step:
            mean = (2p-1) dx = nu dt,   var = dx^2 - (nu dt)^2 = sigma^2 dt
        =>  dx = sqrt(sigma^2 dt + (nu dt)^2),   p = (1 + nu dt / dx) / 2.
        (The draft's dx = sigma sqrt(dt) matches the mean but under-shoots the variance by
        (nu dt)^2 - negligible here, but this version is exact at no cost.)"""
        m = sample_moments(r)
        nu, sigma = _annualise(m["mean"], m["var"])
        dx = math.sqrt(m["var"] + m["mean"] ** 2)
        p = 0.5 * (1 + m["mean"] / dx)
        B = int(round(C.B_LOG / dx))
        A = int(round(-C.A_LOG / dx))
        return cls(dx=dx, p=p, A=A, B=B, nu=nu, sigma=sigma)

    @property
    def effective_up_return(self) -> float:
        return math.exp(self.B * self.dx) - 1

    @property
    def effective_down_return(self) -> float:
        return math.exp(-self.A * self.dx) - 1

    def increments(self, rng: np.random.Generator, shape) -> np.ndarray:
        up = rng.random(shape) < self.p
        return np.where(up, self.dx, -self.dx)

    def moments(self) -> dict:
        m = (2 * self.p - 1) * self.dx
        v = self.dx ** 2 - m ** 2
        # skew of a two-point distribution; excess kurtosis = 1/(pq) - 6
        q = 1 - self.p
        skew = (q - self.p) / math.sqrt(self.p * q)
        return {"mean": m, "var": v, "skew": skew, "exkurt": 1 / (self.p * q) - 6}

    def theory(self) -> dict:
        pw, et = T.rw_first_passage(self.p, self.A, self.B)
        return {
            "p_win_exact_lattice": pw,
            "E_tau_days_exact_lattice": et,
            # lattice barriers are hit exactly: no overshoot, stop return is deterministic
            "expected_stop_return_daily": self.effective_down_return,
        }


# =============================================================== Model C: Merton jump-diffusion
@dataclass
class MertonModel:
    nu_d: float          # annual log drift of the diffusion part
    sigma_d: float       # annual diffusion volatility (moment-matched)
    lam: float           # jumps per year
    mu_J: float          # mean log jump
    sigma_J: float       # std of log jump
    # diagnostics from the filter (not used for simulation)
    sigma_d_filter: float = 0.0
    n_jumps: int = 0
    jump_dates: list = field(default_factory=list)
    jump_sizes: list = field(default_factory=list)
    filter_iterations: int = 0

    name = "Merton"

    @property
    def k(self) -> float:
        return math.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1

    @property
    def mu(self) -> float:
        """Arithmetic drift mu in dX = (mu - sigma^2/2 - lam k) dt + sigma dW + J dN."""
        return self.nu_d + 0.5 * self.sigma_d ** 2 + self.lam * self.k

    @classmethod
    def calibrate(cls, r: pd.Series, k_sigma: float = C.JUMP_K) -> "MertonModel":
        """Iterated k-sigma filter, then moment-match the diffusion part.

        1. Flag |r - m| > k*s using m, s from the *currently normal* days; repeat until the
           flagged set stops changing (a single pass is circular: s includes the jumps).
        2. lam = #jump days / years.  Jump-day return = diffusion + jump, so
           mu_J = mean(r_jump) - m,  sigma_J^2 = var(r_jump) - s^2.
        3. Choose the diffusion (nu_d, sigma_d) so the model's daily mean and variance equal the
           sample's exactly:
              mean: nu_d dt + lam dt mu_J                      = sample mean
              var : sigma_d^2 dt + lam dt (mu_J^2 + sigma_J^2) = sample var
        """
        x = r.to_numpy()
        mask = np.ones(len(x), dtype=bool)
        it = 0
        for it in range(1, 101):
            m, s = x[mask].mean(), x[mask].std(ddof=1)
            new = np.abs(x - m) <= k_sigma * s
            if np.array_equal(new, mask):
                break
            mask = new
        m, s = x[mask].mean(), x[mask].std(ddof=1)
        jumps = ~mask
        n_j = int(jumps.sum())
        years = len(x) * C.DT
        lam = n_j / years
        J = x[jumps] - m
        mu_J = float(J.mean())
        sigma_J = math.sqrt(max(J.var(ddof=1) - s ** 2, 1e-8)) if n_j > 1 else 0.0

        sm = sample_moments(x)
        lam_dt = lam * C.DT
        var_d = sm["var"] - lam_dt * (mu_J ** 2 + sigma_J ** 2)
        if var_d <= 0:
            raise ValueError("jump component explains more than all variance; raise k_sigma")
        nu_d = (sm["mean"] - lam_dt * mu_J) / C.DT
        sigma_d = math.sqrt(var_d / C.DT)

        return cls(nu_d=nu_d, sigma_d=sigma_d, lam=lam, mu_J=mu_J, sigma_J=sigma_J,
                   sigma_d_filter=s / math.sqrt(C.DT), n_jumps=n_j,
                   jump_dates=[str(d.date()) for d in r.index[jumps]],
                   jump_sizes=[float(v) for v in x[jumps]],
                   filter_iterations=it)

    def increments(self, rng: np.random.Generator, shape) -> np.ndarray:
        diff = self.nu_d * C.DT + self.sigma_d * math.sqrt(C.DT) * rng.standard_normal(shape)
        N = rng.poisson(self.lam * C.DT, shape)
        # sum of N iid N(mu_J, sigma_J^2) = N*mu_J + sqrt(N)*sigma_J*Z
        jump = N * self.mu_J + np.sqrt(N) * self.sigma_J * rng.standard_normal(shape)
        return diff + jump

    def moments(self) -> dict:
        """Daily cumulants: kappa_n(jump part) = lam dt * E[J^n]."""
        ld = self.lam * C.DT
        mJ, sJ = self.mu_J, self.sigma_J
        mean = self.nu_d * C.DT + ld * mJ
        var = self.sigma_d ** 2 * C.DT + ld * (mJ ** 2 + sJ ** 2)
        k3 = ld * (mJ ** 3 + 3 * mJ * sJ ** 2)
        k4 = ld * (mJ ** 4 + 6 * mJ ** 2 * sJ ** 2 + 3 * sJ ** 4)
        return {"mean": mean, "var": var, "skew": k3 / var ** 1.5, "exkurt": k4 / var ** 2}

    def theory(self) -> dict:
        # No elementary closed form for two-sided exit with jumps -> Monte Carlo (Step 3).
        return {}

    def jumps_per_year(self) -> dict:
        s = pd.Series(1, index=pd.to_datetime(self.jump_dates))
        return s.groupby(s.index.year).sum().astype(int).to_dict()


def calibrate_all(r: pd.Series) -> dict:
    return {
        "RandomWalk": RandomWalkModel.calibrate(r),
        "GBM": GBMModel.calibrate(r),
        "Merton": MertonModel.calibrate(r),
    }


def rolling_window_returns(full_returns: pd.Series, entry_date, window: int = C.ROLLING_DAYS) -> pd.Series:
    """Trailing `window` daily log returns known at the Close of `entry_date`."""
    return full_returns.loc[:entry_date].iloc[-window:]


def calibrate_rolling_at(full_returns: pd.Series, entry_date, window: int = C.ROLLING_DAYS) -> dict:
    """Calibrate all three models on the trailing `window` trading days up to `entry_date`."""
    return calibrate_all(rolling_window_returns(full_returns, entry_date, window))


def rolling_entry_models(full_returns: pd.Series, entry_dates, window: int = C.ROLLING_DAYS) -> list[dict]:
    """List of calibrated model dicts, one per trade entry date."""
    return [calibrate_rolling_at(full_returns, d, window) for d in entry_dates]
