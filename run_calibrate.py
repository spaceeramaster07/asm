"""Step 0: load data, check the formulas, and build the monthly rolling calibration grid.

Run:  .venv/Scripts/python run_calibrate.py
Writes results/calibration_<dataset>.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import theory as T
from barrier.models import calibration_grid, sample_moments
from barrier.rng import rng_for

N_CHECK = 2_000_000  # simulated increments per model for the moment check


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def self_checks() -> None:
    hr("0. Formula self-checks")
    pw, _ = T.rw_first_passage(0.6, 1, 1)
    assert abs(pw - 0.6) < 1e-12
    for p, A, B in [(0.52, 7, 12), (0.48, 9, 15), (0.5, 8, 14)]:
        f = T.rw_first_passage(p, A, B)
        ls = T.rw_first_passage_linear_system(p, A, B)
        assert abs(f[0] - ls[0]) < 1e-9 and abs(f[1] - ls[1]) < 1e-6
    sig = 0.2
    p_mu0, _ = T.gbm_first_passage(-0.5 * sig ** 2, sig)
    assert abs(p_mu0 - 1 / 3) < 1e-12
    p0, _ = T.gbm_first_passage(0.0, sig)
    assert abs(T.gbm_first_passage(1e-7, sig)[0] - p0) < 1e-5
    print(f"zero LOG drift (nu=0): p_win = {p0:.4f};  zero PRICE drift (mu=0): p_win = "
          f"{T.price_martingale_p_win():.4f} (exactly 1/3)")
    print("all self-checks passed")


def moment_check(fits: dict, r, rng) -> None:
    """Every model reproduces the window's daily mean and variance exactly."""
    sm = sample_moments(r)
    for name, m in fits.items():
        th = m.moments()
        assert abs(th["mean"] - sm["mean"]) < 1e-12, name
        assert abs(th["var"] - sm["var"]) < 1e-12, name
        sim = sample_moments(m.increments(rng, N_CHECK))
        print(f"  {name:<11} model sd {math.sqrt(th['var'])*100:.3f}%  sim sd "
              f"{math.sqrt(sim['var'])*100:.3f}%   skew {th['skew']:+.2f} (data {sm['skew']:+.2f})"
              f"   exkurt {th['exkurt']:6.2f} (data {sm['exkurt']:.2f})")


def run_dataset(ds: C.Dataset) -> None:
    hr(f"{ds.label} ({ds.ticker})")
    full, train, test, q = D.load(ds)
    for k, v in q.items():
        print(f"{k:<32}{v}")
    print(f"{'burn-in (train) rows':<32}{len(train)}  ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"{'test rows':<32}{len(test)}  ({test.index[0].date()} -> {test.index[-1].date()})")

    r_full = D.log_returns(full)
    grid = calibration_grid(r_full, test.index)
    print(f"\nRolling {C.ROLLING_YEARS}-year calibration: {len(grid.dates)} monthly fits "
          f"({grid.dates[0].date()} -> {grid.dates[-1].date()})")

    first = grid.fits[0]
    print(f"moment check on the first window (ending {grid.dates[0].date()}):")
    moment_check(first, r_full.loc[:grid.dates[0]].iloc[-C.ROLLING_DAYS:], rng_for(ds.key, "moments"))

    rows = []
    for d, f in zip(grid.dates, grid.fits):
        g, rw, m = f["GBM"], f["RandomWalk"], f["Merton"]
        rows.append({"date": str(d.date()), "mu": g.mu, "nu": g.nu, "sigma": g.sigma, "nu_se": g.nu_se,
                     "rw_p": rw.p, "rw_dx": rw.dx, "rw_A": rw.A, "rw_B": rw.B,
                     "lam": m.lam, "mu_J": m.mu_J, "sigma_J": m.sigma_J, "sigma_d": m.sigma_d,
                     "nu_d": m.nu_d, "n_jumps": m.n_jumps})
    col = lambda k: np.array([r[k] for r in rows])
    for k, lab, scale in [("mu", "mu /yr", 1), ("sigma", "sigma /yr", 1), ("nu_se", "drift SE /yr", 1),
                          ("lam", "jumps /yr", 1), ("mu_J", "mean jump", 1)]:
        v = col(k) * scale
        print(f"  {lab:<14} mean {v.mean():+.4f}   range {v.min():+.4f} .. {v.max():+.4f}")

    out = {"dataset": ds.key, "data_quality": q, "window_days": C.ROLLING_DAYS,
           "config": {"train": ds.train, "test": ds.test, "dt": C.DT, "b": C.B_LOG, "a": C.A_LOG,
                      "jump_k": C.JUMP_K},
           "grid": rows}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    path = C.RESULTS_DIR / f"calibration_{ds.key}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved {path}")


def main() -> None:
    self_checks()
    for ds in C.DATASETS.values():
        run_dataset(ds)


if __name__ == "__main__":
    main()
