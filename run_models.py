"""Steps 1-2: load data, calibrate the three models, verify them, and print the closed-form theory.

Run:  .venv/Scripts/python run_models.py
Writes results/calibration.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import theory as T
from barrier.models import GBMModel, calibrate_all, sample_moments

RNG_SEED = 20240601
N_CHECK = 2_000_000  # increments drawn per model to check calibration


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def fmt_mom(m: dict) -> str:
    return (f"mean={m['mean']*1e4:+8.3f}bp  sd={math.sqrt(m['var'])*100:6.3f}%  "
            f"skew={m['skew']:+6.2f}  exkurt={m['exkurt']:6.2f}")


def self_checks() -> None:
    hr("0. Formula self-checks")
    pw, _ = T.rw_first_passage(0.6, 1, 1)
    print(f"RW A=B=1, p=0.6 -> p_win = {pw:.4f} (must be 0.6000)")
    assert abs(pw - 0.6) < 1e-12
    for p, A, B in [(0.52, 7, 12), (0.48, 9, 15), (0.5, 8, 14)]:
        f = T.rw_first_passage(p, A, B)
        ls = T.rw_first_passage_linear_system(p, A, B)
        print(f"RW p={p} A={A} B={B}: formula ({f[0]:.6f}, {f[1]:.3f})  "
              f"linear system ({ls[0]:.6f}, {ls[1]:.3f})")
        assert abs(f[0] - ls[0]) < 1e-9 and abs(f[1] - ls[1]) < 1e-6
    p0, _ = T.gbm_first_passage(0.0, 0.2)
    pm = T.price_martingale_p_win()
    print(f"zero LOG drift (nu=0)        : p_win = {p0:.4f}  (= |a|/(b-|a|...) = 36.6%)")
    print(f"zero PRICE drift (mu=0)      : p_win = {pm:.4f}  (exactly 1/3)")
    sig = 0.2
    p_mu0, _ = T.gbm_first_passage(-0.5 * sig ** 2, sig)
    assert abs(p_mu0 - 1 / 3) < 1e-12, p_mu0
    p_eps, _ = T.gbm_first_passage(1e-7, sig)
    assert abs(p_eps - p0) < 1e-5
    print("all self-checks passed")


def report_period(label: str, r, rng) -> dict:
    hr(f"{label}: {r.index[0].date()} -> {r.index[-1].date()}  ({len(r)} returns, "
       f"{len(r)*C.DT:.2f} yrs @252)")
    models = calibrate_all(r)
    rw, gbm, mer = models["RandomWalk"], models["GBM"], models["Merton"]
    sm = sample_moments(r)

    print("\n-- Daily log-return moments: data vs model (theory) vs model (2M simulated draws)")
    print(f"{'data':<12}{fmt_mom(sm)}")
    for name, m in models.items():
        sim = sample_moments(m.increments(rng, N_CHECK))
        print(f"{name:<12}{fmt_mom(m.moments())}")
        print(f"{'  (sim)':<12}{fmt_mom(sim)}")
        assert abs(m.moments()["mean"] - sm["mean"]) < 1e-12, name
        assert abs(m.moments()["var"] - sm["var"]) < 1e-12, name

    print("\n-- GBM")
    print(f"nu (log drift)  = {gbm.nu:+.4f}/yr   SE = {gbm.nu_se:.4f}  "
          f"(95% CI {gbm.nu-1.96*gbm.nu_se:+.3f} .. {gbm.nu+1.96*gbm.nu_se:+.3f})")
    print(f"mu (arith drift)= {gbm.mu:+.4f}/yr   sigma = {gbm.sigma:.4f}/yr")

    print("\n-- Random walk")
    print(f"dx = {rw.dx:.5f}   p = {rw.p:.5f}   B = {rw.B} steps   A = {rw.A} steps")
    print(f"lattice barriers actually used: {rw.effective_up_return:+.2%} / "
          f"{rw.effective_down_return:+.2%}   (target +20.00% / -10.00%)")

    print("\n-- Merton jump-diffusion (iterated 3-sigma filter, "
          f"{mer.filter_iterations} iterations)")
    print(f"jumps = {mer.n_jumps}  -> lambda = {mer.lam:.2f}/yr")
    print(f"mu_J = {mer.mu_J:+.4f}   sigma_J = {mer.sigma_J:.4f}   k = {mer.k:+.4f}")
    print(f"sigma_d: filter {mer.sigma_d_filter:.4f} -> moment-matched {mer.sigma_d:.4f}"
          f"   nu_d = {mer.nu_d:+.4f}   implied mu = {mer.mu:+.4f}")
    print(f"jump days per year: {mer.jumps_per_year()}")
    order = np.argsort(mer.jump_sizes)
    print("most negative:", [(mer.jump_dates[i], f"{mer.jump_sizes[i]:+.3f}") for i in order[:4]])
    print("most positive:", [(mer.jump_dates[i], f"{mer.jump_sizes[i]:+.3f}")
                             for i in order[::-1][:4]])

    print("\n-- Closed-form barrier theory (+20% / -10%, no time cap)")
    th_rw, th_g = rw.theory(), gbm.theory()
    print(f"{'model':<28}{'p_win':>8}{'p_ruin':>8}{'E[tau] days':>13}{'stop return':>13}")
    print(f"{'RW exact lattice':<28}{th_rw['p_win_exact_lattice']:>8.2%}"
          f"{1-th_rw['p_win_exact_lattice']:>8.2%}{th_rw['E_tau_days_exact_lattice']:>13.1f}"
          f"{th_rw['expected_stop_return_daily']:>+13.2%}")
    print(f"{'GBM continuous':<28}{th_g['p_win_continuous']:>8.2%}"
          f"{1-th_g['p_win_continuous']:>8.2%}{th_g['E_tau_days_continuous']:>13.1f}"
          f"{-0.10:>+13.2%}")
    print(f"{'GBM daily closes (BGK)':<28}{th_g['p_win_daily_bgk']:>8.2%}"
          f"{1-th_g['p_win_daily_bgk']:>8.2%}{th_g['E_tau_days_daily_bgk']:>13.1f}"
          f"{th_g['expected_stop_return_daily']:>+13.2%}")
    print(f"{'Merton':<28}{'-- needs Monte Carlo (Step 3) --':>42}")

    print("\n-- Drift uncertainty -> p_win (GBM continuous, sigma fixed)")
    for z in (-1.96, -1, 0, 1, 1.96):
        nu = gbm.nu + z * gbm.nu_se
        p, t = T.gbm_first_passage(nu, gbm.sigma)
        print(f"  nu = {nu:+.3f} ({z:+.2f} SE): p_win = {p:6.2%}   E[tau] = {t*252:6.1f} d")

    return {
        "period": [str(r.index[0].date()), str(r.index[-1].date())],
        "sample_moments": sm,
        "GBM": {**asdict(gbm), "mu": gbm.mu, "nu_se": gbm.nu_se, "theory": th_g},
        "RandomWalk": {**asdict(rw), "effective_up": rw.effective_up_return,
                       "effective_down": rw.effective_down_return, "theory": th_rw},
        "Merton": {**asdict(mer), "mu": mer.mu, "k": mer.k, "moments": mer.moments()},
    }


def sensitivity(sigma: float) -> None:
    hr(f"p_win(mu) sensitivity curve, GBM continuous, sigma = {sigma:.3f} (training)")
    print(f"{'mu':>7}{'nu':>8}{'p_win':>9}{'E[tau] d':>10}")
    for mu in np.arange(-0.05, 0.2501, 0.025):
        nu = mu - 0.5 * sigma ** 2
        p, t = T.gbm_first_passage(nu, sigma)
        print(f"{mu:>+7.3f}{nu:>+8.3f}{p:>9.2%}{t*252:>10.1f}")


def run_dataset(ds: C.Dataset, rng) -> None:
    print("\n\n" + "#" * 78 + f"\n#  {ds.label} ({ds.ticker})\n" + "#" * 78)
    full, train, test, q = D.load(ds)
    hr("1. Data quality")
    for k, v in q.items():
        print(f"{k:<32}{v}")
    print(f"{'train rows':<32}{len(train)}  ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"{'test rows':<32}{len(test)}  ({test.index[0].date()} -> {test.index[-1].date()})")

    out = {"dataset": ds.key, "data_quality": q,
           "config": {"train": ds.train, "test": ds.test, "dt": C.DT,
                      "b": C.B_LOG, "a": C.A_LOG, "jump_k": C.JUMP_K}}
    out["train"] = report_period("2. TRAINING calibration (used for all predictions)",
                                 D.log_returns(train), rng)
    out["test_oracle"] = report_period(
        "3. TEST-period calibration (ORACLE - diagnostics only, never used for predictions)",
        D.log_returns(test), rng)
    sensitivity(GBMModel.calibrate(D.log_returns(train)).sigma)

    C.RESULTS_DIR.mkdir(exist_ok=True)
    path = C.RESULTS_DIR / f"calibration_{ds.key}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved {path}")


def main() -> None:
    self_checks()
    rng = np.random.default_rng(RNG_SEED)
    for ds in C.DATASETS.values():
        run_dataset(ds, rng)


if __name__ == "__main__":
    main()
