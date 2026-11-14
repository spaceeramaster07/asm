"""Q1: does the simulator agree with the theory?

Parameters: the rolling 5-year fit at the first test date (any parameters would do; this is a code check).
Uncapped Monte Carlo pools are compared with
  - Random walk : exact gambler's-ruin formula on the lattice
  - GBM         : continuous-monitoring formula (biased for daily closes), the BGK-corrected formula,
                  and the exact daily-close answer from the Markov-chain grid solver
  - Merton      : exact daily-close answer from the grid solver (no closed form exists)
plus printed consistency checks (Merton with lambda = 0 vs GBM, grid convergence).

Run:  .venv/Scripts/python run_q1_theory.py
Writes results/q1_<dataset>.json and results/plots/q1_convergence_<dataset>.png
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import engine as E
from barrier import plotstyle as P
from barrier import theory as T
from barrier.models import calibration_grid
from barrier.rng import rng_for

NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
MAX_REPS = 1_000
# color = model (same as every other plot), line style = which reference it is compared with
SERIES = {
    "rw": ("Random walk vs exact lattice formula", P.MODEL_COLOR["RandomWalk"], "-"),
    "grid_gbm": ("Daily GBM vs exact daily grid", P.MODEL_COLOR["GBM"], "-"),
    "bgk": ("Daily GBM vs BGK-corrected formula", P.MODEL_COLOR["GBM"], ":"),
    "cont": ("Daily GBM vs continuous formula (biased)", P.MODEL_COLOR["GBM"], "--"),
    "grid_merton": ("Merton vs exact daily grid", P.MODEL_COLOR["Merton"], "-"),
}
SHORT = {"rw": "RW", "grid_gbm": "GBM grid", "bgk": "GBM BGK", "cont": "GBM continuous",
         "grid_merton": "Merton grid"}


def convergence(ep: E.Episodes, p_ref: float, t_ref: float) -> dict:
    """Split the pool into disjoint replications of size N; RMSE and 95% CI coverage per N."""
    out = {}
    win = (ep.outcome == E.WIN).astype(float)
    tau = ep.tau.astype(float)
    for N in NS:
        R = min(len(ep) // N, MAX_REPS)
        w = win[: R * N].reshape(R, N)
        t = tau[: R * N].reshape(R, N)
        p_hat, t_hat = w.mean(1), t.mean(1)
        p_se = np.sqrt(np.clip(p_hat * (1 - p_hat), 1e-12, None) / N)
        t_se = t.std(1, ddof=1) / math.sqrt(N)
        out[N] = {
            "reps": R,
            "p_rmse": float(np.sqrt(np.mean((p_hat - p_ref) ** 2))),
            "t_rmse": float(np.sqrt(np.mean((t_hat - t_ref) ** 2))),
            "p_cover": float(np.mean(np.abs(p_hat - p_ref) <= 1.96 * p_se)),
            "t_cover": float(np.mean(np.abs(t_hat - t_ref) <= 1.96 * t_se)),
        }
    return out


def q1(ds) -> dict:
    full, _, test, _ = D.load(ds)
    fit_date = test.index[0]
    models = calibration_grid(D.log_returns(full), test.index[:1]).fits[0]
    rw, gbm, mer = models["RandomWalk"], models["GBM"], models["Merton"]
    print(f"\n--- Q1 ({ds.label}): rolling {C.ROLLING_YEARS}y fit at {fit_date.date()}: "
          f"mu = {gbm.mu:+.2%}, sigma = {gbm.sigma:.2%}, lambda = {mer.lam:.1f}/yr; "
          f"pool = {C.N_CONVERGENCE_POOL:,} uncapped paths per model")

    t0 = time.time()
    pools = {k: E.simulate(m, C.N_CONVERGENCE_POOL, rng_for(ds.key, "q1", k))
             for k, m in (("rw", rw), ("gbm", gbm), ("merton", mer))}
    print(f"simulated in {time.time()-t0:.1f}s")

    th_rw, th_g = rw.theory(), gbm.theory()
    t0 = time.time()
    p_grid_g, t_grid_g = T.grid_first_passage(gbm.daily_cdf)
    p_grid_m, t_grid_m = T.grid_first_passage(mer.daily_cdf)
    print(f"grid solves in {time.time()-t0:.1f}s")
    refs = {
        "rw": (pools["rw"], th_rw["p_win_exact_lattice"], th_rw["E_tau_days_exact_lattice"]),
        "grid_gbm": (pools["gbm"], p_grid_g, t_grid_g),
        "bgk": (pools["gbm"], th_g["p_win_daily_bgk"], th_g["E_tau_days_daily_bgk"]),
        "cont": (pools["gbm"], th_g["p_win_continuous"], th_g["E_tau_days_continuous"]),
        "grid_merton": (pools["merton"], p_grid_m, t_grid_m),
    }

    out = {"dataset": ds.key, "fit_date": str(fit_date.date()), "pool": C.N_CONVERGENCE_POOL,
           "params": {"mu": gbm.mu, "sigma": gbm.sigma, "lam": mer.lam, "mu_J": mer.mu_J,
                      "sigma_J": mer.sigma_J, "rw_p": rw.p, "rw_A": rw.A, "rw_B": rw.B},
           "comparisons": {}}
    print(f"\n{'comparison':<40}{'theory p':>10}{'sim p':>9}{'z':>7}{'theory E[t]':>13}{'sim':>9}{'z':>7}")
    for key, (ep, p_ref, t_ref) in refs.items():
        s = E.summarize(ep)
        zp = (s["p_win"] - p_ref) / s["p_win_se"]
        zt = (s["tau_mean"] - t_ref) / s["tau_se"]
        print(f"{SERIES[key][0]:<40}{p_ref:>10.5f}{s['p_win']:>9.5f}{zp:>+7.1f}"
              f"{t_ref:>13.2f}{s['tau_mean']:>9.2f}{zt:>+7.1f}")
        out["comparisons"][key] = {"label": SERIES[key][0], "p_ref": p_ref, "t_ref": t_ref,
                                   "sim_p": s["p_win"], "sim_p_se": s["p_win_se"], "z_p": zp,
                                   "sim_t": s["tau_mean"], "sim_t_se": s["tau_se"], "z_t": zt,
                                   "by_N": convergence(ep, p_ref, t_ref)}

    # ---- printed consistency checks (no plot)
    print("\nconsistency checks")
    chk = {}
    mer0 = replace(mer, lam=0.0, nu_d=gbm.nu, sigma_d=gbm.sigma)
    p0, t0_ = T.grid_first_passage(mer0.daily_cdf)
    chk["merton_lam0_vs_gbm"] = {"dp": p0 - p_grid_g, "dt": t0_ - t_grid_g}
    print(f"  Merton with lambda = 0 vs GBM (grid): dp = {p0 - p_grid_g:+.1e}, dE[t] = {t0_ - t_grid_g:+.1e}")
    assert abs(p0 - p_grid_g) < 1e-10 and abs(t0_ - t_grid_g) < 1e-7
    ep0 = E.simulate(mer0, 200_000, rng_for(ds.key, "q1", "merton0"))
    s0 = E.summarize(ep0)
    z0 = (s0["p_win"] - p_grid_g) / s0["p_win_se"]
    chk["merton_lam0_sim_z"] = z0
    print(f"  Merton with lambda = 0, simulated (200k) vs GBM grid: z(win) = {z0:+.1f}")
    assert abs(z0) < 4
    conv = {}
    for n in (C.GRID_N // 2, C.GRID_N, 2 * C.GRID_N):
        conv[n] = T.grid_first_passage(mer.daily_cdf, n=n)
    dn = abs(conv[2 * C.GRID_N][0] - conv[C.GRID_N][0])
    chk["grid_convergence_merton"] = {str(n): v for n, v in conv.items()}
    print(f"  grid convergence (Merton p_win): n = {C.GRID_N // 2}: {conv[C.GRID_N // 2][0]:.6f}, "
          f"{C.GRID_N}: {conv[C.GRID_N][0]:.6f}, {2 * C.GRID_N}: {conv[2 * C.GRID_N][0]:.6f}  "
          f"(change on doubling {dn:.1e})")
    assert dn < 1e-4
    bgk_err = {"dp": th_g["p_win_daily_bgk"] - p_grid_g, "dt": th_g["E_tau_days_daily_bgk"] - t_grid_g}
    chk["bgk_leftover_error"] = bgk_err
    print(f"  BGK leftover error vs exact daily grid: dp = {bgk_err['dp']:+.5f}, "
          f"dE[t] = {bgk_err['dt']:+.2f} days")
    out["checks"] = chk
    plot_q1(ds, out)
    return out


def plot_q1(ds, out) -> None:
    import matplotlib.ticker as tk
    fig, axes = P.new_fig(ncols=2, w=5.8, h=4.4)
    for ax, metric, title, unit in [(axes[0, 0], "p_rmse", "Win probability", "RMSE (probability)"),
                                    (axes[0, 1], "t_rmse", "Mean stopping time", "RMSE (days)")]:
        Ns = np.array(NS)
        for key, (label, col, ls) in SERIES.items():
            by = out["comparisons"][key]["by_N"]
            by = {int(k): v for k, v in by.items()}
            ys = [by[N][metric] for N in NS]
            ax.plot(Ns, ys, color=col, lw=2, ls=ls, marker="o", ms=4, label=label)
            if key in ("cont", "grid_merton"):
                P.end_label(ax, Ns[-1], ys[-1], SHORT[key])
        ax.set_xlim(NS[0] * 0.8, NS[-1] * 3.2)
        y0 = {int(k): v for k, v in out["comparisons"]["grid_gbm"]["by_N"].items()}[NS[0]][metric]
        ax.plot(Ns, y0 * np.sqrt(NS[0] / Ns), color=P.INK2, lw=1, ls=(0, (1, 3)),
                label=r"$\propto 1/\sqrt{N}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(tk.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(tk.NullFormatter())
        P.titles(ax, title, "Episodes per estimate, N", unit)
    P.legend(axes[0, 0], loc="lower left", fontsize=7.8)
    P.save(fig, C.PLOTS_DIR / f"q1_convergence_{ds.key}.png",
           f"Monte Carlo convergence to theory ({ds.label}, rolling fit at {out['fit_date']})")


def main() -> None:
    import sys
    if "plots" in sys.argv[1:]:                      # redraw from saved JSON, no simulation
        for ds in C.DATASETS.values():
            plot_q1(ds, json.loads((C.RESULTS_DIR / f"q1_{ds.key}.json").read_text()))
        return
    for ds in C.DATASETS.values():
        res = q1(ds)
        path = C.RESULTS_DIR / f"q1_{ds.key}.json"
        path.write_text(json.dumps(res, indent=2, default=str))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
