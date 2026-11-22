"""Q4: what bet size does each model recommend? (rolling monthly fits, exact exit distributions)

For every monthly fit and model: Kelly fraction f*, growth at f*, Half-Kelly share of the excess growth
over cash, cash break-even and the ruin cliff. Also: f* when the drift is one standard error higher /
lower, f* with the real stop fills swapped in (diagnostic: uses test-period fills), and the Kelly lookup
tables the Q5 traders use (with an accuracy check).

Run:  .venv/Scripts/python run_q4_kelly.py   (after run_q3_barriers.py)
Writes results/q4_<dataset>.json and plots q4_growth / q4_fstar_series.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import kelly as K
from barrier import plotstyle as P
from barrier import theory as T
from barrier.models import GBMModel, MertonModel
from barrier.predict import MODELS, load_grid_and_predictions
from barrier.rng import rng_for

FS = np.round(np.arange(0, C.F_GRID_MAX + 1e-9, 0.02), 4)


def _solve_gbm(args):
    nu, sigma = args
    return K.compress(T.exit_distribution(GBMModel(nu=nu, sigma=sigma, n_obs=C.ROLLING_DAYS)))


def per_fit(ds, grid, preds, fills) -> dict:
    rf = ds.rf
    rows = {m: [] for m in MODELS}
    curves = {m: [] for m in MODELS}
    for k in range(len(grid.fits)):
        for m in MODELS:
            xd = preds[m][k]
            o = K.compress(xd)
            f = K.kelly_fraction(o, rf)
            g0 = K.growth_per_year(o, 0.0, rf)
            gs = K.growth_per_year(o, f, rf)
            gh = K.growth_per_year(o, f / 2, rf)
            be = K.cash_breakeven(o, rf, f)
            rows[m].append({"date": str(grid.dates[k].date()), "f_star": f, "g_star": gs, "g_cash": g0,
                            "g_half": gh,
                            "half_share": (gh - g0) / (gs - g0) if gs - g0 > 1e-9 else None,
                            "cash_breakeven": be,
                            "breakeven_over_fstar": be / f if (be and f > 0) else None,
                            "ruin_cliff": K.ruin_cliff(xd)})
            curves[m].append(K.growth_curve(o, FS, rf))
    # drift +/- 1 SE (GBM) and real-fill swap
    gbms = [grid.fits[k]["GBM"] for k in range(len(grid.fits))]
    jobs = [(g.nu + s * g.nu_se, g.sigma) for g in gbms for s in (-1, 1)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(_solve_gbm, jobs, chunksize=4))
    for k in range(len(grid.fits)):
        rows["GBM"][k]["f_star_drift_lo"] = K.kelly_fraction(res[2 * k], rf)
        rows["GBM"][k]["f_star_drift_hi"] = K.kelly_fraction(res[2 * k + 1], rf)
        rows["GBM"][k]["f_star_real_fills"] = K.kelly_fraction(K.swap_ruin_fills(preds["GBM"][k], fills), rf)
    return rows, {m: np.array(v) for m, v in curves.items()}


def table_check(table, rf, mu_J, sigma_J, rng, n=60) -> dict:
    """Interpolated table vs direct computation at random parameter points."""
    mu = rng.uniform(-0.05, 0.30, n)
    sg = rng.uniform(0.10, 0.35, n)
    lam = rng.uniform(2, 15, n)
    err_g, err_m, cap_g, cap_m = [], [], [], []
    cap = lambda x: min(x, C.LEVERAGE_CAP)
    for a, b, l in zip(mu, sg, lam):
        direct = K.kelly_fraction(K.compress(T.exit_distribution(K._gbm_from(a, b), n=K.TABLE_GRID_N)), rf)
        tab = float(table.f_gbm(a, b)[0])
        err_g.append(tab - direct)
        cap_g.append(cap(tab) - cap(direct))
        mm = K.merton_from(a, b, l, mu_J, sigma_J)
        if mm is not None:
            direct = K.kelly_fraction(K.compress(T.exit_distribution(mm, n=K.TABLE_GRID_N)), rf)
            tab = float(table.f_merton(a, b, l)[0])
            err_m.append(tab - direct)
            cap_m.append(cap(tab) - cap(direct))
    a_ = lambda v: np.abs(np.array(v))
    return {"gbm_max_abs_err": float(a_(err_g).max()), "gbm_mean_abs_err": float(a_(err_g).mean()),
            "merton_max_abs_err": float(a_(err_m).max()), "merton_mean_abs_err": float(a_(err_m).mean()),
            "gbm_max_abs_err_capped": float(a_(cap_g).max()), "merton_max_abs_err_capped": float(a_(cap_m).max()),
            "n_points": n}


def run(ds) -> dict:
    full, test, grid, preds = load_grid_and_predictions(ds)
    q3 = json.loads((C.RESULTS_DIR / f"q3_{ds.key}.json").read_text())
    fills = np.array(q3["stop_fill"]["real_values"])
    rows, curves = per_fit(ds, grid, preds, fills)

    _, train, _, _ = D.load(ds)
    mtrain = MertonModel.calibrate(D.log_returns(train))
    table = K.load_or_build_table(ds, mtrain.mu_J, mtrain.sigma_J)
    check = table_check(table, ds.rf, mtrain.mu_J, mtrain.sigma_J, rng_for(ds.key, "q4", "tablecheck"))

    out = {"dataset": ds.key, "rf": ds.rf, "borrow_spread": C.BORROW_SPREAD, "trade_cost": C.TRADE_COST,
           "per_fit": rows, "table": {"mu_J": mtrain.mu_J, "sigma_J": mtrain.sigma_J, **check},
           "summary": {}}
    for m in MODELS:
        f = np.array([r["f_star"] for r in rows[m]])
        hs = np.array([r["half_share"] for r in rows[m] if r["half_share"] is not None])
        be = np.array([r["breakeven_over_fstar"] for r in rows[m] if r["breakeven_over_fstar"]])
        cl = np.array([r["ruin_cliff"] for r in rows[m]])
        gs = np.array([r["g_star"] - r["g_cash"] for r in rows[m]])
        out["summary"][m] = {"f_star_mean": float(f.mean()), "f_star_median": float(np.median(f)),
                             "f_star_min": float(f.min()), "f_star_max": float(f.max()),
                             "share_months_f_above_cap": float(np.mean(f > C.LEVERAGE_CAP)),
                             "share_months_f_zero": float(np.mean(f < 1e-3)),
                             "excess_growth_mean": float(gs.mean()),
                             "half_share_median": float(np.median(hs)) if hs.size else None,
                             "breakeven_over_fstar_median": float(np.median(be)) if be.size else None,
                             "ruin_cliff_median": float(np.median(cl))}
        out["summary"][m]["mean_curve"] = curves[m].mean(0).tolist()
        # a wipe-out month has growth -inf: keep it (it is the worst outcome), so a percentile curve
        # ends where that share of months would be wiped out instead of averaging the survivors
        cv = np.where(np.isnan(curves[m]), -np.inf, curves[m])
        with np.errstate(all="ignore"):
            for q in (10, 50, 90):
                out["summary"][m][f"curve_p{q}"] = [None if not np.isfinite(v) else float(v)
                                                    for v in np.percentile(cv, q, axis=0, method="lower")]
    g = rows["GBM"]
    out["summary"]["GBM"]["f_star_real_fills_mean"] = float(np.mean([r["f_star_real_fills"] for r in g]))
    out["summary"]["GBM"]["drift_band_width_mean"] = float(np.mean([r["f_star_drift_hi"] - r["f_star_drift_lo"]
                                                                    for r in g]))
    out["fs"] = FS.tolist()
    report(ds, out)
    plot_growth(ds, out)
    plot_series(ds, out)
    return out


def report(ds, out) -> None:
    print(f"\n--- Q4 ({ds.label}): r_f = {ds.rf:.1%}, borrow r_f + {C.BORROW_SPREAD:.0%}, "
          f"cost {C.TRADE_COST:.2%} per side, {len(out['per_fit']['GBM'])} monthly fits")
    print(f"{'model':<11}{'f* mean':>8}{'median':>8}{'range':>14}{'>3x':>6}{'=0':>6}{'excess g/yr':>12}"
          f"{'half share':>11}{'b-even/f*':>10}{'cliff':>7}")
    for m in MODELS:
        s = out["summary"][m]
        print(f"{m:<11}{s['f_star_mean']:>8.2f}{s['f_star_median']:>8.2f}"
              f"{s['f_star_min']:>7.2f}-{s['f_star_max']:<6.2f}{s['share_months_f_above_cap']:>6.0%}"
              f"{s['share_months_f_zero']:>6.0%}{s['excess_growth_mean']:>+12.2%}"
              f"{(s['half_share_median'] or float('nan')):>11.0%}"
              f"{(s['breakeven_over_fstar_median'] or float('nan')):>10.2f}{s['ruin_cliff_median']:>7.1f}")
    g = out["summary"]["GBM"]
    print(f"GBM f* with real stop fills (diagnostic): mean {g['f_star_real_fills_mean']:.2f}; "
          f"drift +/-1 SE moves f* by {g['drift_band_width_mean']:.2f} on average")
    t = out["table"]
    print(f"Kelly table accuracy ({t['n_points']} random points): GBM max |err| {t['gbm_max_abs_err']:.3f} "
          f"(mean {t['gbm_mean_abs_err']:.3f}); Merton max {t['merton_max_abs_err']:.3f} "
          f"(mean {t['merton_mean_abs_err']:.3f}); within the 3x cap: GBM max "
          f"{t['gbm_max_abs_err_capped']:.3f}, Merton max {t['merton_max_abs_err_capped']:.3f}")


def plot_growth(ds, out) -> None:
    fig, axes = P.new_fig(ncols=2, w=5.4, h=4.4, sharey=True)
    # bets are capped at 3x in the game, so the curves are drawn up to the cap
    fs_all = np.array(out["fs"])
    keep = fs_all <= C.LEVERAGE_CAP + 1e-9
    fs = fs_all[keep]
    arr = lambda v: np.array([np.nan if x is None else x for x in v], float)[keep]
    lo_all = []
    for ax, m in zip(axes[0], ("GBM", "Merton")):
        s = out["summary"][m]
        p10, p50, p90 = arr(s["curve_p10"]), arr(s["curve_p50"]), arr(s["curve_p90"])
        col = P.MODEL_COLOR[m]
        ax.fill_between(fs, p10, p90, color=col, alpha=0.18, lw=0, label="10th-90th percentile of months")
        ax.plot(fs, p50, color=col, lw=2, label="median month")
        i = int(np.nanargmax(p50))
        ax.plot(fs[i], p50[i], "o", color=col, ms=8, mec=P.SURFACE, mew=1.5)
        P.end_label(ax, fs[i], p50[i], f"median-month peak {fs[i]:.1f}x", dx=-30, dy=11)
        h = int(np.argmin(np.abs(fs - fs[i] / 2)))
        ax.plot(fs[h], p50[h], "o", color=P.SURFACE, ms=7, mec=col, mew=2)
        P.end_label(ax, fs[h], p50[h], "Half Kelly", dx=-4, dy=-11)
        ax.axhline(ds.rf, color=P.INK2, lw=1, ls="--", label=f"cash ({ds.rf:.1%})")
        lo_all.append(np.nanmin(p10))
        P.pct_axis(ax.yaxis, 0)
        P.titles(ax, f"{P.MODEL_LABEL[m]}: f* in {s['f_star_min']:.1f}-{s['f_star_max']:.1f}x across months",
                 "Fraction of wealth bet, f (up to the 3x cap)",
                 "Expected growth per year" if m == "GBM" else None)
        ax.set_xlim(0, C.LEVERAGE_CAP)
    axes[0, 0].set_ylim(max(-0.10, min(lo_all) - 0.01), None)
    P.legend(axes[0, 0], loc="lower left", fontsize=7.5)
    P.save(fig, C.PLOTS_DIR / f"q4_growth_{ds.key}.png",
           f"Q4 - Growth vs bet size from each month's rolling fit, {ds.label} (random walk ~ GBM)")


def plot_series(ds, out) -> None:
    import pandas as pd
    fig, axes = P.new_fig(w=9.6, h=4.4)
    ax = axes[0, 0]
    g = out["per_fit"]["GBM"]
    d = pd.to_datetime([r["date"] for r in g])
    ax.fill_between(d, [r["f_star_drift_lo"] for r in g], [r["f_star_drift_hi"] for r in g],
                    color=P.MODEL_COLOR["GBM"], alpha=0.15, lw=0, label="GBM f*, drift +/-1 SE")
    for m in MODELS:
        ax.plot(d, [r["f_star"] for r in out["per_fit"][m]], color=P.MODEL_COLOR[m], lw=1.6,
                label=f"{P.MODEL_LABEL[m]} f*")
    ax.plot(d, [r["f_star_real_fills"] for r in g], color=P.INK, lw=1.1, ls="--",
            label="GBM f* with real stop fills (diagnostic)")
    ax.axhline(C.LEVERAGE_CAP, color=P.INK2, lw=0.8, ls=":", label="3x cap")
    ax.set_ylim(-0.2, C.F_GRID_MAX + 2.2)
    P.titles(ax, "Kelly fraction from each month's rolling 5-year fit", None, "Kelly fraction f*")
    P.legend(ax, loc="upper center", ncol=3, fontsize=7.8)
    P.save(fig, C.PLOTS_DIR / f"q4_fstar_series_{ds.key}.png", f"Q4 - Rolling Kelly fraction, {ds.label}")


def main() -> None:
    import sys
    if "plots" in sys.argv[1:]:                      # redraw from saved JSON, no recomputation
        for ds in C.DATASETS.values():
            out = json.loads((C.RESULTS_DIR / f"q4_{ds.key}.json").read_text())
            plot_growth(ds, out)
            plot_series(ds, out)
        return
    for ds in C.DATASETS.values():
        out = run(ds)
        path = C.RESULTS_DIR / f"q4_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=lambda o: None if o is None else str(o)))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
