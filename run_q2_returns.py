"""Q2: do the models fit daily log returns? (out of sample, rolling monthly fits)

Every test-period day is scored with the latest rolling fit made before it. Also compares rolling window
lengths (3 / 5 / 7 years) on the training period, so the window is not chosen on test data.

Run:  .venv/Scripts/python run_q2_returns.py
Writes results/q2_<dataset>.json and plots q2_loglik / q2_qq / q2_clustering.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from barrier import config as C
from barrier import data as D
from barrier import diagnostics as G
from barrier import plotstyle as P
from barrier.models import calibration_grid
from barrier.rng import rng_for

COL = {"GBM": P.MODEL_COLOR["GBM"], "Merton": P.MODEL_COLOR["Merton"]}


def window_comparison(ds, r_full, train) -> dict:
    """Score the same training-period days with 3/5/7-year windows."""
    longest = max(C.WINDOW_YEARS_SENSITIVITY) * C.TRADING_DAYS
    days = train.index[longest + 1:]            # every window has enough history before these days
    res = {}
    for y in C.WINDOW_YEARS_SENSITIVITY:
        grid = calibration_grid(r_full, days, window=y * C.TRADING_DAYS)
        sc = G.score_days(grid, r_full.loc[days[0]:days[-1]])
        res[y] = {m: float(sc[m]["loglik"].mean()) for m in G.SCORED_MODELS}
        res[y]["n_days"] = int(len(sc["dates"]))
    best = {m: max(C.WINDOW_YEARS_SENSITIVITY, key=lambda y: res[y][m]) for m in G.SCORED_MODELS}
    return {"span": [str(days[0].date()), str(days[-1].date())], "by_years": res, "best": best}


def run(ds) -> dict:
    full, train, test, _ = D.load(ds)
    r_full = D.log_returns(full)
    grid = calibration_grid(r_full, test.index)
    sc = G.score_days(grid, r_full.loc[test.index[0]:test.index[-1]])
    n = len(sc["dates"])
    print(f"\n--- Q2 ({ds.label}): {n} out-of-sample days {sc['dates'][0].date()} -> "
          f"{sc['dates'][-1].date()}, {len(grid.dates)} monthly rolling fits")

    out = {"dataset": ds.key, "n_days": n, "first_day": str(sc["dates"][0].date()),
           "last_day": str(sc["dates"][-1].date()),
           "note_random_walk": "not scored: its daily return takes only the two values +/-dx, so any "
                               "real return has zero probability (log-likelihood = -inf)",
           "models": {}}
    for m in G.SCORED_MODELS:
        out["models"][m] = {"loglik_mean": float(sc[m]["loglik"].mean()),
                            **G.residual_tests(sc[m]["z"])}

    d = sc["Merton"]["loglik"] - sc["GBM"]["loglik"]
    ci = G.block_bootstrap_mean_ci(d, rng=rng_for(ds.key, "q2", "ci"))
    years = pd.Series(d, index=sc["dates"]).groupby(sc["dates"].year)
    out["merton_minus_gbm"] = {"mean_per_day": float(d.mean()), "ci95": ci, "total": float(d.sum()),
                               "by_year": {int(k): float(v) for k, v in years.sum().items()},
                               "share_days_merton_better": float(np.mean(d > 0))}
    out["window_comparison"] = window_comparison(ds, r_full, train)

    report(ds, out)
    plot_loglik(ds, sc, out)
    plot_qq(ds, sc, out)
    plot_clustering(ds, sc)
    return out


def report(ds, out) -> None:
    md = out["merton_minus_gbm"]
    print(f"{'model':<8}{'loglik/day':>11}{'z sd':>7}{'skew':>7}{'exkurt':>8}{'|z|>3':>8}{'|z|>4':>9}"
          f"{'JB p':>9}{'LB z^2 p':>10}{'ARCH p':>9}{'Berk p':>9}{'acf|z|':>8}")
    for m, s in out["models"].items():
        print(f"{m:<8}{s['loglik_mean']:>11.4f}{s['sd']:>7.3f}{s['skew']:>+7.2f}{s['exkurt']:>8.2f}"
              f"{s['share_gt3']:>8.2%}{s['share_gt4']:>9.3%}{s['jb_p']:>9.1e}{s['ljung_box_z2_p']:>10.1e}"
              f"{s['arch_lm_p']:>9.1e}{s['berkowitz_p']:>9.1e}{s['acf_abs_z_lag1']:>+8.3f}")
    print(f"(normal: |z|>3 {2*stats.norm.sf(3):.2%}, |z|>4 {2*stats.norm.sf(4):.4%})")
    print(f"Merton - GBM: {md['mean_per_day']:+.4f}/day (95% CI {md['ci95'][0]:+.4f} .. "
          f"{md['ci95'][1]:+.4f}), total {md['total']:+.0f}, Merton better on "
          f"{md['share_days_merton_better']:.0%} of days")
    top = sorted(md["by_year"].items(), key=lambda kv: -abs(kv[1]))[:4]
    print("  largest yearly contributions:", ", ".join(f"{y}: {v:+.0f}" for y, v in top))
    wc = out["window_comparison"]
    print(f"window comparison on training days {wc['span'][0]} -> {wc['span'][1]} (mean loglik/day):")
    for y, v in wc["by_years"].items():
        print(f"  {y} years: GBM {v['GBM']:.4f}   Merton {v['Merton']:.4f}")
    print(f"  best: GBM {wc['best']['GBM']}y, Merton {wc['best']['Merton']}y "
          f"(pipeline uses {C.ROLLING_YEARS}y)")


def plot_loglik(ds, sc, out) -> None:
    fig, axes = P.new_fig(w=9.0, h=3.8)
    ax = axes[0, 0]
    d = sc["Merton"]["loglik"] - sc["GBM"]["loglik"]
    cum = np.cumsum(d)
    ax.plot(sc["dates"], cum, color=COL["Merton"], lw=2)
    ax.axhline(0, color=P.INK2, lw=0.8)
    # label the two biggest 1-month gains (the crises where jumps matter)
    gain = cum[21:] - cum[:-21]
    used = []
    for i in np.argsort(gain)[::-1]:
        if all(abs(i - j) > 250 for j in used):
            used.append(i)
            j = i + 21
            ax.annotate(f"{sc['dates'][j]:%b %Y}: +{gain[i]:.0f}", (sc["dates"][j], cum[j]),
                        xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=8,
                        color=P.INK2, arrowprops=dict(arrowstyle="-", color=P.INK2, lw=0.7))
        if len(used) == 2:
            break
    md = out["merton_minus_gbm"]
    P.titles(ax, f"Cumulative log-likelihood, Merton minus GBM ({md['mean_per_day']:+.3f}/day, "
                 f"95% CI {md['ci95'][0]:+.3f} to {md['ci95'][1]:+.3f})",
             None, "Cumulative difference (nats)")
    P.save(fig, C.PLOTS_DIR / f"q2_loglik_{ds.key}.png",
           f"Q2 - Where Merton beats GBM out of sample, {ds.label}")


def plot_qq(ds, sc, out) -> None:
    fig, axes = P.new_fig(w=5.0, h=4.6)
    ax = axes[0, 0]
    for m in G.SCORED_MODELS:
        z = np.sort(sc[m]["z"])
        q = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        s = out["models"][m]
        ax.plot(q, z, ".", ms=2.5, color=COL[m],
                label=f"{P.MODEL_LABEL[m]} (|z|>4: {s['share_gt4']:.2%})")
    lim = 6.5
    ax.plot([-lim, lim], [-lim, lim], color=P.INK2, lw=0.9, ls="--", label="perfect fit")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-lim, lim)
    P.titles(ax, "PIT residuals vs normal quantiles", "Normal quantile", "Residual z")
    P.legend(ax, loc="upper left")
    P.save(fig, C.PLOTS_DIR / f"q2_qq_{ds.key}.png", f"Q2 - Tails, {ds.label}")


def plot_clustering(ds, sc) -> None:
    fig, axes = P.new_fig(w=7.4, h=4.0)
    ax = axes[0, 0]
    L = 60
    lags = np.arange(1, L + 1)
    for m in G.SCORED_MODELS:
        a = G.acf(np.abs(sc[m]["z"]), L)
        ax.plot(lags, a, color=COL[m], lw=2, label=P.MODEL_LABEL[m])
        P.end_label(ax, lags[-1], a[-1], m)
    ax.set_xlim(0, L + 8)
    band = 1.96 / np.sqrt(len(sc["dates"]))
    ax.axhspan(-band, band, color=P.GRID, alpha=0.8, lw=0, label="95% band if independent")
    ax.axhline(0, color=P.INK2, lw=0.8)
    P.titles(ax, "Autocorrelation of |z|: is volatility clustering left over?", "Lag (trading days)",
             "Autocorrelation")
    P.legend(ax, loc="upper right")
    P.save(fig, C.PLOTS_DIR / f"q2_clustering_{ds.key}.png", f"Q2 - Volatility clustering, {ds.label}")


def main() -> None:
    for ds in C.DATASETS.values():
        out = run(ds)
        path = C.RESULTS_DIR / f"q2_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
