"""Q3: do the models predict the +20% / -10% barrier game? (out of sample, rolling monthly fits)

For every test-period start day with a full 504-day window, each model's latest rolling fit gives an
exact capped forecast (theory.exit_distribution): P(win), P(ruin), P(timeout), the exit-day
distribution and the stop-fill distribution. Forecasts are scored against what happened:
  1. outcome: 3-class log score and Brier score, vs a no-model baseline (trailing outcome frequencies
     of past start days that had already resolved), plus a calibration plot of P(win);
  2. duration: randomised PIT of the realised exit day (uniform if the forecast is right);
  3. paired ranking: per-start-day score differences with block-bootstrap CIs;
  4. stop fills: real vs predicted distribution of the loss on a stopped trade (input to Q4/Q5).

Run:  .venv/Scripts/python run_q3_barriers.py
Writes results/q3_<dataset>.json and plots q3_scores / q3_calibration / q3_pit / q3_stopfill.
"""
from __future__ import annotations

import json

import numpy as np

from barrier import config as C
from barrier import plotstyle as P
from barrier import scanner as S
from barrier.engine import RUIN, TIMEOUT, WIN
from barrier.predict import MODELS, load_grid_and_predictions
from barrier.rng import rng_for

CLASSES = (WIN, RUIN, TIMEOUT)
CLASS_NAMES = ("win", "ruin", "timeout")
BASELINE = "No model (trailing frequencies)"


def class_index(outcome: np.ndarray) -> np.ndarray:
    return np.select([outcome == WIN, outcome == RUIN], [0, 1], 2)


def block_indices(n, block, reps, rng):
    nb = -(-n // block)
    return ((rng.integers(0, n, (reps, nb))[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def baseline_probs(full, start_dates, window=C.ROLLING_DAYS) -> np.ndarray:
    """Outcome frequencies among the past `window` start days whose trades had resolved by the start
    date (Laplace-smoothed). Uses only information available at the start date's close."""
    sc = S.scan(full, mode="close")
    cls = class_index(sc.outcome.to_numpy())
    resolved_at = np.arange(len(sc)) + sc.tau.to_numpy()
    resolved_at = np.where(sc.complete.to_numpy(), resolved_at, np.iinfo(np.int64).max)
    pos = full.index.get_indexer(start_dates)
    out = np.empty((len(pos), 3))
    for i, s in enumerate(pos):
        lo = max(0, s - window)
        ok = resolved_at[lo:s] <= s
        counts = np.bincount(cls[lo:s][ok], minlength=3) + 1.0
        out[i] = counts / counts.sum()
    return out


def run(ds) -> dict:
    full, test, grid, preds = load_grid_and_predictions(ds)
    ov = S.overlapping(S.scan(test, mode="close"))
    n = len(ov)
    pos = grid.index_for(ov.start)
    y = class_index(ov.outcome.to_numpy())
    tau = ov.tau.to_numpy()
    print(f"\n--- Q3 ({ds.label}): {n} start days {ov.start.iloc[0].date()} -> {ov.start.iloc[-1].date()}, "
          f"{len(grid.dates)} monthly fits")

    probs = {m: np.array([[preds[m][k].p_win, preds[m][k].p_ruin, preds[m][k].p_timeout]
                          for k in range(len(grid.fits))])[pos] for m in MODELS}
    probs[BASELINE] = baseline_probs(full, ov.start)
    onehot = np.eye(3)[y]
    scores = {m: {"log": np.log(np.clip(p[np.arange(n), y], 1e-12, None)),
                  "brier": ((p - onehot) ** 2).sum(1)} for m, p in probs.items()}

    rng = rng_for(ds.key, "q3")
    idx = block_indices(n, C.BOOT_BLOCK_STARTS, C.BOOT_REPS, rng)
    ci = lambda v: [float(np.percentile(v[idx].mean(1), 2.5)), float(np.percentile(v[idx].mean(1), 97.5))]

    out = {"dataset": ds.key, "n_starts": n, "first_start": str(ov.start.iloc[0].date()),
           "last_start": str(ov.start.iloc[-1].date()), "block": C.BOOT_BLOCK_STARTS}
    real_freq = onehot.mean(0)
    out["real"] = {"p_win": float(real_freq[0]), "p_ruin": float(real_freq[1]),
                   "p_timeout": float(real_freq[2]), "p_win_ci": ci(onehot[:, 0]),
                   "tau_mean": float(tau.mean()), "tau_mean_ci": ci(tau.astype(float)),
                   "tau_median": float(np.median(tau))}

    # ---------------- outcome and duration forecasts
    out["models"] = {}
    for m in list(MODELS) + [BASELINE]:
        p = probs[m]
        rec = {"pred_p_win": float(p[:, 0].mean()), "pred_p_ruin": float(p[:, 1].mean()),
               "pred_p_timeout": float(p[:, 2].mean()),
               "log_score": float(scores[m]["log"].mean()), "log_score_ci": ci(scores[m]["log"]),
               "brier": float(scores[m]["brier"].mean()), "brier_ci": ci(scores[m]["brier"])}
        for ref in ("GBM", BASELINE):
            if ref != m:
                d_log = scores[m]["log"] - scores[ref]["log"]
                d_br = scores[m]["brier"] - scores[ref]["brier"]
                rec[f"vs {ref}"] = {"log_diff": float(d_log.mean()), "log_diff_ci": ci(d_log),
                                    "brier_diff": float(d_br.mean()), "brier_diff_ci": ci(d_br)}
        out["models"][m] = rec

    pit = {}
    for m in MODELS:
        pmf = np.array([preds[m][k].tau_pmf() for k in range(len(grid.fits))])[pos]   # (n, cap)
        cdf = np.cumsum(pmf, axis=1)
        before = np.where(tau > 1, cdf[np.arange(n), np.maximum(tau - 2, 0)], 0.0)
        u = before + rng.random(n) * pmf[np.arange(n), tau - 1]
        pit[m] = np.clip(u, 0, 1)
        hist = np.histogram(pit[m], bins=10, range=(0, 1))[0] / n
        boot = np.array([np.histogram(pit[m][ix], bins=10, range=(0, 1))[0] / n for ix in idx[:500]])
        out["models"][m]["tau_pred_mean"] = float((pmf * np.arange(1, C.CAP_DAYS + 1)).sum(1).mean())
        out["models"][m]["pit_hist"] = hist.tolist()
        out["models"][m]["pit_hist_ci"] = np.percentile(boot, [2.5, 97.5], axis=0).T.tolist()
        out["models"][m]["pit_bins_outside_ci"] = int(sum(not (lo <= 0.1 <= hi)
                                                         for lo, hi in out["models"][m]["pit_hist_ci"]))
        out["models"][m]["pit_mean"] = float(pit[m].mean())

    # ---------------- calibration of P(win)
    out["calibration"] = {}
    for m in list(MODELS) + [BASELINE]:
        pw = probs[m][:, 0]
        edges = np.unique(np.quantile(pw, np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(edges, pw, side="right") - 1, 0, len(edges) - 2)
        rows = []
        for j in range(len(edges) - 1):
            sel = b == j
            if sel.sum() < 20:
                continue
            boot = [onehot[ix, 0][sel[ix]].mean() for ix in idx[:300] if sel[ix].any()]
            rows.append({"pred": float(pw[sel].mean()), "real": float(onehot[sel, 0].mean()),
                         "n": int(sel.sum()), "ci": [float(np.percentile(boot, 2.5)),
                                                     float(np.percentile(boot, 97.5))]})
        out["calibration"][m] = rows

    # ---------------- stop fills (close-checked): real vs predicted
    real_fill = ov.ret.to_numpy()[ov.outcome.to_numpy() == RUIN]
    out["stop_fill"] = {"real": {"n": int(real_fill.size), "mean": float(real_fill.mean()),
                                 "p05": float(np.percentile(real_fill, 5)),
                                 "p01": float(np.percentile(real_fill, 1)),
                                 "min": float(real_fill.min()),
                                 "mean_ci": ci_ruin(ov, idx)},
                        "real_values": real_fill.tolist(), "models": {}}
    counts = np.bincount(pos, minlength=len(grid.fits))
    for m in MODELS:
        r_all, w_all = [], []
        for k in np.nonzero(counts)[0]:
            r, p = preds[m][k].ret_ruin()
            r_all.append(r)
            w_all.append(p * preds[m][k].p_ruin * counts[k])
        r_all, w_all = np.concatenate(r_all), np.concatenate(w_all)
        w_all /= w_all.sum()
        o = np.argsort(r_all)
        cw = np.cumsum(w_all[o])
        q = lambda a: float(r_all[o][np.searchsorted(cw, a)])
        out["stop_fill"]["models"][m] = {"mean": float((r_all * w_all).sum()), "p05": q(0.05),
                                         "p01": q(0.01)}
        out["stop_fill"]["models"][m]["_dist"] = (r_all[o], w_all[o])

    report(ds, out)
    plot_scores(ds, out)
    plot_calibration(ds, out)
    plot_pit(ds, out)
    plot_stopfill(ds, out)
    for m in MODELS:
        out["stop_fill"]["models"][m].pop("_dist")
    return out


def ci_ruin(ov, idx):
    ruin = (ov.outcome.to_numpy() == RUIN).astype(float)
    rr = np.nan_to_num(ov.ret.to_numpy()) * ruin
    with np.errstate(invalid="ignore"):
        m = rr[idx].sum(1) / ruin[idx].sum(1)
    return [float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))]


def report(ds, out) -> None:
    r = out["real"]
    print(f"real: win {r['p_win']:.1%} [{r['p_win_ci'][0]:.1%}, {r['p_win_ci'][1]:.1%}]  ruin "
          f"{r['p_ruin']:.1%}  timeout {r['p_timeout']:.1%}  mean days {r['tau_mean']:.0f} "
          f"[{r['tau_mean_ci'][0]:.0f}, {r['tau_mean_ci'][1]:.0f}]")
    print(f"{'forecaster':<34}{'P(win)':>8}{'days':>6}{'log score':>11}{'vs baseline [95% CI]':>28}"
          f"{'Brier':>8}{'PIT bins off':>13}")
    for m, s in out["models"].items():
        vb = s.get(f"vs {BASELINE}")
        vbs = f"{vb['log_diff']:+.3f} [{vb['log_diff_ci'][0]:+.3f}, {vb['log_diff_ci'][1]:+.3f}]" if vb else "-"
        days = f"{s['tau_pred_mean']:.0f}" if "tau_pred_mean" in s else "-"
        pit = f"{s['pit_bins_outside_ci']}/10" if "pit_bins_outside_ci" in s else "-"
        print(f"{m:<34}{s['pred_p_win']:>8.1%}{days:>6}{s['log_score']:>11.3f}{vbs:>28}"
              f"{s['brier']:>8.3f}{pit:>13}")
    for m in ("RandomWalk", "Merton"):
        v = out["models"][m]["vs GBM"]
        print(f"  {m} - GBM: log {v['log_diff']:+.4f} [{v['log_diff_ci'][0]:+.4f}, "
              f"{v['log_diff_ci'][1]:+.4f}]   Brier {v['brier_diff']:+.4f} "
              f"[{v['brier_diff_ci'][0]:+.4f}, {v['brier_diff_ci'][1]:+.4f}]")
    sf = out["stop_fill"]
    print(f"stop fill: real mean {sf['real']['mean']:+.2%} [{sf['real']['mean_ci'][0]:+.2%}, "
          f"{sf['real']['mean_ci'][1]:+.2%}], 1% {sf['real']['p01']:+.2%}, worst {sf['real']['min']:+.2%}"
          f" (n = {sf['real']['n']})")
    for m, v in sf["models"].items():
        print(f"  {m:<11} mean {v['mean']:+.2%}   5% {v['p05']:+.2%}   1% {v['p01']:+.2%}")


LABEL = {**P.MODEL_LABEL, BASELINE: "No model"}
COLOR = {**P.MODEL_COLOR, BASELINE: P.INK2}


def plot_scores(ds, out) -> None:
    fig, axes = P.new_fig(ncols=2, w=5.4, h=3.4)
    for ax, key, title in [(axes[0, 0], "log", "Log score vs no-model baseline (higher = better)"),
                           (axes[0, 1], "brier", "Brier score vs no-model baseline (lower = better)")]:
        for i, m in enumerate(MODELS):
            v = out["models"][m][f"vs {BASELINE}"]
            d, lo, hi = v[f"{key}_diff"], *v[f"{key}_diff_ci"]
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=COLOR[m], ms=7, capsize=4, lw=1.6)
        ax.axvline(0, color=P.INK2, lw=0.9)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([LABEL[m] for m in MODELS])
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        P.titles(ax, title, "Mean difference per start day (95% block-bootstrap CI)")
    P.save(fig, C.PLOTS_DIR / f"q3_scores_{ds.key}.png",
           f"Q3 - Barrier outcome forecasts vs no model, {ds.label}")


def plot_calibration(ds, out) -> None:
    fig, axes = P.new_fig(w=5.6, h=4.8)
    ax = axes[0, 0]
    # GBM and the random walk forecast almost exactly like Merton, so only Merton is drawn
    for m, off in (("Merton", -0.004), (BASELINE, 0.004)):
        rows = out["calibration"][m]
        x = np.array([r["pred"] for r in rows]) + off
        yv = [r["real"] for r in rows]
        err = [[r["real"] - r["ci"][0] for r in rows], [r["ci"][1] - r["real"] for r in rows]]
        ax.errorbar(x, yv, yerr=err, fmt="o-", color=COLOR[m], ms=6, lw=2, capsize=3,
                    elinewidth=1.2, label=f"{LABEL[m]} forecast", mec=P.SURFACE, mew=1.5)
    ax.plot([0, 1], [0, 1], color=P.INK2, lw=0.9, ls="--", label="perfect calibration")
    ax.text(0.98, 0.03, "GBM and random walk: same as Merton within 1 point",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=P.INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    P.pct_axis(ax.xaxis)
    P.pct_axis(ax.yaxis)
    P.titles(ax, "Predicted vs actual win rate (deciles of forecast)", "Predicted P(win)",
             "Actual share of wins")
    P.legend(ax, loc="upper left", fontsize=8)
    P.save(fig, C.PLOTS_DIR / f"q3_calibration_{ds.key}.png", f"Q3 - Calibration of P(win), {ds.label}")


def plot_pit(ds, out) -> None:
    fig, axes = P.new_fig(ncols=3, w=3.8, h=3.4, sharey=True)
    x = np.arange(10) * 0.1 + 0.05
    for ax, m in zip(axes[0], MODELS):
        s = out["models"][m]
        h = np.array(s["pit_hist"])
        lo, hi = np.array(s["pit_hist_ci"]).T
        ax.bar(x, h, width=0.085, color=COLOR[m], alpha=0.85)
        ax.errorbar(x, h, yerr=[h - lo, hi - h], fmt="none", ecolor=P.INK, lw=1, capsize=2)
        ax.axhline(0.1, color=P.INK2, lw=1, ls="--")
        ax.set_xlim(0, 1)
        P.titles(ax, f"{LABEL[m]} ({s['pit_bins_outside_ci']}/10 bins off)", "PIT of realised exit day",
                 "Share of start days" if m == MODELS[0] else None)
    P.save(fig, C.PLOTS_DIR / f"q3_pit_{ds.key}.png",
           f"Q3 - Exit-day forecasts (flat at 10% if right; left = exits earlier than predicted), {ds.label}")


def plot_stopfill(ds, out) -> None:
    fig, axes = P.new_fig(w=7.2, h=4.2)
    ax = axes[0, 0]
    sf = out["stop_fill"]
    bins = np.linspace(-0.20, -0.095, 43)
    ax.hist(np.clip(sf["real_values"], bins[0], None), bins=bins, density=True, color=P.GRID,
            edgecolor=P.INK2, lw=0.5, label=f"real (mean {sf['real']['mean']:+.1%})")
    for m in MODELS:
        r, w = sf["models"][m]["_dist"]
        dens = np.histogram(np.clip(r, bins[0], None), bins=bins, weights=w)[0] / np.diff(bins)
        ax.step(bins[:-1], dens, where="post", color=COLOR[m], lw=1.8,
                label=f"{LABEL[m]} (mean {sf['models'][m]['mean']:+.1%})")
    P.pct_axis(ax.xaxis)
    P.titles(ax, "Loss on stopped trades, daily close checks (left edge collects <= -20%)",
             "Return at exit", "Density")
    P.legend(ax, loc="upper left", fontsize=8)
    P.save(fig, C.PLOTS_DIR / f"q3_stopfill_{ds.key}.png", f"Q3 - Stop-loss fills, {ds.label}")


def main() -> None:
    for ds in C.DATASETS.values():
        out = run(ds)
        path = C.RESULTS_DIR / f"q3_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
