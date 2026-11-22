"""Q5: which betting strategy wins, and what does mis-estimating the parameters cost?

Worlds (all strategies play the same paths in each world):
  RandomWalk / GBM / Merton : true parameters = fit on the full history; nobody in the game sees them
  Bootstrap (headline)      : stationary block bootstrap of the real daily returns
  Real                      : the actual test period, one career per start offset (0..59 days)
Every career starts with 5 years of visible history; the trader re-estimates before every trade from
the path's own past only (rolling 5-year window), using the Kelly tables built by run_q4_kelly.py.

Run:  .venv/Scripts/python run_q5_tournament.py   (after run_q4_kelly.py)
Writes results/q5_<dataset>.json and plots q5_leaderboard / q5_worlds / q5_frontier / q5_misestimation.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import kelly as K
from barrier import plotstyle as P
from barrier import scanner as S
from barrier import theory as T
from barrier import tournament as TN
from barrier.bootstrap import stationary_bootstrap
from barrier.models import MertonModel, calibrate_all
from barrier.rng import rng_for

WORLDS = ("RandomWalk", "GBM", "Merton", "Bootstrap")
WORLD_LABEL = {"RandomWalk": "Random-walk world", "GBM": "GBM world", "Merton": "Merton world",
               "Bootstrap": "Bootstrap world (real returns)", "Real": "Real path"}
# model worlds wear their model's color; the two real-data views are ink (filled dot / cross)
WORLD_COLOR = {"RandomWalk": P.MODEL_COLOR["RandomWalk"], "GBM": P.MODEL_COLOR["GBM"],
               "Merton": P.MODEL_COLOR["Merton"], "Bootstrap": P.INK, "Real": P.INK}
BATCH = 200
PATH_LEN = C.ROLLING_DAYS + C.K_ROUNDS * 320


def load_table(ds):
    _, train, _, _ = D.load(ds)
    m = MertonModel.calibrate(D.log_returns(train))
    return K.load_or_build_table(ds, m.mu_J, m.sigma_J)


def real_chain_length(ds) -> int:
    _, _, test, _ = D.load(ds)
    return len(S.chain(S.scan(test, mode="close"), 0))


def concat(results):
    f = lambda attr, ax: np.concatenate([getattr(r, attr) for r in results], axis=ax)
    return TN.CareerResult(results[0].names, f("W", 1), f("mdd", 1), f("bust", 1), f("days", 0),
                           f("bh_W", 0), f("bh_mdd", 0), f("f_gbm_raw", 0), f("rounds", 0))


def misestimation(true_model, res: TN.CareerResult, rf: float) -> dict:
    """Trader's GBM Kelly vs the true world's Kelly, and the growth it costs."""
    o = K.compress(T.exit_distribution(true_model))
    f_true = K.kelly_fraction(o, rf)
    fs = np.round(np.arange(0, C.F_GRID_MAX + 1e-9, 0.02), 4)
    curve = K.growth_curve(o, fs, rf)
    g_star = K.growth_per_year(o, f_true, rf)
    f_raw = res.f_gbm_raw[np.isfinite(res.f_gbm_raw)]
    out = {"f_true": f_true, "g_true_star": g_star, "fs": fs.tolist(),
           "g_true_curve": [float(v) if np.isfinite(v) else None for v in curve],
           "ruin_cliff": K.ruin_cliff(T.exit_distribution(true_model)),
           "ratio_pct": {str(q): float(np.percentile(f_raw / max(f_true, 1e-9), q))
                         for q in (5, 10, 25, 50, 75, 90, 95)},
           "f_used_hist": np.histogram(np.minimum(f_raw, C.F_GRID_MAX), bins=60,
                                       range=(0, C.F_GRID_MAX))[0].tolist()}
    sample = np.random.default_rng(0).choice(f_raw, min(4000, f_raw.size), replace=False)
    for lab, mult in (("full", 1.0), ("half", 0.5)):
        f_use = np.minimum(mult * sample, C.LEVERAGE_CAP)
        g = np.array([K.growth_per_year(o, f, rf) for f in f_use])
        out[f"growth_loss_{lab}"] = float(g_star - np.mean(g))
        out[f"p_overbet_{lab}"] = float(np.mean(f_use > f_true))
    return out


def run_world(args) -> dict:
    ds_key, world, k_real = args
    ds = C.DATASETS[ds_key]
    t0 = time.time()
    full, _, _, _ = D.load(ds)
    r_full = D.log_returns(full).to_numpy()
    true = calibrate_all(D.log_returns(full))
    table = load_table(ds)
    strategies = TN.standard_strategies()
    parts = []
    for b0 in range(0, C.N_CAREERS, BATCH):
        rng = rng_for(ds.key, "q5", world, b0)
        n = min(BATCH, C.N_CAREERS - b0)
        if world == "Bootstrap":
            r = stationary_bootstrap(r_full, n, PATH_LEN, C.BOOT_WORLD_MEAN_BLOCK, rng)
        else:
            r = true[world].increments(rng, (n, PATH_LEN))
        parts.append(TN.run_careers(r, strategies, table, ds.rf, C.K_ROUNDS))
    res = concat(parts)
    out = {"world": world, "n_careers": int(len(res.rounds)),
           "share_reached_K": float(np.mean(res.rounds >= C.K_ROUNDS)), "metrics": {}}
    for k in sorted(set(C.K_REPORT) | {k_real}):
        out["metrics"][f"K{k}"] = TN.metrics_at(res, k)
    # terminal CAGR samples at the real chain length (for the real path's percentile)
    yrs = np.maximum(res.days[:, k_real], 1) / C.TRADING_DAYS
    out["cagr_at_k_real"] = {n: ((np.maximum(res.W[i, :, k_real], 1e-12) / C.W0) ** (1 / yrs) - 1).tolist()
                             for i, n in enumerate(res.names)}
    if world != "Bootstrap":
        out["misestimation"] = misestimation(true[world], res, ds.rf)
    out["seconds"] = time.time() - t0
    return out


def run_real(ds) -> dict:
    full, _, test, _ = D.load(ds)
    r_ser = D.log_returns(full)
    t0 = int(r_ser.index.get_indexer([test.index[0]])[0])
    r = r_ser.to_numpy()[t0 - C.ROLLING_DAYS:]
    offsets = np.arange(C.CHAIN_OFFSETS)
    rr = np.repeat(r[None, :], len(offsets), 0)
    res = TN.run_careers(rr, TN.standard_strategies(), load_table(ds), ds.rf, C.K_ROUNDS,
                         entry0=C.ROLLING_DAYS + offsets)
    out = {"rounds_offset0": int(res.rounds[0]),
           "rounds_range": [int(res.rounds.min()), int(res.rounds.max())], "strategies": {}}
    names = res.names + ["Buy & Hold"]
    for j, name in enumerate(names):
        cagr, Wf, md = [], [], []
        for b in range(len(offsets)):
            k = int(res.rounds[b])
            W = res.bh_W[b, k] if name == "Buy & Hold" else res.W[j, b, k]
            m = res.bh_mdd[b, k] if name == "Buy & Hold" else res.mdd[j, b, k]
            yrs = res.days[b, k] / C.TRADING_DAYS
            Wf.append(float(W))
            md.append(float(m))
            cagr.append(float((max(W, 1e-12) / C.W0) ** (1 / yrs) - 1))
        out["strategies"][name] = {"W_offset0": Wf[0], "cagr_offset0": cagr[0], "mdd_offset0": md[0],
                                   "cagr_range": [min(cagr), max(cagr)], "cagr_median": float(np.median(cagr)),
                                   "W_path0": [float(v) for v in (res.bh_W[0] if name == "Buy & Hold"
                                                                   else res.W[j, 0])
                                               [: int(res.rounds[0]) + 1]]}
    return out


def redraw() -> None:
    """Redraw the Q5 plots from the saved JSON (no simulation)."""
    for ds in C.DATASETS.values():
        out = json.loads((C.RESULTS_DIR / f"q5_{ds.key}.json").read_text())
        plot_leaderboard(ds, out)
        plot_worlds(ds, out)
        plot_frontier(ds, out)
        plot_misestimation(ds, out)


def main() -> None:
    import sys
    if "plots" in sys.argv[1:]:
        redraw()
        return
    k_real = {k: real_chain_length(ds) for k, ds in C.DATASETS.items()}
    for ds in C.DATASETS.values():
        load_table(ds)                      # build once, before the workers need it
    jobs = [(k, w, k_real[k]) for k in C.DATASETS for w in WORLDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(run_world, jobs))
    print(f"simulated {len(jobs)} worlds in {time.time() - t0:.0f}s "
          f"({', '.join(f'{j[0]}/{j[1]} {r['seconds']:.0f}s' for j, r in zip(jobs, res))})")
    for ds in C.DATASETS.values():
        out = {"dataset": ds.key, "rf": ds.rf, "k_real": k_real[ds.key], "reference": TN.REFERENCE,
               "strategies": [s.name for s in TN.standard_strategies()] + ["Buy & Hold"],
               "worlds": {j[1]: r for j, r in zip(jobs, res) if j[0] == ds.key},
               "real": run_real(ds)}
        boot = out["worlds"]["Bootstrap"]["cagr_at_k_real"]
        out["real_percentile_in_bootstrap"] = {
            n: float(np.mean(np.array(boot[n]) <= out["real"]["strategies"][n]["cagr_offset0"]) * 100)
            for n in boot}
        for w in out["worlds"].values():
            w.pop("cagr_at_k_real")
        report(ds, out)
        plot_leaderboard(ds, out)
        plot_worlds(ds, out)
        plot_frontier(ds, out)
        plot_misestimation(ds, out)
        path = C.RESULTS_DIR / f"q5_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=lambda o: None if o is None else str(o)))
        print(f"saved {path}")


def report(ds, out) -> None:
    print(f"\n=== Q5 {ds.label}: {C.N_CAREERS} careers x {C.K_ROUNDS} rounds per world; reference = "
          f"{out['reference']}; real chain = {out['k_real']} rounds")
    for w in WORLDS:
        wd = out["worlds"][w]
        m = wd["metrics"][f"K{C.K_ROUNDS}"]
        print(f"\n[{WORLD_LABEL[w]}, K = {C.K_ROUNDS}; reached K: {wd['share_reached_K']:.0%}]")
        print(f"{'strategy':<32}{'W med':>11}{'CAGR':>8}{'log g':>8}{'vs ref [95% CI]':>24}{'MDD':>6}"
              f"{'P(>50%)':>8}{'bust':>6}{'Sharpe':>7}")
        for name in sorted(m, key=lambda n: -m[n]["log_growth_mean"]):
            s = m[name]
            print(f"{name:<32}{s['W_median']:>11,.0f}{s['cagr_median']:>+8.1%}{s['log_growth_mean']:>+8.2%}"
                  f"{s['vs_ref_log_growth']:>+9.2%} [{s['vs_ref_ci'][0]:+.2%},{s['vs_ref_ci'][1]:+.2%}]"
                  f"{s['mdd_median']:>6.0%}{s['p_mdd_gt_50']:>8.0%}{s['p_bust']:>6.1%}"
                  f"{s['sharpe_median']:>7.2f}")
        if "misestimation" in wd:
            me = wd["misestimation"]
            print(f"  mis-estimation: true f* = {me['f_true']:.2f}; trader's f*/true median "
                  f"{me['ratio_pct']['50']:.2f} (10-90%: {me['ratio_pct']['10']:.2f}-{me['ratio_pct']['90']:.2f});"
                  f" growth lost {me['growth_loss_full']:.2%}/yr (Kelly), {me['growth_loss_half']:.2%}/yr "
                  f"(Half); ruin cliff {me['ruin_cliff']:.1f}x")
    rl = out["real"]
    print(f"\n[Real path: offset 0 = {rl['rounds_offset0']} rounds (range {rl['rounds_range']})]")
    for name, s in sorted(rl["strategies"].items(), key=lambda kv: -kv[1]["cagr_offset0"]):
        pc = out["real_percentile_in_bootstrap"].get(name)
        pcs = f"{pc:5.0f}" if pc is not None else "    -"
        print(f"{name:<32} CAGR {s['cagr_offset0']:+6.1%} (offsets {s['cagr_range'][0]:+.1%}..{s['cagr_range'][1]:+.1%})"
              f"  MDD {s['mdd_offset0']:4.0%}  percentile in bootstrap {pcs}")


def short(name):
    return name.replace("Drawdown-controlled", "DD-ctrl").replace("Vol-targeted", "Vol-tgt")


def plot_leaderboard(ds, out) -> None:
    m = out["worlds"]["Bootstrap"]["metrics"][f"K{C.K_ROUNDS}"]
    names = sorted(m, key=lambda n: m[n]["vs_ref_log_growth"])
    fig, axes = P.new_fig(ncols=2, w=5.8, h=0.34 * len(names) + 1.7)
    ax, ax2 = axes[0, 0], axes[0, 1]
    y = np.arange(len(names))
    xmax = max(m[n]["vs_ref_ci"][1] for n in names)
    for i, n in enumerate(names):
        s = m[n]
        lo, hi = s["vs_ref_ci"]
        d = s["vs_ref_log_growth"]
        is_ref = n == out["reference"]
        col = P.INK if is_ref else P.CAT[0]
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col, ms=8 if is_ref else 7,
                    capsize=3, lw=1.4, mec=P.SURFACE, mew=1.5)
        ax.text(xmax + 0.004, i, "reference" if is_ref else f"{d:+.2%}", va="center", fontsize=7.8,
                color=P.INK2)
        ax2.barh(i - 0.19, s["mdd_median"], height=0.34, color=P.CAT[0],
                 label="median max drawdown" if i == 0 else None)
        ax2.barh(i + 0.19, s["p_mdd_gt_50"], height=0.34, color=P.CAT[1],
                 label="P(drawdown > 50%)" if i == 0 else None)
    ax.set_xlim(None, xmax + 0.012)
    ax.axvline(0, color=P.INK2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([short(n) for n in names], fontsize=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    P.pct_axis(ax.xaxis, 1)
    P.pct_axis(ax2.xaxis, 0)
    ax2.set_xlim(0, 1)
    P.titles(ax, f"Log growth per year vs {out['reference']} (95% CI)", "Difference")
    P.titles(ax2, "Drawdown risk", "Share")
    P.legend(ax2, loc="lower right", fontsize=7.5)
    P.save(fig, C.PLOTS_DIR / f"q5_leaderboard_{ds.key}.png",
           f"Q5 - Bootstrap world (real returns), {C.N_CAREERS} careers x {C.K_ROUNDS} rounds, {ds.label}")


def plot_worlds(ds, out) -> None:
    ref = out["worlds"]["Bootstrap"]["metrics"][f"K{C.K_ROUNDS}"]
    names = sorted(ref, key=lambda n: ref[n]["log_growth_mean"])
    fig, axes = P.new_fig(w=8.4, h=0.32 * len(names) + 1.6)
    ax = axes[0, 0]
    for w in WORLDS:
        m = out["worlds"][w]["metrics"][f"K{C.K_ROUNDS}"]
        boot = w == "Bootstrap"
        ax.plot([m[n]["log_growth_mean"] for n in names], range(len(names)), "o", color=WORLD_COLOR[w],
                ms=8 if boot else 6, label=WORLD_LABEL[w], mec=P.SURFACE, mew=1.2, zorder=4 if boot else 3)
    rl = out["real"]["strategies"]
    ax.plot([np.log1p(rl[n]["cagr_offset0"]) for n in names], range(len(names)), "x", color=P.INK, ms=8,
            mew=1.8, label=f"real path ({out['real']['rounds_offset0']} rounds)", zorder=5)
    ax.axvline(np.log1p(ds.rf), color=P.INK2, lw=0.9, ls="--", label=f"cash ({ds.rf:.1%})")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short(n) for n in names], fontsize=8)
    lo = max(-0.15, min(ax.get_xlim()[0], -0.02))
    ax.set_xlim(lo, None)
    P.pct_axis(ax.xaxis, 0)
    P.titles(ax, "Mean log growth per year in each world (left edge clips busts)", "Log growth per year")
    P.legend(ax, loc="upper left", fontsize=7.5)
    P.save(fig, C.PLOTS_DIR / f"q5_worlds_{ds.key}.png", f"Q5 - Do the worlds agree on the ranking? {ds.label}")


def plot_frontier(ds, out) -> None:
    fig, axes = P.new_fig(w=7.0, h=4.8)
    ax = axes[0, 0]
    kelly = [s for s in TN.standard_strategies() if s.kind == "gbm_kelly"]
    for w in WORLDS:
        m = out["worlds"][w]["metrics"][f"K{C.K_ROUNDS}"]
        x = [m[s.name]["mdd_median"] for s in kelly]
        yv = [m[s.name]["log_growth_mean"] for s in kelly]
        ax.plot(x, yv, "-o", color=WORLD_COLOR[w], lw=1.6, ms=4, label=WORLD_LABEL[w])
        if w == "Bootstrap":
            for s, xi, yi in zip(kelly, x, yv):
                ax.annotate(f"{s.mult:.2f}x", (xi, yi), fontsize=7.5, color=P.INK2,
                            xytext=(5, 3), textcoords="offset points")
    ax.axhline(np.log1p(ds.rf), color=P.INK2, lw=0.9, ls="--", label="cash")
    P.pct_axis(ax.xaxis, 0)
    P.pct_axis(ax.yaxis, 1)
    P.titles(ax, "Fractional GBM Kelly: growth vs drawdown (0.25x to 2x f*)", "Median max drawdown",
             "Mean log growth per year")
    P.legend(ax, loc="lower right", fontsize=7.5)
    P.save(fig, C.PLOTS_DIR / f"q5_frontier_{ds.key}.png", f"Q5 - Kelly frontier, {ds.label}")


def plot_misestimation(ds, out) -> None:
    ws = [w for w in WORLDS if "misestimation" in out["worlds"][w]]
    fig = P.plt.figure(figsize=(4.6 * len(ws), 5.6), facecolor=P.SURFACE)
    gs = fig.add_gridspec(2, len(ws), height_ratios=[2.2, 1], hspace=0.12)
    for c, w in enumerate(ws):
        ax = fig.add_subplot(gs[0, c])
        axh = fig.add_subplot(gs[1, c], sharex=ax)
        for a in (ax, axh):
            P.style_ax(a)
        me = out["worlds"][w]["misestimation"]
        fs = np.array(me["fs"])
        g = np.array([np.nan if v is None else v for v in me["g_true_curve"]])
        col = WORLD_COLOR[w]
        ax.plot(fs, g, color=col, lw=2, label="true growth g(f)")
        i = int(np.nanargmax(g))
        ax.plot(fs[i], g[i], "o", color=col, ms=8, mec=P.SURFACE, mew=1.5)
        P.end_label(ax, fs[i], g[i], f"true f* = {me['f_true']:.2f}x", dy=9)
        ax.axhline(ds.rf, color=P.INK2, lw=0.9, ls="--", label=f"cash ({ds.rf:.1%})")
        for a in (ax, axh):
            a.axvline(C.LEVERAGE_CAP, color=P.INK2, lw=0.8, ls=":")
        ax.set_ylim(max(-0.10, np.nanmin(g) - 0.01), np.nanmax(g) + 0.025)
        P.pct_axis(ax.yaxis, 0)
        P.titles(ax, f"{WORLD_LABEL[w]}\nlost {me['growth_loss_full']:.1%}/yr at Kelly, "
                     f"{me['growth_loss_half']:.1%}/yr at Half Kelly", None,
                 "True growth per year" if c == 0 else None)
        P.legend(ax, loc="lower left", fontsize=7.5)
        ax.tick_params(labelbottom=False)
        h = np.array(me["f_used_hist"], float)
        edges = np.linspace(0, C.F_GRID_MAX, len(h) + 1)
        axh.bar(edges[:-1], h / h.sum(), width=np.diff(edges) * 0.85, align="edge", color=col)
        axh.axvline(me["f_true"], color=P.INK2, lw=0.9, ls="--")
        P.pct_axis(axh.yaxis, 0)
        P.titles(axh, "", "Fraction bet f (dotted: 3x cap)",
                 "Trader's f* estimates" if c == 0 else None)
        axh.set_xlim(0, C.F_GRID_MAX)
    P.save(fig, C.PLOTS_DIR / f"q5_misestimation_{ds.key}.png",
           f"Q5 - Cost of estimating f* from 5 years of data, {ds.label}")


if __name__ == "__main__":
    main()
