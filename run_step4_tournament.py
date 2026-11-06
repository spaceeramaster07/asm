"""Step 4: betting tournament (Q5, Q6, Q7).

Run:  .venv/Scripts/python run_step4_tournament.py   (needs run_engine_a.py output)
Writes results/tournament_<dataset>.json and plots.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import engine as E
from barrier import plotstyle as P
from barrier import scanner as S
from barrier import theory as T
from barrier import tournament as TN
from barrier.models import GBMModel, calibrate_all

WORLDS = ("RandomWalk", "GBM", "Merton")
N_Q7_CAREERS = 1_000
N_Q7_BELIEF_EPISODES = 20_000


def rng_for(*key):
    return np.random.default_rng([C.SEED, 4] + [sum(map(ord, str(k))) for k in key])


def episodes_RT(ds, params, model):
    ep = E.apply_cap(E.load(C.RESULTS_DIR / "sims" / f"{ds.key}_{params}_{model}.npz"), C.CAP_DAYS)
    return np.expm1(ep.exit_log), ep.tau.astype(float)


# ------------------------------------------------------------------ Q5
def q5(ds, models_train) -> dict:
    fs = np.round(np.arange(0, 3.5001, 0.01), 4)
    curves, out = {}, {}
    sets = [(f"{m} (training)", "train", m) for m in WORLDS] + [("GBM (test-period, hindsight)", "oracle", "GBM")]
    for label, p, m in sets:
        R, Tt = episodes_RT(ds, p, m)
        g = TN.growth_curve(R, Tt, ds.rf, fs)
        fstar = TN.kelly_fraction(R, Tt, ds.rf)
        gstar = TN.growth_curve(R, Tt, ds.rf, np.array([fstar, fstar / 2, 2 * fstar]))
        g0 = g[0]                                  # f = 0: all cash
        # With r_f > 0 the meaningful break-even is "no better than cash", not "zero growth".
        fw = np.round(np.arange(0, C.Q7_UNCAPPED_MAX + 0.001, 0.01), 4)   # wider grid
        gw = TN.growth_curve(R, Tt, ds.rf, fw)
        pos = fw[(fw > fstar) & (gw <= g0)]
        f0 = float(pos[0]) if pos.size else None
        yr = C.TRADING_DAYS / Tt.mean()
        curves[label] = g
        out[label] = {
            "f_star": fstar, "f_cash_breakeven": f0,
            "breakeven_over_f_star": (f0 / fstar) if (f0 and fstar > 0) else None,
            "g_cash_per_year": float(g0 * yr),
            "g_star_per_episode": float(gstar[0]),
            "g_star_per_year": float(gstar[0] * yr),
            "excess_g_star_per_year": float((gstar[0] - g0) * yr),
            "half_kelly_share_of_excess": float((gstar[1] - g0) / (gstar[0] - g0))
            if gstar[0] > g0 else None,
            "g_at_2fstar_per_year": float(gstar[2] * yr),
            "mean_days": float(Tt.mean()),
            "worst_1pct_return": float(np.percentile(R, 1)),
        }
    # textbook two-point formula: GBM daily win prob, exact +20/-10, no cash, no overshoot
    p_bgk = models_train["GBM"].theory()["p_win_daily_bgk"]
    g2 = TN.two_point_growth(p_bgk, fs)
    f2 = p_bgk / 0.10 - (1 - p_bgk) / 0.20
    curves["Textbook two-point (GBM p, no r_f, no overshoot)"] = g2
    out["textbook"] = {"p": p_bgk, "f_star": f2}

    print(f"\n--- Q5 Kelly ({ds.label}, r_f = {ds.rf:.1%})")
    print(f"{'world':<32}{'f*':>6}{'=cash at':>9}{'ratio':>7}{'cash/yr':>9}{'g*/yr':>8}"
          f"{'excess':>8}{'half/full':>10}{'g(2f*)/yr':>11}{'worst1%':>9}")
    for k, v in out.items():
        if k == "textbook":
            print(f"{'textbook two-point formula':<32}{v['f_star']:>6.2f}   (p = {v['p']:.3f})")
            continue
        f0 = f"{v['f_cash_breakeven']:.2f}" if v["f_cash_breakeven"] else ">3.5"
        ratio = f"{v['breakeven_over_f_star']:.2f}" if v["breakeven_over_f_star"] else "-"
        hk = f"{v['half_kelly_share_of_excess']:.0%}" if v["half_kelly_share_of_excess"] else "-"
        print(f"{k:<32}{v['f_star']:>6.2f}{f0:>9}{ratio:>7}{v['g_cash_per_year']:>+9.2%}"
              f"{v['g_star_per_year']:>+8.2%}{v['excess_g_star_per_year']:>+8.2%}"
              f"{hk:>10}{v['g_at_2fstar_per_year']:>+11.2%}{v['worst_1pct_return']:>+9.1%}")
    plot_q5(ds, fs, curves, out)
    return out


def plot_q5(ds, fs, curves, out):
    fig, axes = P.new_fig(w=7.4, h=4.6)
    ax = axes[0, 0]
    cols = [P.CAT[0], P.CAT[1], P.CAT[2], P.INK2, P.INK2]
    styles = ["-", "-", "-", "-", ":"]
    for (lab, g), col, ls in zip(curves.items(), cols, styles):
        gg = np.where(np.isfinite(g), g, np.nan)
        ax.plot(fs, gg, color=col, lw=2 if ls == "-" else 1.5, ls=ls,
                label=lab.replace("RandomWalk", "Random walk"))
        info = out.get(lab)
        if info and info["f_star"] <= 3.5:
            ax.plot(info["f_star"], np.interp(info["f_star"], fs, gg), "o", color=col, ms=7,
                    mec=P.SURFACE, mew=1.5)
    ax.axhline(0, color=P.INK, lw=0.8)
    lo = np.nanmin([np.nanmin(np.where(np.isfinite(g), g, np.nan)) for g in curves.values()])
    hi = np.nanmax([np.nanmax(g) for g in curves.values()])
    ax.set_ylim(max(lo, -0.6 * abs(hi) - 0.01), hi * 1.15 + 0.002)
    P.pct_axis(ax.yaxis, 1)
    P.titles(ax, f"Expected log growth per episode vs fraction bet (r_f = {ds.rf:.1%}); dots = Kelly f*",
             "Fraction of wealth bet per episode, f (1 = no leverage)", "E[log growth] per episode")
    P.legend(ax, loc="upper left", handlelength=3)
    P.save(fig, C.PLOTS_DIR / f"q5_kelly_{ds.key}.png", f"Q5 - Kelly growth curve, {ds.label}")


# ------------------------------------------------------------------ Q6
def q6(ds, models_train, test, f_trader) -> dict:
    strategies = TN.standard_strategies(f_trader)
    close = test.Close.to_numpy()
    sc = S.scan(test, mode="close")
    chains = [S.chain(sc, o) for o in range(C.CHAIN_OFFSETS)]
    K_real = len(chains[0])

    out = {"f_trader": f_trader, "K_real": K_real, "worlds": {}, "hist": {}}
    for K in (K_real, C.K_LONG):
        for w in WORLDS:
            t0 = time.time()
            book = TN.simulate_careers(models_train[w], strategies, C.N_CAREERS, K, ds.rf,
                                       rng_for(ds.key, "q6", w, K))
            out["worlds"][f"{w}/K{K}"] = TN.metrics(book)
            if K == K_real:
                out["hist"][w] = {st.name: np.percentile(book.hist[i], [10, 50, 90], axis=0).tolist()
                                  for i, st in enumerate(strategies)}
                out["worlds"][f"{w}/K{K}"]["_terminal"] = {st.name: book.W[i]
                                                           for i, st in enumerate(strategies)}
            print(f"  simulated {w:<10} K={K:<4} in {time.time()-t0:.1f}s")

    # real careers: one per chain start offset
    real = []
    for ch in chains:
        bk = TN.real_career(close, ch, strategies, ds.rf)
        m = TN.metrics(bk)
        bh = TN.buy_and_hold_real(close, int(ch.entry_idx.iloc[0]), int(ch.exit_idx.iloc[-1]))
        m["Buy & Hold"].update({"W_median": bh["W"], "mdd_median": bh["mdd"],
                                "cagr_median": (bh["W"] / C.W0) ** (C.TRADING_DAYS / bh["days"]) - 1})
        real.append((m, bk))
    m0, bk0 = real[0]
    bh_path = [C.W0] + [C.W0 * close[e] / close[int(chains[0].entry_idx.iloc[0])]
                        for e in chains[0].exit_idx]
    out["real"] = {
        "offset0": m0,
        "hist0": {st.name: (bk0.hist[i, 0].tolist() if st.name != "Buy & Hold" else bh_path)
                  for i, st in enumerate(strategies)},
        "W_range_offsets": {st.name: [float(np.min([r[0][st.name]["W_median"] for r in real])),
                                      float(np.max([r[0][st.name]["W_median"] for r in real]))]
                            for st in strategies},
    }
    # where the real trader lands among simulated careers (same K)
    out["real_percentile"] = {
        w: {st.name: float(np.mean(out["worlds"][f"{w}/K{K_real}"]["_terminal"][st.name]
                                   <= m0[st.name]["W_median"]) * 100) for st in strategies}
        for w in WORLDS}
    for w in WORLDS:
        out["worlds"][f"{w}/K{K_real}"].pop("_terminal")

    print_q6(ds, out, strategies)
    plot_q6(ds, out, strategies)
    return out


def print_q6(ds, out, strategies):
    K_real = out["K_real"]
    print(f"\n--- Q6 tournament ({ds.label}); trader's Kelly f* = {out['f_trader']:.2f} "
          f"(training GBM, r_f {ds.rf:.1%}); careers = {C.N_CAREERS}")
    hdr = f"{'strategy':<12}{'f':>5}{'W med':>9}{'W p5':>8}{'W p95':>9}{'CAGR':>8}{'MDD med':>9}" \
          f"{'P(MDD>50%)':>11}{'P(W<50)':>9}{'P(bust)':>8}"
    for K in (K_real, C.K_LONG):
        for w in WORLDS:
            print(f"\n[{w} world, K = {K} rounds, ~{out['worlds'][f'{w}/K{K}']['Buy & Hold']['years_median']:.1f} yrs]")
            print(hdr)
            for name, m in out["worlds"][f"{w}/K{K}"].items():
                f = "-" if m["f"] is None else f"{m['f']:.2f}"
                print(f"{name:<12}{f:>5}{m['W_median']:>9.1f}{m['W_p05']:>8.1f}{m['W_p95']:>9.1f}"
                      f"{m['cagr_median']:>+8.1%}{m['mdd_median']:>9.1%}{m['p_mdd_gt_50']:>11.1%}"
                      f"{m['p_lose_half']:>9.1%}{m['p_bankrupt']:>8.1%}")
    print(f"\n[REAL {ds.label}, chain from {ds.test[0]}: K = {K_real} rounds]   "
          f"+ range of final wealth over {C.CHAIN_OFFSETS} chain start days + percentile among "
          f"simulated careers (RW / GBM / Merton)")
    for name, m in out["real"]["offset0"].items():
        lo, hi = out["real"]["W_range_offsets"][name]
        pc = " / ".join(f"{out['real_percentile'][w][name]:.0f}" for w in WORLDS)
        print(f"{name:<12}  W {m['W_median']:>7.1f}  (range {lo:6.1f}-{hi:6.1f})  CAGR "
              f"{m['cagr_median']:>+6.1%}  MDD {m['mdd_median']:>6.1%}   pct {pc}")


def strategy_color(i, name):
    # Buy & Hold is the benchmark: neutral ink instead of a 6th hue (green vs aqua clashed).
    return P.INK if name == "Buy & Hold" else P.CAT[i]


def plot_q6(ds, out, strategies):
    fig, axes = P.new_fig(ncols=4, w=4.1, h=4.3, sharey=True)
    K = out["K_real"]
    x = np.arange(K + 1)
    for ax, w in zip(axes[0, :3], WORLDS):
        for i, st in enumerate(strategies):
            col = strategy_color(i, st.name)
            p10, p50, p90 = (np.array(v) for v in out["hist"][w][st.name])
            if st.name in ("Full Kelly", "Half Kelly"):
                ax.fill_between(x, p10, p90, color=col, alpha=0.12, lw=0)
            ax.plot(x, p50, color=col, lw=2, ls="--" if st.name == "Buy & Hold" else "-",
                    label=st.name.replace("Rs", "Rs "))
        P.titles(ax, f"{P.MODEL_LABEL[w]} world (median)", "Round",
                 "Wealth (start = 100, log scale)" if w == WORLDS[0] else None)
    ax = axes[0, 3]
    for i, st in enumerate(strategies):
        ax.plot(x, out["real"]["hist0"][st.name], color=strategy_color(i, st.name), lw=2,
                ls="--" if st.name == "Buy & Hold" else "-", marker="o", ms=3.5)
    P.titles(ax, f"Real {ds.label}: one trader, {ds.test[0][:4]}-{ds.test[1][:4]}", "Round")
    for a in axes[0]:
        a.set_yscale("log")
        a.yaxis.set_major_locator(P.tk.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.yaxis.set_major_formatter(P.tk.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.yaxis.set_minor_formatter(P.tk.NullFormatter())
        a.axhline(C.W0, color=P.INK2, lw=0.8, ls=":")
    P.legend(axes[0, 0], loc="upper left")
    P.save(fig, C.PLOTS_DIR / f"q6_tournament_{ds.key}.png",
           f"Q6 - Six strategies over {K} rounds (shaded: 10-90% of careers for Full/Half Kelly), {ds.label}")


# ------------------------------------------------------------------ Q7
def q7(ds, models_train) -> dict:
    g = models_train["GBM"]
    mu_true, sig = g.mu, g.sigma
    rf = ds.rf
    fs_eval = {}
    beliefs = {}
    for c in C.Q7_MULTIPLIERS:
        believed = GBMModel(nu=c * mu_true - 0.5 * sig ** 2, sigma=sig, n_obs=g.n_obs)
        ep = E.apply_cap(E.simulate(believed, N_Q7_BELIEF_EPISODES, rng_for(ds.key, "q7b", c)),
                         C.CAP_DAYS)
        f = TN.kelly_fraction(np.expm1(ep.exit_log), ep.tau.astype(float), rf)
        beliefs[c] = f
        fs_eval[c] = {"Full Kelly (3x cap)": min(f, C.LEVERAGE_CAP),
                      "Half Kelly (3x cap)": min(f / 2, C.LEVERAGE_CAP),
                      "Full Kelly (no cap)": min(f, C.Q7_UNCAPPED_MAX),
                      "Half Kelly (no cap)": min(f / 2, C.Q7_UNCAPPED_MAX)}
    uniq = sorted({v for d in fs_eval.values() for v in d.values()})
    strategies = [TN.Strategy(f"f={v:.3f}", "fraction", v, cap=C.Q7_UNCAPPED_MAX) for v in uniq]

    out = {"mu_true": mu_true, "sigma": sig, "f_believed": beliefs, "worlds": {}}
    print(f"\n--- Q7 drift misestimation ({ds.label}): true mu = {mu_true:.2%}, sigma = {sig:.2%}, "
          f"r_f = {rf:.1%}; careers {N_Q7_CAREERS} x {C.K_LONG} rounds")
    for w in ("GBM", "Merton"):
        R, Tt = episodes_RT(ds, "train", w)
        g_true = TN.growth_curve(R, Tt, rf, np.array(uniq)) / Tt.mean() * C.TRADING_DAYS
        f_true = TN.kelly_fraction(R, Tt, rf)
        book = TN.simulate_careers(models_train[w], strategies, N_Q7_CAREERS, C.K_LONG, rf,
                                   rng_for(ds.key, "q7", w))
        met = TN.metrics(book)
        rows = {}
        for c in C.Q7_MULTIPLIERS:
            rows[c] = {}
            for lab, f in fs_eval[c].items():
                m = met[f"f={f:.3f}"]
                rows[c][lab] = {"f": f, "growth_per_year": float(g_true[uniq.index(f)]),
                                "W_median": m["W_median"], "p_lose_half": m["p_lose_half"],
                                "p_bankrupt": m["p_bankrupt"], "mdd_median": m["mdd_median"]}
        out["worlds"][w] = {"f_true": f_true, "rows": rows}

        print(f"\n[true world = {w}; true Kelly f* = {f_true:.2f}]")
        print(f"{'c':>5}{'believed f*':>12} | " + " | ".join(
            f"{lab:<20}" for lab in fs_eval[1.0]))
        print(f"{'':>17} | " + " | ".join(f"{'f':>5}{'g/yr':>8}{'P(W<50)':>8}" for _ in fs_eval[1.0]))
        for c in C.Q7_MULTIPLIERS:
            print(f"{c:>5.2f}{beliefs[c]:>12.2f} | " + " | ".join(
                f"{r['f']:>5.2f}{r['growth_per_year']:>+8.1%}{r['p_lose_half']:>8.1%}"
                for r in rows[c].values()))
    plot_q7(ds, out)
    return out


def plot_q7(ds, out):
    fig, axes = P.new_fig(ncols=2, nrows=2, w=5.4, h=3.7)
    cs = np.array(C.Q7_MULTIPLIERS)
    style = {"Full Kelly (3x cap)": (P.CAT[1], "-"), "Half Kelly (3x cap)": (P.CAT[0], "-"),
             "Full Kelly (no cap)": (P.CAT[1], "--"), "Half Kelly (no cap)": (P.CAT[0], "--")}
    for row, w in enumerate(("GBM", "Merton")):
        rows = {float(k): v for k, v in out["worlds"][w]["rows"].items()}  # JSON keys are str
        for lab, (col, ls) in style.items():
            g = [rows[c][lab]["growth_per_year"] for c in C.Q7_MULTIPLIERS]
            pl = [rows[c][lab]["p_lose_half"] for c in C.Q7_MULTIPLIERS]
            busted = ~np.isfinite(g)
            g = np.where(busted, np.nan, g)
            axes[row, 0].plot(cs, g, color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
            axes[row, 1].plot(cs, pl, color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
            if busted.any():
                c0 = cs[busted][0]
                axes[row, 0].text(
                    cs[-1], 0.003, f"{lab}, c >= {c0:g}:\ngrowth = -inf (one jump\ncan wipe the trader out)",
                    color=P.INK2, fontsize=7.5, ha="right", va="bottom")
        for ax in axes[row]:
            ax.axvline(1.0, color=P.INK2, lw=0.8, ls=":")
        axes[row, 0].axhline(0, color=P.INK, lw=0.8)
        axes[row, 0].yaxis.set_major_locator(P.tk.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        P.pct_axis(axes[row, 0].yaxis, 0)
        P.pct_axis(axes[row, 1].yaxis, 0)
        xl = "Trader's drift estimate / true drift (c)" if row == 1 else None
        P.titles(axes[row, 0], f"{P.MODEL_LABEL[w]} world: expected growth per year", xl,
                 "Log growth / year")
        P.titles(axes[row, 1], f"{P.MODEL_LABEL[w]} world: P(lose half) over {C.K_LONG} rounds", xl)
    P.legend(axes[0, 1], loc="upper left", handlelength=3.5)
    P.save(fig, C.PLOTS_DIR / f"q7_drift_error_{ds.key}.png",
           f"Q7 - Betting Kelly on a misestimated drift, {ds.label}")


def main() -> None:
    """No arguments: run Q5, Q6, Q7. Or pass parts, e.g. `q5 plots` to recompute Q5 and
    redraw the Q6/Q7 plots from the saved JSON without re-simulating (~4 min saved)."""
    parts = set(sys.argv[1:]) or {"q5", "q6", "q7"}
    for ds in C.DATASETS.values():
        t0 = time.time()
        print("\n" + "#" * 100 + f"\n#  {ds.label}\n" + "#" * 100)
        _, train, test, _ = D.load(ds)
        models_train = calibrate_all(D.log_returns(train))
        path = C.RESULTS_DIR / f"tournament_{ds.key}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        res.update({"dataset": ds.key, "rf": ds.rf})
        if "q5" in parts:
            res["q5"] = q5(ds, models_train)
        f_trader = min(res["q5"]["GBM (training)"]["f_star"], C.LEVERAGE_CAP)
        if "q6" in parts:
            res["q6"] = q6(ds, models_train, test, f_trader)
        elif "plots" in parts:
            plot_q6(ds, res["q6"], TN.standard_strategies(res["q6"]["f_trader"]))
        if "q7" in parts:
            res["q7"] = q7(ds, models_train)
        elif "plots" in parts:
            plot_q7(ds, res["q7"])
        path.write_text(json.dumps(res, indent=2, default=lambda o: o.tolist()
                                   if hasattr(o, "tolist") else str(o)))
        print(f"\nsaved {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
