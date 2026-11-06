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
    fig, axes = P.new_fig(ncols=2, w=6.0, h=4.6)
    ax, ax_b = axes[0, 0], axes[0, 1]
    cols = [P.CAT[0], P.CAT[1], P.CAT[2], P.INK2, P.INK2]
    styles = ["-", "-", "-", "-", ":"]
    for (lab, g), col, ls in zip(curves.items(), cols, styles):
        gg = np.where(np.isfinite(g), g, np.nan)
        ax.plot(fs, gg, color=col, lw=2.1 if ls == "-" else 1.5, ls=ls,
                label=lab.replace("RandomWalk", "Random walk"))
        info = out.get(lab)
        if info and info["f_star"] <= 3.5:
            ax.plot(info["f_star"], np.interp(info["f_star"], fs, gg), "o", color=col, ms=7,
                    mec=P.SURFACE, mew=1.5, zorder=5)

    # Highlight GBM (training) cash baseline, Half Kelly, Full Kelly, and Cash Break-Even (~2f*)
    g_gbm = curves["GBM (training)"]
    info_gbm = out["GBM (training)"]
    g_cash = float(g_gbm[0])
    f_star = info_gbm["f_star"]
    f_be = info_gbm["f_cash_breakeven"]
    ax.axhline(g_cash, color=P.CAT[1], lw=1.1, ls="--",
               label=f"100% Cash baseline (r_f = {ds.rf:.1%}/yr)")
    if f_be and f_be <= 3.5:
        mask_ex = (fs <= f_be) & np.isfinite(g_gbm)
        ax.fill_between(fs[mask_ex], g_cash, g_gbm[mask_ex], color=P.CAT[1], alpha=0.10,
                        label="Excess growth over cash (GBM)")
        ax.plot(f_be, g_cash, "s", color=P.CAT[1], ms=6, mec=P.SURFACE, mew=1.2, zorder=5)
        ax.annotate(f"Cash break-even\nf = {f_be:.2f} ({f_be/f_star:.2f}x f*)",
                    xy=(f_be, g_cash), xytext=(min(f_be + 0.25, 2.65), g_cash + 0.015),
                    color=P.INK, fontsize=7.8,
                    arrowprops=dict(arrowstyle="->", color=P.CAT[1], lw=0.9))
    g_hk = float(np.interp(0.5 * f_star, fs, g_gbm))
    ax.plot(0.5 * f_star, g_hk, "^", color=P.CAT[1], ms=6.5, mec=P.SURFACE, mew=1.2, zorder=5)

    ax.axhline(0, color=P.INK, lw=0.8)
    lo = np.nanmin([np.nanmin(np.where(np.isfinite(g), g, np.nan)) for g in curves.values()])
    hi = np.nanmax([np.nanmax(g) for g in curves.values()])
    ax.set_ylim(max(lo, -0.6 * abs(hi) - 0.01), hi * 1.18 + 0.003)
    P.pct_axis(ax.yaxis, 1)
    P.titles(ax, f"Expected log growth per episode vs bet fraction (r_f = {ds.rf:.1%})",
             "Fraction of wealth bet per episode, f (1 = no leverage)", "E[log growth] per episode")
    P.legend(ax, loc="upper left", handlelength=2.6, fontsize=8.0)

    # Right panel: Grouped horizontal bar chart comparing Half Kelly, Full Kelly f*, and Cash Break-Even f0
    bar_Order = [
        ("Textbook 2-pt (no r_f)", "textbook", "#8c8a84"),
        ("Random walk (train)", "RandomWalk (training)", P.CAT[0]),
        ("GBM (train)", "GBM (training)", P.CAT[1]),
        ("Merton (train)", "Merton (training)", P.CAT[2]),
        ("GBM (test oracle)", "GBM (test-period, hindsight)", P.INK2),
    ][::-1]
    y_pos = np.arange(len(bar_Order))
    f_stars = [out[k]["f_star"] for _, k, _ in bar_Order]
    f_bes = [(out[k].get("f_cash_breakeven") or 0.0) for _, k, _ in bar_Order]

    h = 0.34
    ax_b.barh(y_pos + h / 2, f_stars, height=h, color=[c for _, _, c in bar_Order], alpha=0.90,
              label="Optimal Kelly f*")
    ax_b.barh(y_pos - h / 2, f_bes, height=h, color=[c for _, _, c in bar_Order], alpha=0.35,
              edgecolor=[c for _, _, c in bar_Order], lw=1.2, hatch="//",
              label="Cash break-even f0 (~2x f*)")
    for idx, (lab, k, col) in enumerate(bar_Order):
        fs_v = out[k]["f_star"]
        ax_b.text(fs_v + 0.06, idx + h / 2, f"f* = {fs_v:.2f}x", va="center", ha="left",
                  fontsize=8.0, color=P.INK, fontweight="bold")
        fb_v = out[k].get("f_cash_breakeven")
        if fb_v:
            ratio = out[k]["breakeven_over_f_star"]
            ax_b.text(fb_v + 0.06, idx - h / 2, f"{fb_v:.2f}x ({ratio:.2f}x f*)",
                      va="center", ha="left", fontsize=7.8, color=P.INK2)

    ax_b.axvline(1.0, color=P.INK2, lw=0.9, ls=":", label="f = 1.0 (unleveraged)")
    ax_b.axvline(C.LEVERAGE_CAP, color="#e34948", lw=1.0, ls="--", label=f"Tournament cap ({C.LEVERAGE_CAP:.0f}x)")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([b[0] for b in bar_Order], fontsize=8.5, color=P.INK)
    ax_b.set_xlim(0, max(max(f_stars), max(f_bes)) * 1.38 + 0.4)
    P.titles(ax_b, "Kelly f* vs cash break-even leverage (bar comparison)",
             "Fraction of wealth bet, f (leverage multiple)", None)
    ax_b.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor=P.SURFACE,
                edgecolor=P.GRID, framealpha=0.94, labelcolor=P.INK2)

    P.save(fig, C.PLOTS_DIR / f"q5_kelly_{ds.key}.png",
           f"Q5 - Kelly growth curve & cash break-even leverage, {ds.label}")


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


def strategy_label(ds, name: str) -> str:
    if name == "Flat Rs50":
        return "Flat Rs 50" if ds.key == "nifty" else "Flat $50"
    return name.replace("Rs", "Rs ")


def plot_q6(ds, out, strategies):
    fig, axes = P.new_fig(ncols=4, w=4.15, h=4.35, sharey=True)
    K = out["K_real"]
    x = np.arange(K + 1)
    for ax, w in zip(axes[0, :3], WORLDS):
        for i, st in enumerate(strategies):
            col = strategy_color(i, st.name)
            p10, p50, p90 = (np.array(v) for v in out["hist"][w][st.name])
            if st.name in ("Full Kelly", "Half Kelly"):
                ax.fill_between(x, p10, p90, color=col, alpha=0.12, lw=0)
            ax.plot(x, p50, color=col, lw=2, ls="--" if st.name == "Buy & Hold" else "-",
                    label=strategy_label(ds, st.name))
        P.titles(ax, f"{P.MODEL_LABEL[w]} world (median)", "Round",
                 "Wealth (start = 100, log scale)" if w == WORLDS[0] else None)
    ax = axes[0, 3]
    for i, st in enumerate(strategies):
        col = strategy_color(i, st.name)
        y_path = np.array(out["real"]["hist0"][st.name], float)
        ax.plot(x, y_path, color=col, lw=2,
                ls="--" if st.name == "Buy & Hold" else "-", marker="o", ms=3.5)
        lo_w, hi_w = out["real"]["W_range_offsets"][st.name]
        w_end = y_path[-1]
        x_off = K + 0.18 * (i - 2.5)
        ax.vlines(x_off, lo_w, hi_w, color=col, lw=1.6, alpha=0.75)
        ax.plot(x_off, w_end, "s", color=col, ms=3.2)

    # Annotate severe drawdown on S&P 500 Full Kelly
    fk_real = np.array(out["real"]["hist0"]["Full Kelly"], float)
    idx_min = int(np.argmin(fk_real))
    mdd_fk = out["real"]["offset0"]["Full Kelly"]["mdd_median"]
    if mdd_fk >= 0.70:
        ax.annotate(f"{mdd_fk:.0%} max drawdown",
                    xy=(idx_min, fk_real[idx_min]),
                    xytext=(max(2, idx_min - 18), fk_real[idx_min] * 0.72),
                    color=strategy_color(2, "Full Kelly"), fontsize=7.8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=strategy_color(2, "Full Kelly"), lw=1.0))

    P.titles(ax, f"Real {ds.label}: one trader + 60-offset range", "Round")
    for a in axes[0]:
        a.set_yscale("log")
        a.yaxis.set_major_locator(P.tk.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        a.yaxis.set_major_formatter(P.tk.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.yaxis.set_minor_formatter(P.tk.NullFormatter())
        a.axhline(C.W0, color=P.INK2, lw=0.8, ls=":")
    P.legend(axes[0, 0], loc="upper left")
    P.save(fig, C.PLOTS_DIR / f"q6_tournament_{ds.key}.png",
           f"Q6 - Six strategies over {K} rounds (shaded: 10-90% of careers for Full/Half Kelly), {ds.label}")
    plot_q6_frontier(ds, out, strategies)


def plot_q6_frontier(ds, out, strategies):
    """Two-panel non-line figure for Q6:
    Left: Risk-Return scatter/bubble frontier (Median Max Drawdown vs Median CAGR, GBM world & Real).
    Right: Grouped horizontal bar chart of drawdown & ruin tail risks over K=100 rounds (Merton world)."""
    fig, axes = P.new_fig(ncols=2, w=6.1, h=4.7)
    ax_sc, ax_bar = axes[0, 0], axes[0, 1]
    K = out["K_real"]
    gbm_w = out["worlds"][f"GBM/K{K}"]
    real_w = out["real"]["offset0"]

    ax_sc.axhline(ds.rf, color=P.INK2, lw=1.0, ls=":", label=f"Cash rate r_f ({ds.rf:.1%})")
    for i, st in enumerate(strategies):
        col = strategy_color(i, st.name)
        lab = strategy_label(ds, st.name)
        m_sim = gbm_w[st.name]
        m_re = real_w[st.name]
        # Bubble size scales with P(MDD > 50%) in simulation, plus base size
        sz_sim = 85 + 260 * m_sim["p_mdd_gt_50"]
        ax_sc.scatter(m_sim["mdd_median"], m_sim["cagr_median"], s=sz_sim, color=col, alpha=0.82,
                      edgecolors=P.SURFACE, linewidth=1.5, zorder=4,
                      label=f"{lab} (f={'-' if m_sim['f'] is None else f'{m_sim['f']:.2f}'})")
        ax_sc.scatter(m_re["mdd_median"], m_re["cagr_median"], s=75, marker="D", color=col,
                      edgecolors=P.INK, linewidth=1.0, zorder=5)
        # Connect simulated median to real outcome with a subtle dotted vector
        ax_sc.annotate("", xy=(m_re["mdd_median"], m_re["cagr_median"]),
                       xytext=(m_sim["mdd_median"], m_sim["cagr_median"]),
                       arrowprops=dict(arrowstyle="->", color=col, lw=0.9, ls=":", alpha=0.65))
        dy = -0.0022 if st.name == "Fixed 50%" and ds.key == "nifty" else (
             +0.0018 if st.name == "Full Kelly" and ds.key == "nifty" else 0.0)
        ax_sc.text(m_sim["mdd_median"] + 0.014, m_sim["cagr_median"] + dy, lab,
                   fontsize=7.8, color=col, va="center", ha="left", fontweight="bold")

    # Dummy legend entries for marker shapes
    ax_sc.scatter([], [], s=80, color=P.INK2, label="Circle = GBM world median (size ~ P(MDD>50%))")
    ax_sc.scatter([], [], s=60, marker="D", color=P.INK2, edgecolors=P.INK, label="Diamond = Real test path")
    P.pct_axis(ax_sc.xaxis, 0)
    P.pct_axis(ax_sc.yaxis, 1)
    ax_sc.set_xlim(0, 1.02)
    P.titles(ax_sc, f"Risk-return frontier over {K} rounds (GBM world vs Real {ds.label})",
             "Median maximum drawdown (daily mark-to-market)", "Median annualised CAGR")
    leg_loc = "upper right" if ds.key == "nifty" else "upper left"
    ax_sc.legend(loc=leg_loc, fontsize=7.5, frameon=True, facecolor=P.SURFACE,
                 edgecolor=P.GRID, framealpha=0.92, labelcolor=P.INK2)

    # Right panel: Grouped horizontal bar chart of drawdown & tail risk over K=100 rounds (Merton jump world)
    mer_long = out["worlds"][f"Merton/K{C.K_LONG}"]
    st_rev = list(strategies)[::-1]
    y_pos = np.arange(len(st_rev))
    mdds = [mer_long[st.name]["mdd_median"] for st in st_rev]
    p_mdd50 = [mer_long[st.name]["p_mdd_gt_50"] for st in st_rev]
    p_half = [mer_long[st.name]["p_lose_half"] for st in st_rev]
    cagrs = [mer_long[st.name]["cagr_median"] for st in st_rev]

    h = 0.26
    ax_bar.barh(y_pos + h, mdds, height=h, color="#eb6834", alpha=0.88, label="Median max drawdown")
    ax_bar.barh(y_pos, p_mdd50, height=h, color="#eda100", alpha=0.85, label="P(max drawdown > 50%)")
    ax_bar.barh(y_pos - h, p_half, height=h, color="#e34948", alpha=0.88, label="P(terminal wealth < 50)")

    for idx, (st, mdd_v, pm_v, ph_v, cg_v) in enumerate(zip(st_rev, mdds, p_mdd50, p_half, cagrs)):
        xmax = max(mdd_v, pm_v, ph_v)
        ax_bar.text(min(xmax + 0.02, 0.99), idx, f"CAGR {cg_v:+.1%}",
                    va="center", ha="left", fontsize=7.8, color=P.INK, fontweight="bold")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([strategy_label(ds, st.name) for st in st_rev], fontsize=8.5, color=P.INK)
    ax_bar.set_xlim(0, 1.19)
    P.pct_axis(ax_bar.xaxis, 0)
    P.titles(ax_bar, f"Long-run risk profile ({C.K_LONG} rounds, Merton jump-diffusion world)",
             "Probability / drawdown fraction", None)
    ax_bar.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor=P.SURFACE,
                  edgecolor=P.GRID, framealpha=0.92, labelcolor=P.INK2)

    P.save(fig, C.PLOTS_DIR / f"q6_frontier_{ds.key}.png",
           f"Q6 - Strategy risk-return frontier & long-run drawdown tail risk, {ds.label}")


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
    fig = P.plt.figure(figsize=(14.6, 6.8), facecolor=P.SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.18], hspace=0.34, wspace=0.30)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])
    ax_hm = fig.add_subplot(gs[:, 2])
    for a in (*axes.flat, ax_hm):
        P.style_ax(a)

    cs = np.array(C.Q7_MULTIPLIERS)
    style = {"Full Kelly (3x cap)": (P.CAT[1], "-"), "Half Kelly (3x cap)": (P.CAT[0], "-"),
             "Full Kelly (no cap)": (P.CAT[1], "--"), "Half Kelly (no cap)": (P.CAT[0], "--")}
    w_short = {"GBM": "GBM", "Merton": "Merton"}
    for row, w in enumerate(("GBM", "Merton")):
        rows = {float(k): v for k, v in out["worlds"][w]["rows"].items()}  # JSON keys are str
        axes[row, 0].axhline(ds.rf, color=P.INK2, lw=1.0, ls="--",
                             label=f"Cash rate r_f ({ds.rf:.1%})" if row == 0 else None)
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
        P.titles(axes[row, 0], f"{w_short[w]} world: expected growth / year", xl,
                 "Log growth / year")
        P.titles(axes[row, 1], f"{w_short[w]} world: P(lose half), {C.K_LONG} rounds", xl)
    P.legend(axes[0, 1], loc="upper left", handlelength=3.2, fontsize=8.0)

    # Right column: 2D Heatmap phase diagram of Bet Fraction f(c) vs True Merton Growth & Ruin Cliff
    from matplotlib.colors import TwoSlopeNorm
    R_m, T_m = episodes_RT(ds, "train", "Merton")
    rows_m = {float(k): v for k, v in out["worlds"]["Merton"]["rows"].items()}
    f_max_curve = max(rows_m[c]["Full Kelly (no cap)"]["f"] for c in C.Q7_MULTIPLIERS)
    f_top = min(max(f_max_curve * 1.06, 3.4), 6.2)
    f_grid = np.linspace(0.0, f_top, 240)
    g_grid = TN.growth_curve(R_m, T_m, ds.rf, f_grid) / T_m.mean() * C.TRADING_DAYS
    finite_g = g_grid[np.isfinite(g_grid)]
    vmax = float(np.max(finite_g)) + 0.004
    vmin = min(-0.08, ds.rf - 0.12)
    Z = np.tile(np.clip(np.where(np.isfinite(g_grid), g_grid, np.nan), vmin, vmax)[:, None], (1, len(cs)))

    norm = TwoSlopeNorm(vmin=vmin, vcenter=ds.rf, vmax=vmax)
    im = ax_hm.pcolormesh(cs, f_grid, Z, cmap="RdYlGn", norm=norm, shading="gouraud", alpha=0.72)
    cb = fig.colorbar(im, ax=ax_hm, pad=0.02, fraction=0.046)
    cb.ax.yaxis.set_major_formatter(P.tk.PercentFormatter(1.0, decimals=0))
    cb.set_label("True annualised log growth (Merton world)", color=P.INK2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=P.INK2)

    # Shade the Ruin Cliff (g = -inf) if within f_top
    busted_f = f_grid[~np.isfinite(g_grid)]
    if busted_f.size:
        f_cliff = float(busted_f[0])
        ax_hm.axhspan(f_cliff, f_top, color="#2b2a28", alpha=0.82, hatch="//")
        ax_hm.text(1.85, (f_cliff + f_top) / 2,
                   f"RUIN CLIFF (f >= {f_cliff:.2f}x)\n1 + f * R_min <= 0 -> growth = -inf",
                   color="white", fontsize=7.6, fontweight="bold", ha="center", va="center")

    f_true_m = out["worlds"]["Merton"]["f_true"]
    ax_hm.axhline(f_true_m, color="#006b3c", lw=1.2, ls=":", label=f"True optimal f* ({f_true_m:.2f}x)")
    ax_hm.axhline(C.LEVERAGE_CAP, color=P.INK, lw=1.1, ls="-.", label=f"Leverage cap ({C.LEVERAGE_CAP:.0f}x)")
    ax_hm.axvline(1.0, color=P.INK2, lw=0.8, ls=":")

    for lab, (col, ls) in style.items():
        f_traj = [min(rows_m[c][lab]["f"], f_top - 0.03) for c in C.Q7_MULTIPLIERS]
        ax_hm.plot(cs, f_traj, color=col, ls=ls, lw=2.2, marker="o", ms=4.5,
                   mec=P.SURFACE, mew=0.8, label=lab)

    ax_hm.set_xlim(cs[0], cs[-1])
    ax_hm.set_ylim(0, f_top)
    P.titles(ax_hm, "Phase diagram: believed bet f(c) vs growth zones",
             "Trader's drift estimate / true drift (c)", "Fraction of wealth bet, f (leverage)")
    ax_hm.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor=P.SURFACE,
                 edgecolor=P.GRID, framealpha=0.92, labelcolor=P.INK)

    fig.suptitle(f"Q7 - Betting Kelly on a misestimated drift, {ds.label}",
                 color=P.INK, fontsize=12, x=0.01, ha="left", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.09, top=0.90)
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path_q7 = C.PLOTS_DIR / f"q7_drift_error_{ds.key}.png"
    fig.savefig(path_q7, dpi=150)
    P.plt.close(fig)
    print(f"saved {path_q7}")


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
