"""Step 3: real-data barrier scanner and model scoring (Q2, Q3, Q4).

Run:  .venv/Scripts/python run_step3_real.py      (needs run_engine_a.py output)
Writes results/real_<dataset>.json and plots.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats as st

from barrier import config as C
from barrier import data as D
from barrier import engine as E
from barrier import plotstyle as P
from barrier import scanner as S
from barrier import theory as T
from barrier.models import calibrate_all, rolling_entry_models

MODELS = ("RandomWalk", "GBM", "Merton")
PARAMS = ("train", "rolling", "oracle")


def model_episodes(ds) -> dict:
    return {(p, m): E.apply_cap(E.load(C.RESULTS_DIR / "sims" / f"{ds.key}_{p}_{m}.npz"),
                                C.CAP_DAYS)
            for p in PARAMS for m in MODELS}


def inside(x, ci) -> bool:
    return ci[0] <= x <= ci[1]


def run(ds) -> dict:
    print("\n" + "#" * 96 + f"\n#  {ds.label}: test period {ds.test[0]} -> {ds.test[1]}, "
          f"cap {C.CAP_DAYS} days\n" + "#" * 96)
    full, train, test, _ = D.load(ds)
    rng = np.random.default_rng([C.SEED, 3, len(ds.key)])

    out = {"dataset": ds.key, "cap": C.CAP_DAYS}
    scans = {m: S.scan(test, mode=m) for m in ("close", "intraday")}

    # ---------------- overlapping windows + block bootstrap
    for mode, sc in scans.items():
        ov = S.overlapping(sc)
        s = S.stats(ov)
        s["ci"] = {L: S.block_bootstrap(ov, L, C.BOOT_REPS, rng) for L in C.BOOT_BLOCKS}
        s["first_start"], s["last_start"] = str(ov.start.iloc[0].date()), str(ov.start.iloc[-1].date())
        if mode == "intraday":
            r = ov[ov.outcome == E.RUIN]
            s["ret_mean_ruin_reliable_open"] = float(r[r.exit_open_ok].ret.mean())
            s["n_ruin_reliable_open"] = int(r.exit_open_ok.sum())
        out[f"overlap_{mode}"] = s

    # ---------------- non-overlapping chains from 60 start offsets
    chains = {o: S.chain(scans["close"], o) for o in range(C.CHAIN_OFFSETS)}
    ch_stats = [S.stats(c) for c in chains.values()]
    out["chain_close_offset0"] = S.stats(chains[0])
    out["chain_close_offset0"]["trades"] = [
        {"start": str(r.start.date()), "outcome": int(r.outcome), "tau": int(r.tau),
         "ret": float(r.ret)} for r in chains[0].itertuples()]
    out["chain_intraday_offset0"] = S.stats(S.chain(scans["intraday"], 0))
    out["chain_spread"] = {
        k: {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)), "max": float(np.nanmax(v))}
        for k in ("n", "p_win", "tau_mean", "ret_mean_ruin")
        for v in [np.array([c[k] for c in ch_stats], float)]}

    # ---------------- model predictions + scoring
    eps = model_episodes(ds)
    ov = out["overlap_close"]
    ci = ov["ci"][C.BOOT_BLOCK_MAIN]
    real_tau_chain0 = chains[0].tau.to_numpy()
    scores = {}
    for (p, m), ep in eps.items():
        s = E.summarize(ep)
        ks0 = st.ks_2samp(ep.tau, real_tau_chain0)
        ks_all = [st.ks_2samp(ep.tau, c.tau.to_numpy()).pvalue for c in chains.values()]
        scores[f"{p}/{m}"] = {
            **{k: s[k] for k in ("p_win", "p_ruin", "p_timeout", "tau_mean", "tau_median",
                                 "tau_sd", "tau_p05", "tau_p95", "ret_mean_ruin",
                                 "ret_p01_ruin")},
            "win_in_ci": inside(s["p_win"], ci["p_win"]),
            "tau_in_ci": inside(s["tau_mean"], ci["tau_mean"]),
            "ruin_ret_in_ci": inside(s["ret_mean_ruin"], ci["ret_mean_ruin"]),
            "ks_D_chain0": float(ks0.statistic), "ks_p_chain0": float(ks0.pvalue),
            "ks_p_median_offsets": float(np.median(ks_all)),
            "ks_share_rejected_5pct": float(np.mean(np.array(ks_all) < 0.05)),
        }
    out["models"] = scores

    # ---------------- Q2 sensitivity inputs
    gtrain = calibrate_all(D.log_returns(train))["GBM"]
    gtest_all = calibrate_all(D.log_returns(test))
    gtest = gtest_all["GBM"]
    roll_seq = rolling_entry_models(D.log_returns(full), chains[0].start)
    mu_roll = float(np.mean([m["GBM"].mu for m in roll_seq]))
    sig_roll = float(np.mean([m["GBM"].sigma for m in roll_seq]))
    out["q2"] = {"mu_train": gtrain.mu, "mu_test": gtest.mu, "sigma_train": gtrain.sigma,
                 "sigma_test": gtest.sigma, "mu_train_se": gtrain.nu_se,
                 "mu_rolling": mu_roll, "sigma_rolling": sig_roll}

    report(ds, out)
    plot_durations(ds, eps, chains, S.overlapping(scans["close"]), test, gtest_all["Merton"])
    plot_sensitivity(ds, out)
    plot_q4_overshoot(ds, eps, scans, chains[0], out)
    return out


def report(ds, out) -> None:
    ov, ovi, ch = out["overlap_close"], out["overlap_intraday"], out["chain_close_offset0"]
    ci = ov["ci"]
    print(f"\nOverlapping windows (close): {ov['n']} start days {ov['first_start']} -> {ov['last_start']}")
    print(f"  win {ov['p_win']:.1%}  ruin {ov['p_ruin']:.1%}  timeout {ov['p_timeout']:.1%}  "
          f"mean days {ov['tau_mean']:.0f}  median {ov['tau_median']:.0f}  sd {ov['tau_sd']:.0f}  "
          f"p5-p95 {ov['tau_p05']:.0f}-{ov['tau_p95']:.0f}  stop fill {ov['ret_mean_ruin']:+.2%}")
    for L in C.BOOT_BLOCKS:
        c = ci[L]
        print(f"  95% CI (block {L:>3}d): win {c['p_win'][0]:.1%}-{c['p_win'][1]:.1%}   "
              f"mean days {c['tau_mean'][0]:.0f}-{c['tau_mean'][1]:.0f}   "
              f"stop fill {c['ret_mean_ruin'][0]:+.2%} .. {c['ret_mean_ruin'][1]:+.2%}")
    print(f"Overlapping windows (intraday High/Low + Open gaps): win {ovi['p_win']:.1%}  "
          f"ruin {ovi['p_ruin']:.1%}  mean days {ovi['tau_mean']:.0f}  stop fill "
          f"{ovi['ret_mean_ruin']:+.2%} (reliable-Open exits only: "
          f"{ovi['ret_mean_ruin_reliable_open']:+.2%}, n={ovi['n_ruin_reliable_open']})  "
          f"gap exits {ovi['share_gap_exits']:.1%}  ambiguous days {ovi['share_ambiguous']:.1%}")
    print(f"  -> intraday exits come {ov['tau_mean'] - ovi['tau_mean']:.1f} days earlier on average")
    sp = out["chain_spread"]
    print(f"\nNon-overlapping chain (close, start {ds.test[0]}): {ch['n']} trades  "
          f"win {ch['p_win']:.1%}  mean days {ch['tau_mean']:.0f}  stop fill {ch['ret_mean_ruin']:+.2%}")
    print(f"  across {C.CHAIN_OFFSETS} start offsets: trades {sp['n']['min']:.0f}-{sp['n']['max']:.0f}, "
          f"win {sp['p_win']['min']:.0%}-{sp['p_win']['max']:.0%} (median {sp['p_win']['median']:.0%}), "
          f"mean days {sp['tau_mean']['min']:.0f}-{sp['tau_mean']['max']:.0f}")

    print(f"\n{'model':<18}{'win':>7}{'inCI':>6}{'t/o':>6}{'days':>6}{'inCI':>6}{'sd':>5}"
          f"{'stop':>8}{'inCI':>6}{'stop p1':>9}{'KS D':>7}{'KS p':>7}{'KS p med':>9}{'rej%':>6}")
    print(f"{'REAL (overlap)':<18}{ov['p_win']:>7.1%}{'':>6}{ov['p_timeout']:>6.1%}"
          f"{ov['tau_mean']:>6.0f}{'':>6}{ov['tau_sd']:>5.0f}{ov['ret_mean_ruin']:>+8.2%}")
    print(f"{'REAL (chain 0)':<18}{ch['p_win']:>7.1%}{'':>6}{ch['p_timeout']:>6.1%}"
          f"{ch['tau_mean']:>6.0f}{'':>6}{ch['tau_sd']:>5.0f}{ch['ret_mean_ruin']:>+8.2%}")
    yn = {True: "yes", False: "NO"}
    for k, s in out["models"].items():
        print(f"{k:<18}{s['p_win']:>7.1%}{yn[s['win_in_ci']]:>6}{s['p_timeout']:>6.1%}"
              f"{s['tau_mean']:>6.0f}{yn[s['tau_in_ci']]:>6}{s['tau_sd']:>5.0f}"
              f"{s['ret_mean_ruin']:>+8.2%}{yn[s['ruin_ret_in_ci']]:>6}{s['ret_p01_ruin']:>+9.2%}"
              f"{s['ks_D_chain0']:>7.2f}{s['ks_p_chain0']:>7.2f}{s['ks_p_median_offsets']:>9.2f}"
              f"{s['ks_share_rejected_5pct']:>6.0%}")


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, len(x) + 1) / len(x)


def plot_durations(ds, eps, chains, ov, test, mer_test) -> None:
    chain0 = chains[0]
    fig = P.plt.figure(figsize=(14.4, 7.6), facecolor=P.SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.12)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    ax_tl = fig.add_subplot(gs[1, :])
    for ax in (ax0, ax1, ax2, ax_tl):
        P.style_ax(ax)

    panels = [
        (ax0, "train", "Static training params (frozen at split)"),
        (ax1, "rolling", f"Rolling {C.ROLLING_YEARS}-yr params (updated per trade, out-of-sample)"),
        (ax2, "oracle", "Test-period params (oracle hindsight)"),
    ]
    for ax, p, ttl in panels:
        for idx_o, ch_o in chains.items():
            xo, yo = ecdf(ch_o.tau)
            ax.step(np.r_[0, xo], np.r_[0, yo], where="post", color="#c6c4be", lw=0.7, alpha=0.35,
                    label="Real: 60 start-day offsets" if idx_o == 1 else None)
        for m in MODELS:
            x, y = ecdf(eps[(p, m)].tau)
            ax.plot(x, y, color=P.MODEL_COLOR[m], lw=2, label=P.MODEL_LABEL[m])
        x, y = ecdf(ov.tau)
        ax.plot(x, y, color=P.INK2, lw=1.6, ls="--", label="Real: overlapping windows")
        x, y = ecdf(chain0.tau)
        ax.step(np.r_[0, x], np.r_[0, y], where="post", color=P.INK, lw=2,
                label=f"Real: chain offset 0 (n={len(chain0)})")
        ax.plot(x, y, "o", color=P.INK, ms=3.8)
        ax.set_xlim(0, C.CAP_DAYS + 10)
        P.pct_axis(ax.yaxis)
        P.titles(ax, ttl, "Trading days to exit (cap 504)",
                 "Share of trades finished" if p == "train" else None)
        if p != "train":
            ax.tick_params(labelleft=False)
        if p == "rolling":
            sh30 = float(np.mean(chain0.tau <= 30))
            if sh30 >= 0.15:
                ax.annotate(f"{sh30:.0%} of chain trades exit in <=30d\n(crisis clustering)",
                            xy=(30, sh30), xytext=(115, sh30 - 0.05),
                            color=P.INK, fontsize=8,
                            arrowprops=dict(arrowstyle="->", color=P.INK2, lw=0.9))
    P.legend(ax0, loc="lower right", ncol=1, fontsize=7.8)

    # Bottom panel: calendar timeline barcode / Gantt of sequential chain trades + 3-sigma jumps
    dates = test.index
    outcome_meta = {
        E.WIN: ("#1baf7a", "Win (+20%)"),
        E.RUIN: ("#e34948", "Ruin (-10%)"),
        E.TIMEOUT: ("#8c8a84", "Timeout (504d)"),
    }
    seen_labels = set()
    for r in chain0.itertuples():
        t_start = dates[int(r.entry_idx)]
        t_end = dates[min(int(r.exit_idx), len(dates) - 1)]
        col, lab = outcome_meta[int(r.outcome)]
        lbl = lab if lab not in seen_labels else None
        seen_labels.add(lab)
        width_days = max((t_end - t_start).days, 6)
        ax_tl.bar(t_start, r.tau, width=width_days, align="edge",
                  color=col, alpha=0.82, edgecolor=P.SURFACE, lw=0.7, label=lbl)

    jump_dates = [ dates[dates.get_loc(d)] for d in mer_test.jump_dates if d in dates ] \
        if hasattr(mer_test, "jump_dates") else []
    if not jump_dates and getattr(mer_test, "jump_dates", None):
        import pandas as pd
        jump_dates = pd.to_datetime(mer_test.jump_dates)
    if len(jump_dates):
        ax_tl.plot(jump_dates, np.full(len(jump_dates), -18.0), "|", color=P.INK, ms=8, mew=1.2,
                   label=f"3-sigma jump days (n={len(jump_dates)})")
    ax_tl.axhline(0, color=P.INK2, lw=0.7)
    ax_tl.set_ylim(-42, C.CAP_DAYS + 105)
    P.titles(ax_tl,
             "Calendar timeline of sequential trades (bar width = calendar span, height = trading days; ticks = 3-sigma jump days)",
             "Calendar date (test period)", "Duration (days)")
    P.legend(ax_tl, loc="upper right", ncol=4)
    fig.suptitle(f"Q3/Q4 - Time to hit +20% or -10% & volatility clustering: {ds.label} ({ds.test[0][:4]}-{ds.test[1][:4]})",
                 color=P.INK, fontsize=12, x=0.01, ha="left", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90)
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path_q3 = C.PLOTS_DIR / f"q3_durations_{ds.key}.png"
    fig.savefig(path_q3, dpi=150)
    P.plt.close(fig)
    print(f"saved {path_q3}")


def plot_sensitivity(ds, out) -> None:
    q = out["q2"]
    mus = np.linspace(-0.05, 0.25, 121)
    fig, axes = P.new_fig(w=7.6, h=4.6)
    ax = axes[0, 0]
    se = q.get("mu_train_se", 0.0)
    if se > 0:
        lo_se = max(-0.05, q["mu_train"] - se)
        hi_se = min(0.25, q["mu_train"] + se)
        ax.axvspan(lo_se, hi_se, color=P.CAT[1], alpha=0.11, lw=0,
                   label=f"training drift +-1 SE (+-{se:.1%}/yr)")
    for sig, col, lab in [(q["sigma_train"], P.CAT[1], f"theory, training volatility {q['sigma_train']:.1%}"),
                          (q["sigma_test"], P.CAT[0], f"theory, test-period volatility {q['sigma_test']:.1%}")]:
        pw = [T.gbm_first_passage_daily(m - 0.5 * sig ** 2, sig)[0] for m in mus]
        ax.plot(mus, pw, color=col, lw=2.2, label=lab)
    ov = out["overlap_close"]
    c = ov["ci"][C.BOOT_BLOCK_MAIN]["p_win_resolved"]
    pr = ov["p_win_resolved"]
    ax.axhspan(c[0], c[1], color=P.GRID, alpha=0.75, lw=0, label="real win rate, 95% CI")
    ax.axhline(pr, color=P.INK, lw=1.6, label=f"real win rate (resolved trades) {pr:.0%}")

    # Plot markers at static training, rolling 5y mean, and oracle test point
    p_tr = T.gbm_first_passage_daily(q["mu_train"] - 0.5 * q["sigma_train"] ** 2, q["sigma_train"])[0]
    p_ro = T.gbm_first_passage_daily(q["mu_rolling"] - 0.5 * q["sigma_rolling"] ** 2, q["sigma_rolling"])[0]
    p_te = T.gbm_first_passage_daily(q["mu_test"] - 0.5 * q["sigma_test"] ** 2, q["sigma_test"])[0]
    ax.plot(q["mu_train"], p_tr, "o", color=P.CAT[1], ms=7.5, mec=P.SURFACE, mew=1.5, zorder=5,
            label=f"static training ({p_tr:.1%})")
    ax.plot(q["mu_rolling"], p_ro, "s", color=P.CAT[2], ms=7.5, mec=P.SURFACE, mew=1.5, zorder=6,
            label=f"rolling {C.ROLLING_YEARS}y mean ({p_ro:.1%}, sig={q['sigma_rolling']:.1%})")
    ax.plot(q["mu_test"], p_te, "D", color=P.CAT[0], ms=7.0, mec=P.SURFACE, mew=1.5, zorder=5,
            label=f"oracle test point ({p_te:.1%})")

    lo, hi = sorted([(q["mu_train"], "training drift"), (q["mu_test"], "test-period drift")])
    for (x, lab), ha in [(lo, "right"), (hi, "left")]:
        ax.axvline(x, color=P.INK2, lw=1, ls=":")
        pad = " " if ha == "left" else ""
        ax.text(x, 0.03, f"{pad}{lab}{'' if pad else ' '}\n{pad}{x:.1%}{'' if pad else ' '}",
                color=P.INK2, fontsize=8, va="bottom", ha=ha)
    ax.axhline(1 / 3, color=P.INK2, lw=0.8, ls="--")
    ax.text(0.25, 1 / 3 - 0.015, "1/3 = zero price drift", color=P.INK2, fontsize=8,
            ha="right", va="top")
    P.pct_axis(ax.xaxis)
    P.pct_axis(ax.yaxis)
    ax.set_ylim(0, 1)
    P.titles(ax, "Win probability vs assumed annual drift (daily-close GBM, BGK)",
             "Annual arithmetic drift mu", "P(hit +20% before -10%)")
    P.legend(ax, loc="upper left", ncol=1)
    P.save(fig, C.PLOTS_DIR / f"q2_sensitivity_{ds.key}.png",
           f"Q2 - Drift sensitivity & estimation uncertainty, {ds.label}")


def plot_q4_overshoot(ds, eps, scans, chain0, out) -> None:
    """Two-panel non-line figure for Q4:
    Left: Horizontal stacked outcome bar chart (Win / Ruin / Timeout) across models and real data.
    Right: Horizontal raincloud (half-violin + boxplot + real trade strip) of stop-loss overshoot."""
    fig, axes = P.new_fig(ncols=2, w=6.0, h=4.8)
    ax_bar, ax_rc = axes[0, 0], axes[0, 1]

    # ---------------- Left panel: 100% stacked horizontal outcome bars
    ov_c = out["overlap_close"]
    ov_i = out["overlap_intraday"]
    ch_c = out["chain_close_offset0"]
    ci_win = ov_c["ci"][C.BOOT_BLOCK_MAIN]["p_win"]
    rows = [
        ("Static train: Random walk", out["models"]["train/RandomWalk"], False),
        ("Static train: GBM", out["models"]["train/GBM"], False),
        ("Static train: Merton", out["models"]["train/Merton"], False),
        (f"Rolling {C.ROLLING_YEARS}y: Random walk", out["models"]["rolling/RandomWalk"], False),
        (f"Rolling {C.ROLLING_YEARS}y: GBM", out["models"]["rolling/GBM"], False),
        (f"Rolling {C.ROLLING_YEARS}y: Merton", out["models"]["rolling/Merton"], False),
        ("Oracle: Random walk", out["models"]["oracle/RandomWalk"], False),
        ("Oracle: GBM", out["models"]["oracle/GBM"], False),
        ("Oracle: Merton", out["models"]["oracle/Merton"], False),
        ("Real: Intraday H/L + Open", ov_i, False),
        ("Real: Overlapping (Close)", ov_c, True),
        (f"Real: Chain (n={ch_c['n']})", ch_c, False),
    ][::-1]

    y_pos = np.arange(len(rows))
    wins = np.array([r[1]["p_win"] for r in rows])
    ruins = np.array([r[1]["p_ruin"] for r in rows])
    tos = np.array([r[1]["p_timeout"] for r in rows])

    ax_bar.barh(y_pos, wins, color="#1baf7a", alpha=0.88, height=0.62, label="Win (+20%)")
    ax_bar.barh(y_pos, ruins, left=wins, color="#e34948", alpha=0.85, height=0.62, label="Ruin (-10%)")
    ax_bar.barh(y_pos, tos, left=wins + ruins, color="#8c8a84", alpha=0.80, height=0.62, label="Timeout (504d)")

    for idx, (label, d, has_ci) in enumerate(rows):
        if d["p_win"] >= 0.12:
            ax_bar.text(d["p_win"] / 2, idx, f"{d['p_win']:.1%}", color="white",
                        fontsize=7.8, fontweight="bold", va="center", ha="center")
        if d["p_ruin"] >= 0.12:
            ax_bar.text(d["p_win"] + d["p_ruin"] / 2, idx, f"{d['p_ruin']:.1%}", color="white",
                        fontsize=7.8, fontweight="bold", va="center", ha="center")
        if has_ci:
            ax_bar.errorbar(d["p_win"], idx, xerr=[[d["p_win"] - ci_win[0]], [ci_win[1] - d["p_win"]]],
                            fmt="o", color=P.INK, ms=4, capsize=4, lw=1.4, zorder=6,
                            label="Real win 95% CI")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=P.INK)
    ax_bar.set_xlim(0, 1.0)
    P.pct_axis(ax_bar.xaxis)
    P.titles(ax_bar, "Outcome shares (504-day cap)", "Share of trades", None)
    P.legend(ax_bar, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)

    # ---------------- Right panel: Raincloud / boxplot of stop-loss fill on ruin trades
    ov_close_df = S.overlapping(scans["close"])
    ov_intra_df = S.overlapping(scans["intraday"])
    ruin_series = [
        ("Random walk (rolling 5y)", np.expm1(eps[("rolling", "RandomWalk")].exit_log[eps[("rolling", "RandomWalk")].outcome == E.RUIN]), P.MODEL_COLOR["RandomWalk"]),
        ("GBM (rolling 5y)", np.expm1(eps[("rolling", "GBM")].exit_log[eps[("rolling", "GBM")].outcome == E.RUIN]), P.MODEL_COLOR["GBM"]),
        ("Merton (rolling 5y)", np.expm1(eps[("rolling", "Merton")].exit_log[eps[("rolling", "Merton")].outcome == E.RUIN]), P.MODEL_COLOR["Merton"]),
        ("Merton (static train)", np.expm1(eps[("train", "Merton")].exit_log[eps[("train", "Merton")].outcome == E.RUIN]), "#20855a"),
        ("Real: Intraday + Open gaps", ov_intra_df[ov_intra_df.outcome == E.RUIN].ret.to_numpy(), "#6250d6"),
        ("Real: Daily Close", ov_close_df[ov_close_df.outcome == E.RUIN].ret.to_numpy(), P.INK),
    ][::-1]

    y_rc = np.arange(len(ruin_series))
    x_grid = np.linspace(-0.22, -0.095, 250)
    for idx, (lab, arr, col) in enumerate(ruin_series):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        # Half-violin KDE above the line if variance > 0
        if np.std(arr) > 1e-4 and arr.size > 5:
            kde = st.gaussian_kde(arr, bw_method=0.28)
            dens = kde(x_grid)
            dens = dens / max(dens.max(), 1e-12) * 0.36
            ax_rc.fill_between(x_grid, idx + 0.04, idx + 0.04 + dens, color=col, alpha=0.32, lw=1.2, edgecolor=col)
        else:
            ax_rc.vlines(np.mean(arr), idx - 0.25, idx + 0.35, color=col, lw=3.0)

        # Horizontal boxplot / quantile bar (p5 - p95 whisker, IQR box, mean diamond)
        p01, p05, p25, p50, p75, p95 = np.percentile(arr, [1, 5, 25, 50, 75, 95])
        mean_v = float(np.mean(arr))
        ax_rc.hlines(idx - 0.10, p05, p95, color=col, lw=1.4)
        ax_rc.fill_between([p25, p75], idx - 0.18, idx - 0.02, color=col, alpha=0.65)
        ax_rc.plot(mean_v, idx - 0.10, "D", color=P.SURFACE, mec=col, mew=1.6, ms=5.5,
                   label="Mean stop fill" if idx == 0 else None)
        ax_rc.plot(p01, idx - 0.10, "x", color=col, ms=5.5, mew=1.4,
                   label="Worst 1% (or min)" if idx == 0 else None)
        ax_rc.text(min(mean_v - 0.002, -0.104), idx + 0.16, f"mean {mean_v:+.2%}",
                   color=col, fontsize=7.8, va="bottom", ha="right", fontweight="bold")

    # Overlay actual non-overlapping chain ruin trades as prominent dots on the Real: Daily Close row
    ch_ruin = chain0[chain0.outcome == E.RUIN].ret.to_numpy()
    if ch_ruin.size:
        jitter = np.linspace(-0.26, -0.18, len(ch_ruin))
        ax_rc.plot(ch_ruin, jitter, "o", color="#e34948", mec=P.INK, mew=0.6, ms=5.0,
                   label=f"Real chain ruin trades (n={len(ch_ruin)})")

    ci_stop = ov_c["ci"][C.BOOT_BLOCK_MAIN]["ret_mean_ruin"]
    ax_rc.axvspan(ci_stop[0], ci_stop[1], color=P.GRID, alpha=0.75, zorder=0,
                  label="Real close mean stop 95% CI")
    ax_rc.axvline(-0.10, color=P.INK2, lw=1.1, ls="--", label="Nominal -10% barrier")

    ax_rc.set_yticks(y_rc)
    ax_rc.set_yticklabels([r[0] for r in ruin_series], fontsize=8.5, color=P.INK)
    ax_rc.set_xlim(-0.205, -0.092)
    ax_rc.set_ylim(-0.45, len(ruin_series) - 0.4)
    P.pct_axis(ax_rc.xaxis, 0)
    P.titles(ax_rc, "Stop-loss fill distribution on losing trades (overshoot)",
             "Realised return on ruin exit", None)
    P.legend(ax_rc, loc="upper left", fontsize=8.0)

    P.save(fig, C.PLOTS_DIR / f"q4_overshoot_{ds.key}.png",
           f"Q4 - Outcome shares & stop-loss overshoot distribution, {ds.label} ({ds.test[0][:4]}-{ds.test[1][:4]})")


def main() -> None:
    for ds in C.DATASETS.values():
        out = run(ds)
        path = C.RESULTS_DIR / f"real_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
