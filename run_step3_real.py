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
from barrier.models import calibrate_all

MODELS = ("RandomWalk", "GBM", "Merton")
PARAMS = ("train", "oracle")


def model_episodes(ds) -> dict:
    return {(p, m): E.apply_cap(E.load(C.RESULTS_DIR / "sims" / f"{ds.key}_{p}_{m}.npz"),
                                C.CAP_DAYS)
            for p in PARAMS for m in MODELS}


def inside(x, ci) -> bool:
    return ci[0] <= x <= ci[1]


def run(ds) -> dict:
    print("\n" + "#" * 96 + f"\n#  {ds.label}: test period {ds.test[0]} -> {ds.test[1]}, "
          f"cap {C.CAP_DAYS} days\n" + "#" * 96)
    _, train, test, _ = D.load(ds)
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
    gtest = calibrate_all(D.log_returns(test))["GBM"]
    out["q2"] = {"mu_train": gtrain.mu, "mu_test": gtest.mu, "sigma_train": gtrain.sigma,
                 "sigma_test": gtest.sigma}

    report(ds, out)
    plot_durations(ds, eps, chains[0], S.overlapping(scans["close"]))
    plot_sensitivity(ds, out)
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


def plot_durations(ds, eps, chain0, ov) -> None:
    fig, axes = P.new_fig(ncols=2, w=5.8, h=4.4, sharey=True)
    for ax, p, ttl in [(axes[0, 0], "train", "Models with training parameters (the forecast)"),
                       (axes[0, 1], "oracle", "Models with test-period parameters (oracle)")]:
        for m in MODELS:
            x, y = ecdf(eps[(p, m)].tau)
            ax.plot(x, y, color=P.MODEL_COLOR[m], lw=2, label=P.MODEL_LABEL[m])
        x, y = ecdf(ov.tau)
        ax.plot(x, y, color=P.INK2, lw=1.5, ls="--", label="Real: overlapping windows")
        x, y = ecdf(chain0.tau)
        ax.step(np.r_[0, x], np.r_[0, y], where="post", color=P.INK, lw=2,
                label=f"Real: non-overlapping chain (n={len(chain0)})")
        ax.plot(x, y, "o", color=P.INK, ms=4)
        ax.set_xlim(0, C.CAP_DAYS + 10)
        P.pct_axis(ax.yaxis)
        P.titles(ax, ttl, "Trading days to exit (cap 504)",
                 "Share of trades finished" if p == "train" else None)
    P.legend(axes[0, 0], loc="lower right")
    P.save(fig, C.PLOTS_DIR / f"q3_durations_{ds.key}.png",
           f"Q3/Q4 - Time to hit +20% or -10%: models vs real {ds.label} ({ds.test[0][:4]}-{ds.test[1][:4]})")


def plot_sensitivity(ds, out) -> None:
    q = out["q2"]
    mus = np.linspace(-0.05, 0.25, 121)
    fig, axes = P.new_fig(w=7.2, h=4.4)
    ax = axes[0, 0]
    for sig, col, lab in [(q["sigma_train"], P.CAT[1], f"theory, training volatility {q['sigma_train']:.1%}"),
                          (q["sigma_test"], P.CAT[0], f"theory, test-period volatility {q['sigma_test']:.1%}")]:
        pw = [T.gbm_first_passage_daily(m - 0.5 * sig ** 2, sig)[0] for m in mus]
        ax.plot(mus, pw, color=col, lw=2, label=lab)
    ov = out["overlap_close"]
    c = ov["ci"][C.BOOT_BLOCK_MAIN]["p_win_resolved"]
    pr = ov["p_win_resolved"]
    ax.axhspan(c[0], c[1], color=P.GRID, alpha=0.8, lw=0, label="real win rate, 95% CI")
    ax.axhline(pr, color=P.INK, lw=1.5, label=f"real win rate (resolved trades) {pr:.0%}")
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
    P.legend(ax, loc="upper left")
    P.save(fig, C.PLOTS_DIR / f"q2_sensitivity_{ds.key}.png",
           f"Q2 - Drift sensitivity, {ds.label}")


def main() -> None:
    for ds in C.DATASETS.values():
        out = run(ds)
        path = C.RESULTS_DIR / f"real_{ds.key}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
