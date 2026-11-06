"""Engine A: single-episode Monte Carlo. Answers Q1 and produces each model's predictions.

Run:  .venv/Scripts/python run_engine_a.py
Writes results/engine_a_<dataset>.json, results/sims/*.npz, results/plots/q1_convergence_*.png
"""
from __future__ import annotations

import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barrier import config as C
from barrier import data as D
from barrier import engine as E
from barrier import scanner as S
from barrier import theory as T
from barrier.models import calibrate_all, rolling_entry_models

SIMS_DIR = C.RESULTS_DIR / "sims"
NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 50_000]
MAX_REPS = 1_000

# reference palette (dataviz skill, light mode)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"rw": "#2a78d6", "bgk": "#eb6834", "cont": "#1baf7a"}


def rng_for(*key) -> np.random.Generator:
    """Independent, reproducible stream per (dataset, params, model, purpose)."""
    seed = [C.SEED] + [sum(ord(ch) * 131 ** i for i, ch in enumerate(str(k))) % 2**31
                       for k in key]
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ Q1
def convergence(ep: E.Episodes, p_ref: float, t_ref: float) -> dict:
    """Split the pool into disjoint replications of size N; RMSE and CI coverage per N."""
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
            "p_se_theory": math.sqrt(p_ref * (1 - p_ref) / N),
        }
    return out


def q1(ds, models, results) -> None:
    print(f"\n--- Q1 convergence ({ds.label}, training parameters, uncapped, "
          f"pool = {C.N_CONVERGENCE_POOL:,} paths per model)")
    rw, gbm = models["RandomWalk"], models["GBM"]
    t0 = time.time()
    ep_rw = E.simulate(rw, C.N_CONVERGENCE_POOL, rng_for(ds.key, "q1", "rw"))
    ep_g = E.simulate(gbm, C.N_CONVERGENCE_POOL, rng_for(ds.key, "q1", "gbm"))
    print(f"simulated in {time.time()-t0:.1f}s; still open at safety limit: "
          f"RW {np.mean(ep_rw.outcome == E.TIMEOUT):.2e}, GBM {np.mean(ep_g.outcome == E.TIMEOUT):.2e}")

    th_rw, th_g = rw.theory(), gbm.theory()
    refs = {
        "rw": ("RW sim vs exact lattice formula", ep_rw,
               th_rw["p_win_exact_lattice"], th_rw["E_tau_days_exact_lattice"]),
        "bgk": ("GBM daily sim vs BGK-corrected formula", ep_g,
                th_g["p_win_daily_bgk"], th_g["E_tau_days_daily_bgk"]),
        "cont": ("GBM daily sim vs continuous formula", ep_g,
                 th_g["p_win_continuous"], th_g["E_tau_days_continuous"]),
    }
    q1_out = {}
    print(f"{'comparison':<42}{'theory p':>9}{'pooled p':>10}{'z':>7}"
          f"{'theory E[t]':>12}{'pooled':>9}{'z':>7}")
    for key, (label, ep, p_ref, t_ref) in refs.items():
        s = E.summarize(ep)
        zp = (s["p_win"] - p_ref) / s["p_win_se"]
        zt = (s["tau_mean"] - t_ref) / s["tau_se"]
        print(f"{label:<42}{p_ref:>9.4f}{s['p_win']:>10.4f}{zp:>+7.1f}"
              f"{t_ref:>12.2f}{s['tau_mean']:>9.2f}{zt:>+7.1f}")
        conv = convergence(ep, p_ref, t_ref)
        q1_out[key] = {"label": label, "p_ref": p_ref, "t_ref": t_ref,
                       "pooled_p": s["p_win"], "pooled_p_se": s["p_win_se"], "z_p": zp,
                       "pooled_t": s["tau_mean"], "pooled_t_se": s["tau_se"], "z_t": zt,
                       "by_N": conv}

    print(f"\n{'N':>7}{'reps':>6} | " + " | ".join(
        f"{k:>4}: p_rmse  cover  t_rmse  cover" for k in refs))
    for N in NS:
        row = " | ".join(
            f"      {q1_out[k]['by_N'][N]['p_rmse']:.4f} {q1_out[k]['by_N'][N]['p_cover']:6.1%}"
            f" {q1_out[k]['by_N'][N]['t_rmse']:7.2f} {q1_out[k]['by_N'][N]['t_cover']:6.1%}"
            for k in refs)
        print(f"{N:>7}{q1_out['rw']['by_N'][N]['reps']:>6} | {row}")
    results["q1"] = q1_out
    plot_q1(ds, q1_out)


def plot_q1(ds, q1_out) -> None:
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="#fcfcfb")
    labels = {"rw": "Random walk vs exact lattice formula",
              "bgk": "Daily GBM vs BGK-corrected formula",
              "cont": "Daily GBM vs continuous formula"}
    for ax, metric, title, unit in [(axes[0], "p_rmse", "Win probability", "RMSE (probability)"),
                                    (axes[1], "t_rmse", "Mean stopping time", "RMSE (days)")]:
        ax.set_facecolor("#fcfcfb")
        Ns = np.array(NS)
        for k in ("rw", "bgk", "cont"):
            y = [q1_out[k]["by_N"][N][metric] for N in NS]
            ax.plot(Ns, y, color=SERIES[k], lw=2, marker="o", ms=5, label=labels[k],
                    ls="--" if k == "cont" else "-")
        # 1/sqrt(N) reference anchored on the RW curve at the smallest N
        y0 = q1_out["rw"]["by_N"][NS[0]][metric]
        ax.plot(Ns, y0 * np.sqrt(NS[0] / Ns), color=INK2, lw=1, ls=":",
                label=r"$\propto 1/\sqrt{N}$ reference")
        ax.set_xscale("log"); ax.set_yscale("log")
        tk = matplotlib.ticker
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(tk.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(tk.NullFormatter())
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Episodes per estimate, N", color=INK2)
        ax.set_ylabel(unit, color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    fig.suptitle(f"Q1 - Monte Carlo convergence to closed-form theory ({ds.label}, "
                 "training parameters)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = C.PLOTS_DIR / f"q1_convergence_{ds.key}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# ------------------------------------------------------------------ predictions
def predictions(ds, param_sets, results) -> None:
    SIMS_DIR.mkdir(parents=True, exist_ok=True)
    preds = {}
    for pname, models in param_sets.items():
        preds[pname] = {}
        for mname, model in models.items():
            rng = rng_for(ds.key, pname, mname)
            ep = E.simulate_rolling(model, C.N_EPISODES, rng) if isinstance(model, list) \
                else E.simulate(model, C.N_EPISODES, rng)
            np.savez_compressed(SIMS_DIR / f"{ds.key}_{pname}_{mname}.npz",
                                outcome=ep.outcome, tau=ep.tau, exit_log=ep.exit_log,
                                **{f"snap_{c}": v for c, v in ep.snap.items()})
            preds[pname][mname] = {
                ("uncapped" if cap is None else f"cap{cap}"): E.summarize(E.apply_cap(ep, cap))
                for cap in (None, *C.CAPS_REPORTED)}
    results["predictions"] = preds

    tags = {
        "train": "STATIC TRAINING params (frozen at split)",
        "rolling": f"ROLLING {C.ROLLING_YEARS}-yr params (updated before each trade, out-of-sample)",
        "oracle": "TEST-period params (ORACLE, diagnosis only)",
    }
    for pname in param_sets:
        tag = tags.get(pname, pname)
        for view in ("uncapped", f"cap{C.CAP_DAYS}", "cap252"):
            print(f"\n--- {ds.label} | {tag} | {view} | N = {C.N_EPISODES:,} per model")
            print(f"{'model':<11}{'win':>7}{'ruin':>7}{'t/out':>7}{'mean d':>8}{'median':>8}"
                  f"{'sd':>7}{'p5':>6}{'p95':>6}{'d|win':>7}{'d|ruin':>7}"
                  f"{'stop ret':>9}{'stop p1':>9}{'t/o ret':>8}")
            for mname, r in preds[pname].items():
                s = r[view]
                print(f"{mname:<11}{s['p_win']:>7.1%}{s['p_ruin']:>7.1%}{s['p_timeout']:>7.1%}"
                      f"{s['tau_mean']:>8.1f}{s['tau_median']:>8.0f}{s['tau_sd']:>7.0f}"
                      f"{s['tau_p05']:>6.0f}{s['tau_p95']:>6.0f}{s['tau_mean_win']:>7.0f}"
                      f"{s['tau_mean_ruin']:>7.0f}{s['ret_mean_ruin']:>+9.2%}"
                      f"{s['ret_p01_ruin']:>+9.2%}"
                      + (f"{s['ret_mean_timeout']:>+8.1%}" if s['p_timeout'] > 0 else f"{'-':>8}"))


def main() -> None:
    for ds in C.DATASETS.values():
        print("\n" + "#" * 100 + f"\n#  {ds.label}\n" + "#" * 100)
        full, train, test, _ = D.load(ds)
        ch0 = S.chain(S.scan(test, mode="close"), 0)
        roll_seq = rolling_entry_models(D.log_returns(full), ch0.start)
        rolling_by_model = {m: [entry[m] for entry in roll_seq] for m in ("RandomWalk", "GBM", "Merton")}
        param_sets = {"train": calibrate_all(D.log_returns(train)),
                      "rolling": rolling_by_model,
                      "oracle": calibrate_all(D.log_returns(test))}
        results = {"dataset": ds.key, "n_episodes": C.N_EPISODES,
                   "caps": list(C.CAPS_REPORTED), "main_cap": C.CAP_DAYS}
        t0 = time.time()
        q1(ds, param_sets["train"], results)
        predictions(ds, param_sets, results)
        path = C.RESULTS_DIR / f"engine_a_{ds.key}.json"
        path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nsaved {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
