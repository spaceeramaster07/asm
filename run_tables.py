"""Build the final tables from the saved JSON results (after run_q1 .. run_q5).

Run:  .venv/Scripts/python run_tables.py
Writes results/tables/<question>_<dataset>.md/.csv and results/TABLES.md.
"""
from __future__ import annotations

import json

import pandas as pd

from barrier import config as C

TABLE_DIR = C.RESULTS_DIR / "tables"
MODELS = ("RandomWalk", "GBM", "Merton")
WORLDS = ("Bootstrap", "Merton", "GBM", "RandomWalk")


def load(name):
    return json.loads((C.RESULTS_DIR / name).read_text())


def pct(x, d=1):
    return "-" if x is None or x != x else f"{x * 100:.{d}f}%"


def sgn(x, d=2):
    return "-" if x is None or x != x else f"{x * 100:+.{d}f}%"


def ci(v, d=3):
    return f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]"


def q1(ds):
    r = load(f"q1_{ds.key}.json")
    rows = [{"Comparison": c["label"], "Theory P(win)": f"{c['p_ref']:.5f}", "Sim P(win)": f"{c['sim_p']:.5f}",
             "z (win)": f"{c['z_p']:+.1f}", "Theory E[days]": f"{c['t_ref']:.2f}", "Sim E[days]": f"{c['sim_t']:.2f}",
             "z (days)": f"{c['z_t']:+.1f}"} for c in r["comparisons"].values()]
    return pd.DataFrame(rows)


def q2(ds):
    r = load(f"q2_{ds.key}.json")
    rows = []
    for m, s in r["models"].items():
        rows.append({"Model": m, "Log-lik/day": f"{s['loglik_mean']:.4f}", "|z|>3": pct(s["share_gt3"], 2),
                     "|z|>4": pct(s["share_gt4"], 3), "Excess kurtosis of z": f"{s['exkurt']:.2f}",
                     "JB p": f"{s['jb_p']:.1e}", "Ljung-Box z² p": f"{s['ljung_box_z2_p']:.1e}",
                     "ARCH-LM p": f"{s['arch_lm_p']:.1e}", "Berkowitz p": f"{s['berkowitz_p']:.1e}",
                     "ACF |z| lag 1": f"{s['acf_abs_z_lag1']:+.3f}"})
    md = r["merton_minus_gbm"]
    rows.append({"Model": "Merton − GBM", "Log-lik/day": f"{md['mean_per_day']:+.4f} {ci(md['ci95'])}"})
    rows.append({"Model": "Random walk", "Log-lik/day": "not scored (two-point distribution)"})
    return pd.DataFrame(rows)


def q3(ds):
    r = load(f"q3_{ds.key}.json")
    base = "No model (trailing frequencies)"
    rows = [{"Forecaster": "Real outcome", "P(win)": pct(r["real"]["p_win"]),
             "Mean days": f"{r['real']['tau_mean']:.0f}"}]
    for m, s in r["models"].items():
        vb = s.get(f"vs {base}")
        vg = s.get("vs GBM")
        rows.append({"Forecaster": m, "P(win)": pct(s["pred_p_win"]),
                     "Mean days": f"{s['tau_pred_mean']:.0f}" if "tau_pred_mean" in s else "-",
                     "Log score": f"{s['log_score']:.3f}",
                     "vs no model": f"{vb['log_diff']:+.3f} {ci(vb['log_diff_ci'])}" if vb else "-",
                     "vs GBM": f"{vg['log_diff']:+.4f} {ci(vg['log_diff_ci'], 4)}" if vg else "-",
                     "Brier": f"{s['brier']:.3f}",
                     "PIT bins outside CI": f"{s['pit_bins_outside_ci']}/10" if "pit_bins_outside_ci" in s else "-"})
    sf = r["stop_fill"]
    fill = pd.DataFrame([{"Source": "Real", "Mean fill": pct(sf["real"]["mean"], 2),
                          "95% CI": f"[{pct(sf['real']['mean_ci'][0], 2)}, {pct(sf['real']['mean_ci'][1], 2)}]",
                          "1% worst": pct(sf["real"]["p01"], 2)}]
                        + [{"Source": m, "Mean fill": pct(v["mean"], 2), "1% worst": pct(v["p01"], 2)}
                           for m, v in sf["models"].items()])
    return pd.DataFrame(rows), fill


def q4(ds):
    r = load(f"q4_{ds.key}.json")
    rows = []
    for m in MODELS:
        s = r["summary"][m]
        rows.append({"Model": m, "f* mean": f"{s['f_star_mean']:.2f}", "f* range": f"{s['f_star_min']:.2f}–{s['f_star_max']:.2f}",
                     "Months f* > 3x": pct(s["share_months_f_above_cap"], 0), "Months f* = 0": pct(s["share_months_f_zero"], 0),
                     "Excess growth over cash /yr": sgn(s["excess_growth_mean"]),
                     "Half-Kelly share": pct(s["half_share_median"], 0),
                     "Cash break-even / f*": f"{s['breakeven_over_fstar_median']:.2f}" if s["breakeven_over_fstar_median"] else "-",
                     "Ruin cliff": f"{s['ruin_cliff_median']:.1f}x"})
    return pd.DataFrame(rows)


def q5(ds, k):
    r = load(f"q5_{ds.key}.json")
    order = sorted(r["strategies"], key=lambda n: -r["worlds"]["Bootstrap"]["metrics"][f"K{k}"][n]["log_growth_mean"])
    rows = []
    for n in order:
        b = r["worlds"]["Bootstrap"]["metrics"][f"K{k}"][n]
        row = {"Strategy": n, "Median W": f"{b['W_median']:,.0f}", "Median CAGR": sgn(b["cagr_median"], 1),
               "Mean log growth": sgn(b["log_growth_mean"]),
               f"vs {r['reference']}": f"{sgn(b['vs_ref_log_growth'])} [{sgn(b['vs_ref_ci'][0])}, {sgn(b['vs_ref_ci'][1])}]",
               "Median MDD": pct(b["mdd_median"], 0), "P(MDD>50%)": pct(b["p_mdd_gt_50"], 0),
               "P(bust)": pct(b["p_bust"], 1), "Sharpe": f"{b['sharpe_median']:.2f}"}
        for w in WORLDS[1:]:
            row[f"{w} world log growth"] = sgn(r["worlds"][w]["metrics"][f"K{k}"][n]["log_growth_mean"])
        rl = r["real"]["strategies"][n]
        row["Real path CAGR"] = f"{sgn(rl['cagr_offset0'], 1)} ({sgn(rl['cagr_range'][0], 1)}..{sgn(rl['cagr_range'][1], 1)})"
        pc = r["real_percentile_in_bootstrap"].get(n)
        row["Real pct in bootstrap"] = "-" if pc is None else f"{pc:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for old in TABLE_DIR.glob("table*"):
        old.unlink()
    md = ["# Tables\n", "Generated by `run_tables.py` from `results/q*_<dataset>.json`.\n"]
    for ds in C.DATASETS.values():
        md.append(f"\n## {ds.label}\n")
        t3, fill = q3(ds)
        tables = [("q1", "Q1: theory vs simulation", q1(ds)), ("q2", "Q2: fit to daily returns", q2(ds)),
                  ("q3", "Q3: barrier outcome forecasts", t3), ("q3_fills", "Q3: stop-loss fills", fill),
                  ("q4", "Q4: Kelly bet size (monthly rolling fits)", q4(ds))]
        tables += [(f"q5_K{k}", f"Q5: strategies after {k} rounds (bootstrap world; other worlds and real path)",
                    q5(ds, k)) for k in C.K_REPORT]
        for key, title, df in tables:
            df = df.fillna("")
            df.to_csv(TABLE_DIR / f"{key}_{ds.key}.csv", index=False)
            text = df.to_markdown(index=False)
            (TABLE_DIR / f"{key}_{ds.key}.md").write_text(text, encoding="utf-8")
            md += [f"\n### {title}\n", text, ""]
    (C.RESULTS_DIR / "TABLES.md").write_text("\n".join(md), encoding="utf-8")
    print(f"saved {C.RESULTS_DIR / 'TABLES.md'} and {TABLE_DIR}")


if __name__ == "__main__":
    main()
