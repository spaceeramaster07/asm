"""Step 5: build the final tables from the saved JSON results.

Run:  .venv/Scripts/python run_step5_tables.py    (after Steps 1-4)
Writes results/tables/table1_<dataset>.md/.csv, table2_<dataset>.md/.csv and results/TABLES.md.
"""
from __future__ import annotations

import json

import pandas as pd

from barrier import config as C

TABLE_DIR = C.RESULTS_DIR / "tables"
WORLDS = ("RandomWalk", "GBM", "Merton")
WNAME = {"RandomWalk": "Random walk", "GBM": "GBM", "Merton": "Merton"}


def load(name):
    return json.loads((C.RESULTS_DIR / name).read_text())


def pct(x, d=1):
    return "-" if x is None or x != x else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "-" if x is None or x != x else f"{x:,.{d}f}"


def table1(ds) -> pd.DataFrame:
    cal = load(f"calibration_{ds.key}.json")["train"]
    ea = load(f"engine_a_{ds.key}.json")["predictions"]
    real = load(f"real_{ds.key}.json")
    rows = []
    g, rw = cal["GBM"]["theory"], cal["RandomWalk"]["theory"]
    for src, p, t, stop in [
        ("Theory: GBM continuous (no cap)", g["p_win_continuous"], g["E_tau_days_continuous"], -0.10),
        ("Theory: GBM daily closes, BGK (no cap)", g["p_win_daily_bgk"], g["E_tau_days_daily_bgk"],
         g["expected_stop_return_daily"]),
        ("Theory: random walk lattice (no cap)", rw["p_win_exact_lattice"], rw["E_tau_days_exact_lattice"],
         rw["expected_stop_return_daily"]),
    ]:
        rows.append({"Source": src, "Win": pct(p), "Ruin": pct(1 - p), "Timeout": "-",
                     "Mean days": num(t), "Median": "-", "SD": "-", "p5-p95": "-",
                     "Stop fill": pct(stop, 2), "Stop fill worst 1%": "-", "KS p vs chain": "-"})
    for params, tag in (
        ("train", "Sim, static training params"),
        ("rolling", "Sim, 5y rolling params (out-of-sample)"),
        ("oracle", "Sim, test params (oracle)"),
    ):
        for m in WORLDS:
            s = ea[params][m][f"cap{C.CAP_DAYS}"]
            sc = real["models"][f"{params}/{m}"]
            ok = lambda b: "" if b else " *"  # noqa: E731
            rows.append({
                "Source": f"{tag}: {WNAME[m]}",
                "Win": pct(s["p_win"]) + ok(sc["win_in_ci"]), "Ruin": pct(s["p_ruin"]),
                "Timeout": pct(s["p_timeout"]),
                "Mean days": num(s["tau_mean"]) + ok(sc["tau_in_ci"]),
                "Median": num(s["tau_median"]), "SD": num(s["tau_sd"]),
                "p5-p95": f"{s['tau_p05']:.0f}-{s['tau_p95']:.0f}",
                "Stop fill": pct(s["ret_mean_ruin"], 2) + ok(sc["ruin_ret_in_ci"]),
                "Stop fill worst 1%": pct(s["ret_p01_ruin"], 1),
                "KS p vs chain": f"{sc['ks_p_chain0']:.2f} (med {sc['ks_p_median_offsets']:.2f})",
            })
    ov, ci = real["overlap_close"], real["overlap_close"]["ci"][str(C.BOOT_BLOCK_MAIN)]
    rows.append({
        "Source": f"REAL overlapping, close ({ov['n']} starts)",
        "Win": f"{pct(ov['p_win'])} [{pct(ci['p_win'][0], 0)}-{pct(ci['p_win'][1], 0)}]",
        "Ruin": pct(ov["p_ruin"]), "Timeout": pct(ov["p_timeout"]),
        "Mean days": f"{ov['tau_mean']:.0f} [{ci['tau_mean'][0]:.0f}-{ci['tau_mean'][1]:.0f}]",
        "Median": num(ov["tau_median"]), "SD": num(ov["tau_sd"]),
        "p5-p95": f"{ov['tau_p05']:.0f}-{ov['tau_p95']:.0f}",
        "Stop fill": f"{pct(ov['ret_mean_ruin'], 2)} [{pct(ci['ret_mean_ruin'][0], 1)} to "
                     f"{pct(ci['ret_mean_ruin'][1], 1)}]",
        "Stop fill worst 1%": pct(ov["ret_min_ruin"], 1) + " (min)", "KS p vs chain": "-"})
    oi = real["overlap_intraday"]
    rows.append({
        "Source": "REAL overlapping, intraday High/Low + Open gaps",
        "Win": pct(oi["p_win"]), "Ruin": pct(oi["p_ruin"]), "Timeout": pct(oi["p_timeout"]),
        "Mean days": num(oi["tau_mean"]), "Median": num(oi["tau_median"]), "SD": num(oi["tau_sd"]),
        "p5-p95": f"{oi['tau_p05']:.0f}-{oi['tau_p95']:.0f}",
        "Stop fill": pct(oi["ret_mean_ruin"], 2), "Stop fill worst 1%": pct(oi["ret_min_ruin"], 1) + " (min)",
        "KS p vs chain": "-"})
    ch, sp = real["chain_close_offset0"], real["chain_spread"]
    rows.append({
        "Source": f"REAL non-overlapping chain ({ch['n']} trades)",
        "Win": f"{pct(ch['p_win'])} (start-day range {pct(sp['p_win']['min'], 0)}-{pct(sp['p_win']['max'], 0)})",
        "Ruin": pct(ch["p_ruin"]), "Timeout": pct(ch["p_timeout"]),
        "Mean days": num(ch["tau_mean"]), "Median": num(ch["tau_median"]), "SD": num(ch["tau_sd"]),
        "p5-p95": f"{ch['tau_p05']:.0f}-{ch['tau_p95']:.0f}",
        "Stop fill": pct(ch["ret_mean_ruin"], 2), "Stop fill worst 1%": pct(ch["ret_min_ruin"], 1) + " (min)",
        "KS p vs chain": "-"})
    return pd.DataFrame(rows)


def table2(ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = load(f"tournament_{ds.key}.json")["q6"]
    K = tr["K_real"]
    out = []
    for K_ in (K, C.K_LONG):
        rows = []
        strategies = list(tr["worlds"][f"GBM/K{K_}"].keys())
        for s in strategies:
            r = {"Strategy": s}
            f = tr["worlds"][f"GBM/K{K_}"][s]["f"]
            r["f"] = "-" if f is None else f"{f:.2f}"
            for w in WORLDS:
                m = tr["worlds"][f"{w}/K{K_}"][s]
                r[f"{WNAME[w]}: median W (CAGR)"] = f"{m['W_median']:.0f} ({pct(m['cagr_median'])})"
                r[f"{WNAME[w]}: median MDD"] = pct(m["mdd_median"], 0)
                r[f"{WNAME[w]}: P(lose half)"] = pct(m["p_lose_half"], 1)
            if K_ == K:
                m = tr["real"]["offset0"][s]
                lo, hi = tr["real"]["W_range_offsets"][s]
                r["REAL: W (CAGR)"] = f"{m['W_median']:.0f} ({pct(m['cagr_median'])})"
                r["REAL: W range over start days"] = f"{lo:.0f}-{hi:.0f}"
                r["REAL: MDD"] = pct(m["mdd_median"], 0)
                r["REAL: percentile (RW/GBM/Merton)"] = " / ".join(
                    f"{tr['real_percentile'][w][s]:.0f}" for w in WORLDS)
            rows.append(r)
        out.append(pd.DataFrame(rows))
    return out[0], out[1]


def table2_spectrum(ds) -> pd.DataFrame:
    tr = load(f"tournament_{ds.key}.json")["q6"]
    spec = tr.get("fractional_spectrum", {})
    if not spec:
        return pd.DataFrame()
    rows = []
    for kind in ("Static", "Rolling"):
        for mult in spec["multipliers"]:
            k = f"{kind} {mult:.2f}x"
            mg = spec["worlds"]["GBM"][k]
            mm = spec["worlds"]["Merton"][k]
            mr = spec["real"][k]
            rows.append({
                "Kelly mode": f"{kind} ({mult:.2f}x f*)",
                "Effective f": f"{mg['f']:.2f}",
                "GBM: median W (CAGR)": f"{mg['W_median']:.0f} ({pct(mg['cagr_median'])})",
                "GBM: median MDD": pct(mg["mdd_median"], 0),
                "Merton: median W (CAGR)": f"{mm['W_median']:.0f} ({pct(mm['cagr_median'])})",
                "Merton: median MDD": pct(mm["mdd_median"], 0),
                "Merton: P(lose half)": pct(mm["p_lose_half"], 1),
                "REAL: W (CAGR)": f"{mr['W_median']:.0f} ({pct(mr['cagr_median'])})",
                "REAL: MDD": pct(mr["mdd_median"], 0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# Final tables\n",
          "`*` after a model value = outside the real 95% block-bootstrap interval "
          f"(block {C.BOOT_BLOCK_MAIN} start-days). All simulated and real numbers use the "
          f"{C.CAP_DAYS}-day cap unless marked 'no cap'.\n"]
    for ds in C.DATASETS.values():
        t1 = table1(ds)
        t2, t2l = table2(ds)
        t2s = table2_spectrum(ds)
        K = load(f"tournament_{ds.key}.json")["q6"]["K_real"]
        for name, df in (("table1", t1), ("table2", t2), ("table2_long", t2l), ("table2_spectrum", t2s)):
            if not df.empty:
                df.to_csv(TABLE_DIR / f"{name}_{ds.key}.csv", index=False)
                (TABLE_DIR / f"{name}_{ds.key}.md").write_text(df.to_markdown(index=False))
        md += [f"\n## {ds.label}\n",
               f"\n### Table 1 - Model accuracy (test period {ds.test[0][:4]}-{ds.test[1][:4]})\n\n",
               t1.to_markdown(index=False),
               f"\n\n### Table 2 - Tournament leaderboard, {K} rounds (= real chain length), "
               f"{C.N_CAREERS} careers per world, r_f {ds.rf:.1%}\n\n",
               t2.to_markdown(index=False),
               f"\n\n### Table 2b - Long run, {C.K_LONG} rounds\n\n",
               t2l.to_markdown(index=False), "\n"]
        if not t2s.empty:
            md += [f"\n### Table 2c - Fractional Kelly spectrum ({K} rounds, Static vs 5y Rolling)\n\n",
                   t2s.to_markdown(index=False), "\n"]
    path = C.RESULTS_DIR / "TABLES.md"
    path.write_text("".join(md), encoding="utf-8")
    print(f"saved {path} and {TABLE_DIR}")


if __name__ == "__main__":
    main()
