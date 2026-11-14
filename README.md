# Stochastic Models of a Take-Profit / Stop-Loss Trade and a Kelly Betting Game

Applied Stochastic Models (ASM) Project, 2026 — Sambhav Manohar (S20240020210), Venkat Rahul
(S20240020220), Sham Gagan Siddharth (S20240030401).

A +20% / −10% barrier trade on the Nifty 50 (with the S&P 500 as validation) is modelled with a biased
random walk, geometric Brownian motion and Merton jump-diffusion. The project derives and checks the
theory, tests the models out of sample on real data, and uses them to size bets in a 100-trade Kelly
betting game.

## Contents

| Path | What it is |
|---|---|
| `barrier/` | The Python package: models and calibration, closed forms and the Markov-chain solver, Monte Carlo simulator, real-data scanner, statistical tests, Kelly calculations, bootstrap and betting game |
| `run_calibrate.py` … `run_tables.py` | The pipeline scripts, run in the order below |
| `data/nifty_raw.csv`, `data/spx_raw.csv` | Frozen daily prices (Yahoo Finance). Committed so every result is reproducible |
| `tests/` | Automated checks (formulas, solver, no look-ahead, bootstrap, cost and margin arithmetic) |
| `results/` | Created by the scripts: `q1`–`q5` JSON results, `plots/`, `tables/` |
| `requirements.txt` / `requirements-lock.txt` | Python packages (loose / exact versions used) |

## Reproduce everything

Requires Python 3.12. From the project folder:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt      # on macOS/Linux: .venv/bin/python
.venv/Scripts/python run_calibrate.py       # data checks + monthly rolling fits          (~10 s)
.venv/Scripts/python run_q1_theory.py       # theory vs simulation                         (~1 min)
.venv/Scripts/python run_q2_returns.py      # fit to daily returns                         (~15 s)
.venv/Scripts/python run_q3_barriers.py     # barrier-trade forecasts; builds a cache      (~4 min first time)
.venv/Scripts/python run_q4_kelly.py        # Kelly bet sizes; builds the Kelly tables     (~10 min first time)
.venv/Scripts/python run_q5_tournament.py   # betting game, 8 markets in parallel          (~10-15 min)
.venv/Scripts/python run_tables.py          # results/tables and results/TABLES.md
.venv/Scripts/python -m pytest -q tests     # 19 checks                                     (~20 s)
```

Fixed random seeds make every number identical on each run. `results/cache/` (forecasts and Kelly
tables, about 130 MB) is created automatically and can be deleted at any time.

To redraw plots without recomputing: `run_q1_theory.py plots`, `run_q4_kelly.py plots`,
`run_q5_tournament.py plots`.

## Report and slides (optional)

The submitted report (`Group10_..._Project_Report.pdf`) and the presentation
(`results/Barrier_Trading_Presentation.pptx`) are included. To rebuild them from the results,
install Node.js 18+ and run from the `report/` folder:

```bash
npm install
npm run build      # Word report  -> results/Barrier_Trading_Stochastic_Report.docx
npm run deck       # PowerPoint   -> results/Barrier_Trading_Presentation.pptx
```
