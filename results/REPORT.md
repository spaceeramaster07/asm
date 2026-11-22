# +20% / −10% barrier trading on Nifty 50 (validated on the S&P 500)

**Hypothesis.** A biased random walk, geometric Brownian motion (GBM) and Merton jump-diffusion describe
daily log prices well enough to (a) predict a barrier trade, *buy at the close, sell at the first close
+20% (win) or −10% (stop) from entry, or after 504 trading days (timeout)*, and (b) size bets in a
repeated version of that trade.

**Setup.** Everything is out of sample. Every model is refit on the first trading day of each month on
the trailing **5 years** of daily returns and only ever predicts later days. The training years serve as
burn-in and as the place where the window length was chosen. There are no frozen or hindsight parameters.

| | Burn-in / window choice | Test period | Monthly fits | Cash rate |
|---|---|---|---|---|
| **Nifty 50** | 2007-09 → 2016 | 2017 → 2025 | 108 | 6.0% |
| **S&P 500** | 1962 → 1993 | 1994 → 2025 | 384 | 2.5% |

All three models match each window's daily mean and variance exactly, so differences between them come
from the *shape* of returns (two-point lattice, normal, normal plus jumps). Full tables: [TABLES.md](TABLES.md).

## Headline findings

1. **The code matches the theory** (Q1). Simulation agrees with every exact answer (|z| ≤ 1.6),
   including a new exact daily-close solver for GBM and Merton. The textbook continuous formula is
   biased for daily checking (z = +21 to +68).
2. **Merton fits daily returns clearly better than GBM, but neither captures volatility clustering** (Q2).
   Out of sample, Merton gains +0.19 (Nifty) and +0.15 (S&P) log-likelihood per day, with CIs excluding 0,
   mostly in crisis years. Tails are fixed: days beyond 4σ fall from 0.86% to 0.05% on Nifty. Clustering
   remains for both models (Ljung–Box p ≈ 0; |z| autocorrelation 0.26 to 0.35).
3. **For the barrier trade itself, the models beat having no model, but cannot be told apart** (Q3). On
   the S&P 500 all three models beat trailing outcome frequencies (log score +0.31 to +0.33 per start
   day, CI excludes 0); on Nifty the gain (+0.04) is inside the noise of 9 years. Paired differences
   between the models are all within their CIs. All models get exit timing wrong in the same way:
   too few very fast and very slow exits, the signature of clustering. Only Merton reproduces the stop-loss
   fills (−11.1% predicted vs −12.0% / −11.4% real; 1% tail −17% vs −17%).
4. **The Kelly bet these models recommend is too noisy to use** (Q4–Q5). One standard error of 5-year
   drift moves f\* by about 3 (Nifty 3.5, S&P 2.8). In worlds where the true f\* is known (1.1 Nifty,
   1.75 S&P), the trader's estimate ranges from 0 to about 3× the truth (10th to 90th percentile), which
   costs Full Kelly 3.5%/yr (Nifty) and 2.3%/yr (S&P) of growth.
5. **Betting less wins.** In every world, Half Kelly beats Full Kelly (by 1.5 to 1.7%/yr on Nifty, 0.2
   to 0.6%/yr on the S&P) with far smaller drawdowns, and over-betting (1.25× to 2×) is worse still.
   **Merton Half Kelly is the best model-based strategy in every world**, including resampled real returns
   (+0.18 to +0.20%/yr over GBM Half Kelly). **Buy & hold beats every barrier strategy by about 2%/yr**,
   at the cost of 54 to 61% median drawdowns.
6. **The models are useful but not sufficient.** Model-based Kelly beats model-free (empirical) Kelly in
   every world, so the models extract more from 5 years of data than raw outcomes do. Strategy rankings in
   the GBM and Merton worlds largely match the bootstrap world built from real returns, but the bootstrap
   world has higher drawdowns, the cost of the clustering the models miss.

---

## Q1: Does the code match the theory? **Yes.**

Parameters: the rolling fit at the first test date. 500,000 uncapped simulated trades per model.

| Comparison | Nifty z (win / days) | S&P z (win / days) |
|---|---|---|
| Random walk vs exact gambler's-ruin formula | +0.1 / +0.1 | +0.0 / −1.4 |
| Daily GBM vs **exact daily grid** (new) | −0.5 / −0.2 | −0.0 / −0.9 |
| Daily GBM vs BGK-corrected formula | −0.5 / +0.8 | −0.1 / −0.1 |
| Daily GBM vs continuous formula | **+21.5 / +68.4** | **+22.0 / +50.9** |
| Merton vs **exact daily grid** (new) | +0.9 / +0.9 | +1.6 / +1.0 |

- **The grid solver** (`theory.grid_first_passage`) puts the log price on 1,200 cells between the
  barriers and moves it one day at a time using the model's daily distribution. Doubling the grid
  changes P(win) by less than 2×10⁻⁵. It gives Merton, which has no closed form, an exact check.
- **Consistency checks:** Merton with λ = 0 reproduces GBM to machine precision, and so does its simulation
  (z = +0.2 / +0.4). The BGK correction is almost exact for the win rate (error 0.00001) but leaves
  −0.24 days in the expected trade length.

Figures: [q1_convergence_nifty.png](plots/q1_convergence_nifty.png), [q1_convergence_spx.png](plots/q1_convergence_spx.png)

## Q2: Do the models fit daily returns? **Merton is better than GBM; neither captures clustering.**

Each day's return is scored with the latest fit made *before* that day (2,219 Nifty days, 8,053 S&P).
Residuals are probability-integral transformed, z = Φ⁻¹(F_model(r)), which is i.i.d. N(0,1) if the model
is right.

| | Nifty GBM | Nifty Merton | S&P GBM | S&P Merton | Normal |
|---|---|---|---|---|---|
| Log-likelihood per day | 3.095 | **3.284** | 2.964 | **3.112** | |
| Days with \|z\| > 4 | 0.86% | **0.05%** | 1.07% | **0.10%** | 0.006% |
| Excess kurtosis of z | 11.5 | **0.7** | 7.0 | **0.5** | 0 |
| Autocorrelation of \|z\|, lag 1 | 0.35 | 0.26 | 0.30 | 0.24 | 0 |
| Ljung–Box on z² (10 lags), p | ≈0 | ≈0 | ≈0 | ≈0 | |

- **Merton − GBM:** +0.189 per day on Nifty (95% block-bootstrap CI +0.038 to +0.467) and +0.149 on the
  S&P (+0.051 to +0.283). Merton wins on 72 to 76% of days, and most of the total comes from crises
  (2020 on Nifty; 2008 and 2020 on the S&P).
- **What remains is clustering:** both models fail Ljung–Box and ARCH-LM by a wide margin. Merton's
  "jumps" bunch in stormy years (half of Nifty's test-period jumps fall in 2020), so it approximates
  clustering by adding fat tails, not by changing volatility over time.
- **The random walk is not scored:** its daily return takes only the values ±dx, so any real return has
  zero probability.
- **Window length** (training period only): 5 years scores best for both models on both markets
  (3 and 7 years are lower), so the 5-year window was not tuned on test data.

Figures: [q2_loglik](plots/q2_loglik_nifty.png), [q2_qq](plots/q2_qq_nifty.png), [q2_clustering](plots/q2_clustering_nifty.png) (and `_spx`)

## Q3: Do the models predict the barrier trade? **Better than no model; not distinguishable from each other.**

For every start day with a full 504-day window (1,716 Nifty, 7,550 S&P), the month's rolling fit gives an
exact capped forecast. Scores are averaged over start days with 252-day block-bootstrap CIs (start days
overlap). The no-model baseline uses outcome frequencies of past start days that had already resolved.

| | Nifty P(win) | Nifty log score vs no model | S&P P(win) | S&P log score vs no model |
|---|---|---|---|---|
| Real | 57.5% | | 53.6% | |
| Random walk | 60.4% | +0.045 [−0.18, +0.25] | 54.8% | +0.314 [+0.13, +0.52] |
| GBM | 61.2% | +0.035 [−0.22, +0.26] | 55.2% | +0.324 [+0.13, +0.55] |
| Merton | 62.1% | +0.042 [−0.22, +0.28] | 55.6% | **+0.331 [+0.13, +0.56]** |

- **Models vs no model:** a clear gain on the S&P (30 years). On Nifty, 9 years of overlapping trades are
  too few to separate anything.
- **Models vs each other:** Merton − GBM is +0.007 per day on both markets, with CIs covering 0. Because
  win rate is driven by drift and volatility, which all three share, the barrier trade cannot rank them.
- **Calibration is weak:** months where the models predict a high P(win) do not win more often. That is
  the 5-year drift estimate being noisy, not a fault of any one model.
- **Exit timing:** the PIT histogram of the exit day has too much mass at both ends: real trades are more
  often much faster or much slower than predicted (S&P: 5 to 6 of 10 bins outside their CI; Nifty: 1).
  This is volatility clustering again: storms end trades fast, calm stretches drag them out.
- **Stop-loss fills** (input to the betting game):

  | | Nifty mean | Nifty 1% worst | S&P mean | S&P 1% worst |
  |---|---|---|---|---|
  | Real | −12.0% [−12.8, −11.0] | −17.3% | −11.4% [−12.0, −11.0] | −17.2% |
  | Random walk | −10.0% | −10.5% | −10.0% | −10.6% |
  | GBM | −10.6% | −12.1% | −10.7% | −12.6% |
  | Merton | **−11.2%** | **−17.4%** | **−11.1%** | **−17.0%** |

Figures: [q3_scores](plots/q3_scores_nifty.png), [q3_calibration](plots/q3_calibration_nifty.png), [q3_pit](plots/q3_pit_nifty.png), [q3_stopfill](plots/q3_stopfill_nifty.png) (and `_spx`)

## Q4: What bet size does each model recommend? **About 2× on average, but it swings wildly.**

Per monthly fit, Kelly maximises expected log growth of one round, with cash earning r_f, borrowing at
r_f + 1%, and 0.10% trading cost each way.

| | Nifty GBM | Nifty Merton | S&P GBM | S&P Merton |
|---|---|---|---|---|
| f\* mean (range) | 2.24 (0 – 4.2) | 2.16 (0 – 3.5) | 2.52 (0 – 5.6) | 2.27 (0 – 4.8) |
| Months with f\* > 3× cap | 18% | 17% | 44% | 43% |
| Months with f\* = 0 | 3% | 3% | 22% | 22% |
| Half-Kelly share of excess growth | 76% | 76% | 75% | 75% |
| Cash break-even / f\* | 1.73 | 1.36 | 1.59 | 1.22 |
| Ruin cliff (1-in-a-million loss wipes out) | 7.0× | **3.5×** | 7.0× | **4.3×** |

- **Drift uncertainty dominates:** ±1 standard error of the 5-year drift moves GBM's f\* by 3.5 (Nifty)
  and 2.8 (S&P) on average. The S&P series swings between 0 (2002–2006, 2008–2012) and above 5.
- **Merton bets less** and has an earlier break-even and cliff, because its jumps make big stop-loss gaps
  possible. With the real fills swapped in, GBM's f\* falls from 2.24 to 1.85 on Nifty.
- **Half Kelly keeps about 75% of the excess growth** for half the leverage, the same as the textbook
  result, in every model and market.
- The Q5 traders read f\* from a lookup table. It matches direct computation within 0.23 (Nifty) and
  0.08 (S&P) inside the 3× cap.

Figures: [q4_growth](plots/q4_growth_nifty.png), [q4_fstar_series](plots/q4_fstar_series_nifty.png) (and `_spx`)

## Q5: Which betting strategy wins? **Bet less than Kelly; Merton Half Kelly is the best model-based rule.**

**Design.** A career is one continuous daily price path: 5 years of visible history, then 100 back-to-back
trades. Before each trade the trader refits on the path's own last 5 years and every strategy sets its
bet (capped at 3×). Wealth is marked to market daily; levered positions face a margin call (equity below
25% of the position) and bankruptcy is absorbing. 2,000 careers per world, the same paths for every
strategy. Worlds:

- **Random walk, GBM, Merton:** fixed true parameters from the full history, never shown to the trader.
- **Bootstrap (headline):** real daily returns resampled in blocks averaging 3 months, which keeps real
  crashes, fat tails and clustering.
- **Real path:** the actual test period from 60 start offsets (12 trades on Nifty, 48 on the S&P).

**Bootstrap world after 100 rounds** (mean log growth per year; difference vs GBM Half Kelly, 95% CI):

| Strategy | Nifty growth | vs Half Kelly | Nifty median MDD | S&P growth | vs Half Kelly | S&P median MDD |
|---|---|---|---|---|---|---|
| Buy & hold | 9.8% | +1.98% | 61% | 7.0% | +2.13% | 54% |
| Fixed 50% | 8.2% | +0.34% | 31% | 4.8% | −0.13% | 30% |
| **Merton Half Kelly** | **8.0%** | **+0.18%** | 56% | **5.1%** | **+0.20%** | 65% |
| GBM Half Kelly | 7.8% | 0 | 64% | 4.9% | 0 | 73% |
| Drawdown-controlled Half Kelly | 7.9% | +0.06% | 51% | 4.6% | −0.31% | 59% |
| Vol-targeted Half Kelly | 7.6% | −0.21% | 65% | 4.7% | −0.19% | 74% |
| Empirical Half Kelly | 7.3% | −0.48% | 75% | 4.3% | −0.60% | 77% |
| Martingale | 7.1% | −0.76% | 74% | 4.1% | −0.81% | 69% |
| GBM Kelly | 6.2% | −1.66% | 86% | 4.3% | −0.58% | 89% |
| Merton Kelly | 6.3% | −1.55% | 86% | 4.3% | −0.58% | 89% |
| Empirical Kelly | 5.9% | −1.92% | 89% | 3.9% | −0.99% | 89% |
| GBM 2× Kelly | 4.8% | −3.01% | 93% | 3.9% | −1.03% | 94% |
| Shrinkage Kelly | 6.6% | −1.25% | 74% | 3.5% | −1.40% | 79% |
| Flat 50 | 6.5% | −1.28% | 15% | 3.3% | −1.59% | 15% |

(Cash: 5.8% Nifty, 2.5% S&P in log terms. Every 95% CI half-width is 0.15 points or less, so differences
of about 0.2 points or more are real. No strategy goes bankrupt: the 3× cap and the −10% stop prevent it.)

- **Estimation noise is the main cost.** In the model worlds the trader's f\* is between 0 and about 3.3×
  the true f\* (10th to 90th percentile). That costs Full Kelly 3.5%/yr on Nifty and 2.3%/yr on the S&P,
  and Half Kelly 1.7 to 2.1%/yr ([q5_misestimation](plots/q5_misestimation_nifty.png)). The Kelly frontier
  bends backwards past 0.5×: more leverage gives less growth and deeper drawdowns
  ([q5_frontier](plots/q5_frontier_nifty.png)).
- **Buy & hold wins** because it holds about the right exposure (f = 1, close to the true f\* of 1.1 on
  Nifty) without estimation noise, costs or exits. **Fixed 50%** wins on Nifty for the same reason. Any
  strategy that leans on the 5-year drift estimate pays for its noise.
- **Merton beats GBM as a sizing tool** (Half Kelly, every world, CIs excluding 0), because it bets less
  when jumps make stops gap. **Models beat no model:** empirical Kelly trails model Kelly in every world.
- **The robust add-ons mostly don't help.** Vol targeting at 60 days loses about 0.2%/yr. Drawdown control
  cuts drawdowns but improves growth only in the Nifty bootstrap world (+0.06%). Positive-part shrinkage pulls the drift to the cash rate so
  often that it sits in cash and loses 1.3%/yr.
- **Martingale** is worse than Half Kelly in every world and lands at the 23rd percentile on the real S&P
  path. **Anti-martingale** does well in the bootstrap world (+0.18% Nifty) but finishes at the 11th
  percentile on the real S&P.
- **Do the worlds agree?** Mostly. The GBM and Merton worlds rank strategies like the bootstrap world
  ([q5_worlds](plots/q5_worlds_nifty.png)), so the models are good enough to *compare* strategies. The
  bootstrap world has higher drawdowns (Nifty buy & hold: P(drawdown > 50%) 88% vs 65 to 68% in the
  model worlds), which is the cost of the missing clustering.
- **Real path.** On Nifty (12 trades, 2017–2025) every strategy lands at the 60th to 76th percentile of
  bootstrap careers, a good decade. On the S&P (48 trades) GBM, Merton and drawdown-controlled Half Kelly
  sit at the 62nd to 72nd percentile (vol-targeted Half Kelly at the 46th), while Full Kelly and above land
  at the 25th to 31st percentile with 95 to 99% drawdowns.

Figures: [q5_leaderboard](plots/q5_leaderboard_nifty.png), [q5_worlds](plots/q5_worlds_nifty.png), [q5_frontier](plots/q5_frontier_nifty.png), [q5_misestimation](plots/q5_misestimation_nifty.png) (and `_spx`)

---

## Caveats

- **Nifty's test sample is short:** 9 years, 1,716 overlapping start days (about 4 to 5 independent
  2-year windows) and only 12 real trades. Most Nifty Q3 comparisons are inconclusive for that reason, and
  both real paths are single histories, so they reflect luck as much as skill.
- **The 5-year window drives the betting results.** It suits volatility (Q2 picks it), but its drift
  estimate is uncertain by about ±7 to 8%/yr, and that noise is behind "Full Kelly loses" and much of
  "buy & hold wins". Estimating drift on all past data (an expanding window) while keeping volatility on
  5 years might shrink those gaps. That is untested, and the model and bootstrap worlds would favour it by
  construction (their long-run drift is constant), so the real path is its fair test.
- **Merton's Kelly table fixes the jump sizes** at their training-period estimate. Only drift, volatility
  and jump rate are re-estimated each round.
- **Kelly ignores outcomes with probability below 10⁻⁹ per trade.** Otherwise the open-ended overshoot
  bin of the grid would set g = −∞ at moderate leverage. The ruin cliff is reported separately.
- **Margin calls almost never bind:** at a 3× cap and 25% maintenance, they need a one-day gap worse
  than about −11%; the −10% stop exits first otherwise.
- **Model worlds use one fixed parameter set** (the full-history fit), so they contain no regime change;
  the bootstrap world contains only crashes already seen in history.
- **The cash rate is constant** (6.0% / 2.5%) and the borrowing spread, costs and margin rule are
  assumptions in `barrier/config.py`.
- **Not tested yet:** GARCH(1,1) with Student-t errors. A prototype beat both GBM and Merton on Q2's
  likelihood (+0.28 / +0.30 per day vs GBM) and removed most clustering (see `PLAN.md`, "Later").

## How to run

```bash
.venv/Scripts/python run_calibrate.py      # data checks + monthly rolling fits
.venv/Scripts/python run_q1_theory.py      # ~1 min
.venv/Scripts/python run_q2_returns.py     # ~15 s
.venv/Scripts/python run_q3_barriers.py    # ~4 min first time (caches forecasts)
.venv/Scripts/python run_q4_kelly.py       # ~8 min first time (caches Kelly tables)
.venv/Scripts/python run_q5_tournament.py  # ~15 min (8 worlds in parallel)
.venv/Scripts/python run_tables.py
.venv/Scripts/python -m pytest -q tests
```

To change only the look of a plot, `run_q1_theory.py plots`, `run_q4_kelly.py plots` and
`run_q5_tournament.py plots` redraw from the saved JSON in seconds; Q2 and Q3 rerun in under a minute
from the caches.
