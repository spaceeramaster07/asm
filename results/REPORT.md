# +20% / −10% barrier trading on Nifty 50 (validated on the S&P 500): results

**Setup.** Buy at the close and exit at the first daily close that is +20% (win) or −10% (ruin) from entry. Any trade still open after 504 trading days is a *timeout*. Three models (biased random walk, GBM, Merton jump-diffusion) are calibrated on a training period and tested on a later, unseen test period:

| | Training (models are fitted here) | Test (out-of-sample) | Risk-free rate (assumed) |
|---|---|---|---|
| Nifty 50 (main) | 2007-09 → 2016 | 2017 → 2025 | 6.0% |
| S&P 500 (validation) | 1962 → 1993 | 1994 → 2025 | 2.5% |

All three models reproduce the training data's daily mean and variance exactly, so any difference between them comes from the *shape* of returns. "Oracle" means the same model refitted on the test period. It is used only to diagnose errors, never to make predictions. Full tables are in [TABLES.md](TABLES.md).

## Headline findings

1. **The theory and the code agree.** Simulations converge to the exact formulas at the expected 1/√N rate, provided the daily-checking correction (BGK) is applied. Without it, the continuous formula is off by about 1.2 points of win rate and 12–18 days.
2. **The drift you estimate matters more than which model you choose.** The three models never differ by more than about 2 points. Training-period drift is so uncertain (Nifty: ±8%/yr) that the forecast win rate could plausibly be anywhere from 27% to 62%.
3. **Real markets are "clumpy" in a way no constant-volatility model can be.** Average trade length depends on how you sample real data: 152 vs 224 days on Nifty, 165 vs 215 on the S&P. In the models both ways give the same answer. This is direct evidence of volatility clustering.
4. **Merton is the only model that gets the stop-loss overshoot right.** But most real overshoot comes from checking prices only at the close. Intraday stops land at about −10.1% to −10.3%.
5. **Kelly works as the theory says. Overconfidence is what breaks it.** Over-betting stops paying more than cash at about 2× Kelly, and Half Kelly keeps 75% of the excess growth. A trader who doubles the true drift loses money, and a 3× leverage cap or Half Kelly prevents that.

---

## Q1: Do the simulations converge to the exact formulas? **Yes.**

Each check uses a pool of 500,000 simulated trades per model, with no time cap.

| Comparison | Nifty: z (win) / z (days) | S&P: z (win) / z (days) |
|---|---|---|
| Random walk vs exact lattice formula | +1.3 / +1.6 | +1.6 / +1.3 |
| Daily GBM vs BGK-corrected formula | −1.1 / +2.3 | +0.3 / −0.5 |
| Daily GBM vs continuous formula | **+17.7 / +102** | **+17.6 / +60** |

- Error falls at rate 1/√N. The 95% confidence intervals contain the exact value 93–96% of the time.
- The continuous formula's error stops shrinking (the plateau in the plot). Checking only once a day adds about 1.2 points of win rate and 12 days (Nifty) or 18 days (S&P) to trade length. BGK removes almost all of this. The one exception is a residual of about 0.3 days on Nifty durations (z = 2.3), which is negligible in practice.
- The draft's random-walk formula had q/p and p/q swapped. The corrected version, (1 − r^A)/(1 − r^(A+B)) with r = q/p, matches a direct numerical solution to 9 decimal places.

Figures: [plots/q1_convergence_nifty.png](plots/q1_convergence_nifty.png), [plots/q1_convergence_spx.png](plots/q1_convergence_spx.png)

## Q2: How likely is +20% before −10%, and how sensitive is it to drift? **Very sensitive.**

- **With no edge you mostly lose:** exactly **1/3** if price has zero drift (the price is then a martingale), or **36.6%** if log price has zero drift.
- **Forecast (training) vs reality (test), win share among trades that hit a barrier:**

  | | GBM forecast from training | Real test period, 95% CI |
  |---|---|---|
  | Nifty | 45.7% | **61%** [31%–89%] |
  | S&P 500 | 58.0% | **58%** [45%–72%] |

- **The training data can't pin the edge down.** The standard error on drift is 8.1%/yr for Nifty and 2.5%/yr for the S&P. Across a ±1.96 SE range of drift, the win rate spans **27%–62% (Nifty)** and **40%–71% (S&P)**.
- The sensitivity curve is steep: in the Nifty plot, moving from the training drift (9.6%) to the test-period drift (14.5%) moves the predicted win rate by about 6–11 points.

Figures: [plots/q2_sensitivity_nifty.png](plots/q2_sensitivity_nifty.png), [plots/q2_sensitivity_spx.png](plots/q2_sensitivity_spx.png)

## Q3: How long until a barrier is hit, and do constant-volatility models capture the spread? **They get the order of magnitude, not the shape.**

| Mean days to exit (504-day cap) | Training forecast (GBM) | Oracle GBM | Real: chain | Real: overlapping |
|---|---|---|---|---|
| Nifty | 96 | 183 | 152 | 224 |
| S&P 500 | 241 | 154 | 165 | 215 |

- **Volatility, not drift, sets the duration.** Training-fitted models get duration wrong in opposite directions on the two markets. Nifty's training period was more volatile than its test period (forecast too short); the S&P's was calmer (forecast too long).
- **Evidence of clustering.** In any constant-volatility model, starting a trade on every day ("overlapping") and starting one only after the previous trade ends ("chain") give the same average length. In real data they differ by about 50–70 days. The chain samples crisis periods more often, because trades end quickly there, so it has many very fast trades: **23% of S&P chain trades end within 30 days, against 2% (training models) and 7–8% (oracle models)**. Overlapping windows weight calm periods by calendar time, which makes trades look long.
- Real durations are also more spread out: standard deviation 145 vs 126 days for Nifty (oracle GBM), and 143 vs 117 for the S&P.
- Checking intraday High/Low instead of only the Close ends trades **9–11 days earlier** on average.

Figures: [plots/q3_durations_nifty.png](plots/q3_durations_nifty.png), [plots/q3_durations_spx.png](plots/q3_durations_spx.png)

## Q4: Which model best predicts the real test period? **Merton, on the one dimension where models differ.**

| | Win rate | Mean duration | Stop-loss fill (close-checked) | KS test on durations |
|---|---|---|---|---|
| Random walk | inside CI | Nifty training and S&P oracle: outside | −10.1%: **outside** | S&P training: rejected |
| GBM | inside CI | Nifty training and S&P oracle: outside | −10.5% to −10.8%: **outside** | S&P training: rejected |
| Merton | inside CI | Nifty training and S&P oracle: outside | −10.7% to −11.3%: **inside** for Nifty (both parameter sets) and S&P oracle | S&P training: rejected |
| **Real** | Nifty 57.5%, S&P 53.6% | 224 / 215 days | **Nifty −12.0% [−12.7, −10.9]; S&P −11.4% [−12.0, −11.0]** | |

- **Win rate can't separate the models.** The real intervals are very wide (Nifty 29%–84%), so every model fits.
- **Durations:** the S&P oracle models' *mean* duration (146–162 days) falls below the overlapping interval (177–257) but close to the chain's 165, which is the clustering effect from Q3. The KS test rejects all three training-fitted S&P models (p ≤ 0.01 at all 60 chain start days). The oracle versions are not rejected (p 0.07–0.11). So the rejection is a parameter problem, not a model-shape problem. With only 12 Nifty trades, nothing can be rejected.
- **Overshoot is where Merton wins.** Merton's worst 1% of stops (−17% to −18%) also matches the real worst Nifty stop (−17.4%), while GBM's worst 1% is −12% to −13%.
- **A real stop-loss order behaves differently.** Checking High/Low and filling at the Open on gap days, the average Nifty stop is −10.3% and the S&P −10.05%. Most of the overshoot comes from waiting for the close; overnight gaps account for the rest. (S&P Opens before 2006 are placeholders, so its intraday figure understates gaps.)

## Q5: What bet size maximizes long-run growth? **Kelly behaves as the theory predicts, once cash is accounted for.**

| | Kelly f* | Matches cash return at | Ratio | Half Kelly keeps |
|---|---|---|---|---|
| Nifty, training (GBM / Merton) | 0.59 / 0.60 | 1.20 / 1.21 | **2.02×** | **75%** of excess growth |
| S&P, training (GBM / Merton) | 2.21 / 2.26 | 4.31 / 4.37 | **1.95× / 1.93×** | **75%** |
| Textbook two-point formula (no r_f, no overshoot) | Nifty 1.86, S&P 3.70 | | | |

- With a positive risk-free rate, the right break-even is "**no better than cash**", not "zero growth". On that basis the "2× Kelly" rule holds within 1–4%.
- **The risk-free rate matters far more than jumps.** At 6% cash, Nifty's Kelly fraction falls from 1.86 (textbook) to 0.59: once cash is accounted for, most of the apparent edge disappears.
- **Hypothesis "jump overshoot lowers f\*": not supported.** Merton's f* is the same or slightly *higher*. Because the models are matched on variance, Merton has calmer diffusion between jumps, which offsets its fatter loss tail at these leverage levels.

Figures: [plots/q5_kelly_nifty.png](plots/q5_kelly_nifty.png), [plots/q5_kelly_spx.png](plots/q5_kelly_spx.png)

## Q6: Which strategy wins the tournament? **It depends on how much edge there is. Buy & Hold won in reality.**

The trader bets the Kelly fraction from the training GBM in every world, with 2,000 careers per world. The career length equals the real chain: 12 rounds for Nifty, 48 for the S&P. Drawdowns are measured daily.

**S&P 500 (f\* = 2.21, so Kelly is leveraged); median results in the GBM world:**

| Strategy | Final wealth | CAGR | Median max drawdown | P(drawdown > 50%) | Real S&P: wealth / max drawdown |
|---|---|---|---|---|---|
| Full Kelly (2.21×) | **2,899** | **7.6%** | 78% | ~100% | 1,217 / **92%** |
| Half Kelly (1.11×) | 1,646 | 6.3% | 46% | 36% | 984 / 64% |
| Buy & Hold | 1,461 | 6.0% | 42% | 22% | **1,403** / 57% |
| Fixed 50% | 758 | 4.5% | 20% | 0% | 508 / 34% |
| Martingale | 908 | 4.8% | 43% | 40% | 269 / 78% |
| Flat ₹50 | 503 | 3.6% | 11% | 0% | 372 / 18% |

- Full Kelly has the highest median wealth in every simulated world, but drawdowns above 50% are almost certain. The real Full Kelly trader fell **92% from peak** (2000–02 and 2008) before recovering. Half Kelly delivers (6.3 − 2.5)/(7.6 − 2.5) = **75% of the excess growth**, with much smaller drawdowns.
- The real S&P trader lands at the 8th–49th percentile of simulated careers: typical, and for martingale and flat betting, unlucky.

**Nifty (f\* = 0.59, so Kelly is conservative):** every strategy earns roughly 7% a year in the simulated worlds, and Half Kelly has the smallest drawdowns (9%). Martingale has the **highest median** wealth (163) but a 7–8% chance of losing half. Over 100 rounds its median drawdown is 84%, with a 95% chance of a drawdown above 50%: steady small gains, then a cluster of losses. The real 2017–2025 trader lands at the **88th–99th percentile** for every strategy. The test period was far better than training data suggested, and **Buy & Hold won (₹285, 15.5% CAGR)**.

- **Hypothesis check:** Full Kelly highest median ✔. Severe drawdowns when Kelly means leverage ✔ (in every world, not just jump worlds). Half Kelly about 75% of the growth with far less risk ✔. Martingale "near-total bankruptcy" ✘: the 3× cap and −10% stops prevent it, though its long-run drawdowns are the worst of the six.

Figures: [plots/q6_tournament_nifty.png](plots/q6_tournament_nifty.png), [plots/q6_tournament_spx.png](plots/q6_tournament_spx.png)

## Q7: What if the trader misjudges the drift? **Overestimating is costly; Half Kelly and the leverage cap protect against it.**

The trader assumes drift = c × true drift and bets the Kelly fraction implied by that belief. 1,000 careers of 100 rounds each.

| | Believed f* at c = 2 (true f*) | Full Kelly growth/yr, P(lose half) | Half Kelly growth/yr, P(lose half) | Cash |
|---|---|---|---|---|
| Nifty, GBM world | 2.29 (0.60) | **−1.5%, 47%** | +6.0%, 4% | +5.9% |
| S&P, GBM world, 3× cap | 5.16 → capped at 3 (2.21) | +6.7%, 4% | +7.2%, 2% | +2.5% |
| S&P, GBM world, no cap | 5.16 | **−3.1%, 60%** | +7.2%, 2% | |
| S&P, Merton world, no cap | 5.16 | **−∞ (one jump can wipe out the trader), 64%** | +7.4%, 1.4% | |

- **Doubling the drift more than doubles the bet.** Kelly depends on drift *minus the risk-free rate*, and 2μ − r_f > 2(μ − r_f). On Nifty, doubling the drift makes the trader bet 3.8× the true optimum, not 2×. So Half Kelly lands *above* the optimum rather than on it. That's still safe (about cash-level growth), but the hypothesis' "lands exactly on f\*" is wrong.
- **The 3× leverage cap acts as a second safety net.** On the S&P it keeps an overconfident Full Kelly trader profitable.
- **Underestimating the drift is cheap:** at c = 0.5–0.75 you give up some growth but never risk ruin.

Figures: [plots/q7_drift_error_nifty.png](plots/q7_drift_error_nifty.png), [plots/q7_drift_error_spx.png](plots/q7_drift_error_spx.png)

---

## Limitations

- **Small Nifty sample.** Only 12 independent trades (11–12 depending on start day; win rate 58%–67%). On Nifty alone, only gross model failures can be detected; the S&P (48 trades) carries the statistical weight.
- **Merton's "jumps" are mostly crisis clusters** (2008, 2020, 1987), not isolated shocks. The model spreads them evenly through time.
- **Cash and costs:** constant risk-free rate, borrowing at the same rate, no transaction costs or taxes. The indices are price indices, so dividends (about 1.3%/yr) are missing from every strategy, including Buy & Hold.
- **Real-world tournament:** one historical path per market. It is reported as "what happened to one trader" and placed in context with percentiles. The trader's Kelly comes from the training GBM.
- **Data:** S&P Open prices are placeholders before about 2006, so S&P gap fills before then default to the barrier. Annualization uses 252 days/year; results in *days* don't depend on this.
- **Martingale rule for timeouts:** the bet fraction is left unchanged after a timeout (the plan didn't specify this case).

## Changes from the original plan

- Corrected the random-walk win-probability formula, and matched the lattice to both mean *and* variance.
- Time cap of 504 days instead of 252. With 252, 21–26% of test-period Nifty trades and about 40% of S&P training forecasts would have timed out.
- The Kelly fraction includes the risk-free rate (cash earns it, borrowing pays it), overshoot and timeouts. It is computed by maximizing expected log growth over the simulated trades.
- Tournament careers match the real chain length (12 / 48 rounds) instead of 25, plus a 100-round long-run view.
- Added the S&P 500 as a validation dataset, the oracle comparison, chain start-day robustness, and Open-price gap fills.

## How to reproduce (about 5 minutes in total)

```
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_models.py           # Step 1-2: data, calibration, closed forms
.venv/Scripts/python run_engine_a.py         # Q1 + model predictions (~15 s)
.venv/Scripts/python run_step3_real.py       # Q2-Q4 real-data scoring (~10 s)
.venv/Scripts/python run_step4_tournament.py # Q5-Q7 (~4 min)
.venv/Scripts/python run_step5_tables.py     # TABLES.md
```
Fixed seeds make every number reproducible. Assumptions (dates, risk-free rates, cap, leverage) live in `barrier/config.py`.
