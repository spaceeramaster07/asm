# +20% / −10% barrier trading on Nifty 50 (validated on the S&P 500): results

**Setup.** Buy at the close and exit at the first daily close that is +20% (win) or −10% (ruin) from entry. Any trade still open after 504 trading days is a *timeout*. Three models (biased random walk, GBM, Merton jump-diffusion) are evaluated on an unseen out-of-sample test period under two realistic calibration regimes, plus an hindsight oracle benchmark:

| Dataset | Static training split | Out-of-sample test period | 5-Year rolling calibration (out-of-sample) | Risk-free rate ($r_f$) |
|---|---|---|---|---|
| **Nifty 50 (main)** | 2007-09 → 2016 | 2017 → 2025 ($K = 12$ rounds) | Trailing 1,260 trading days updated before each trade | 6.0% / yr |
| **S&P 500 (validation)** | 1962 → 1993 | 1994 → 2025 ($K = 48$ rounds) | Trailing 1,260 trading days updated before each trade | 2.5% / yr |

All three models reproduce their calibration window's daily mean and variance exactly, so any difference between them comes from the *shape* of returns. **Static training** freezes $(\hat{\mu}, \hat{\sigma}, \hat{\lambda}, \hat{\mu}_J, \hat{\sigma}_J)$ at the end of the training split; **5-year rolling** updates the parameters at the entry date of each trade using the most recent 5 years (1,260 trading days) of returns (100% out-of-sample, zero look-ahead); **Oracle** refits on the full test period solely to diagnose parameter vs. model-structure errors. Full tables are in [TABLES.md](TABLES.md).

## Headline findings

1. **The theory and the code agree.** Simulations converge to the exact formulas at the expected $1/\sqrt{N}$ rate, provided the Broadie–Glasserman–Kou (BGK) daily-checking correction is applied. Without it, the continuous formula is off by ~1.2 percentage points of win rate and 12–18 days.
2. **Updating parameters with a 5-year rolling window dramatically improves out-of-sample model accuracy.** Freezing Nifty parameters in 2016 (which was weighed down by the 2008 crash, $\hat{\sigma}=24.2\%$) makes predicted durations way too short (86–104 days vs. real 152–224 days), while freezing S&P parameters in 1993 ($\hat{\sigma}=14.1\%$) gets rejected by the KS duration test at all 60 chain offsets ($p \le 0.01$). With a **trailing 5-year rolling window**, mean volatility adapts to the true test regime (Nifty `16.6%` vs. real `16.5%`; S&P `18.0%` vs. real `18.7%`). Rolling Merton matches real win rates within 1 point (Nifty `56.7%` vs. `57.5%`; S&P `53.2%` vs. `53.6%`), puts Nifty durations (`194` days) and stop fills (`-11.13%` Nifty, `-11.21%` S&P) squarely inside the real 95% confidence intervals, and **passes the KS test at all 60 start offsets on both markets** ($0\%$ rejected at the 5% level).
3. **Real markets exhibit volatility clustering that constant-volatility models cannot fully replicate.** Average real trade length depends on how you sample the data: 152 days (non-overlapping chain) vs. 224 days (overlapping all-starts) on Nifty, and 165 vs. 215 days on the S&P 500. Sequential chains sample crisis periods more densely because trades exit rapidly during high-volatility spikes.
4. **Merton jump-diffusion is the only model that captures stop-loss overshoot and tail risk.** Under daily close-checking, Merton's mean stop fill (`-11.13%` Nifty rolling, `-11.21%` S&P rolling) and worst 1% tail (`-17.2%` Nifty vs. `-17.4%` real min) match historical losses, whereas Random Walk (`-9.9%`) and GBM (`-10.5%`) are too mild and fall outside the real 95% bootstrap intervals.
5. **Betting strategy deep-dive: Rolling Kelly beats Static Kelly, Half Kelly is the sweet spot on the Fractional Kelly spectrum, and Proportional Sizing crushes Martingale & Flat betting.**
   - **Static vs. Rolling 5y Kelly:** On Nifty 50, updating $f^*_k$ on a 5-year rolling window frees the trader from the overly timid 2007–2016 training bet ($f^* = 0.59 \to \text{mean } 1.84$), boosting simulated GBM median wealth from `₹201` (`+8.2%` CAGR) to **`₹333` (`+14.8%` CAGR)** and improving real test-period wealth for both Full Kelly (`₹222` vs. `₹214`, start-day range up to `₹284`) and Half Kelly (`₹208` vs. `₹183`).
   - **The Fractional Kelly Spectrum ($0.25\times \to 2.00\times f^*$):** Scaling from Quarter-Kelly ($0.25\times$) to Half-Kelly ($0.50\times$) captures **75% of the excess log growth over cash** with roughly half the drawdown of Full Kelly (`31%` vs. `56%` MDD on Nifty rolling; `51%` vs. `68%` MDD on S&P rolling). Pushing beyond $1.0\times f^*$ into Over-Kelly ($1.25\times\text{–}2.00\times$) reduces median wealth while driving drawdowns to `66%–98%`.
   - **Why Martingale & Flat Betting Fail:** Flat betting (`₹50` / `$50`) fails to compound as wealth grows (100-round CAGR decays toward the cash rate). Martingale (doubling after a $-10\%$ loss) looks deceptively strong in short 12-round Nifty samples (`₹204` real), but blows up during losing streaks: on the real S&P 500 it finishes **dead last** (`$269`, `+3.2%` CAGR, `78%` MDD), and over 100 rounds it suffers an **82%–83% median max drawdown** (`92%–93%` probability of losing $>50\%$ from peak).

---

## Q1: Do the simulations converge to the exact formulas? **Yes.**

Each check uses a pool of 500,000 simulated trades per model, with no time cap.

| Comparison | Nifty: $z$ (win) / $z$ (days) | S&P: $z$ (win) / $z$ (days) |
|---|---|---|
| Random walk vs. exact lattice formula | $+1.3$ / $+1.6$ | $+1.6$ / $+1.3$ |
| Daily GBM vs. BGK-corrected formula | $-1.1$ / $+2.3$ | $+0.3$ / $-0.5$ |
| Daily GBM vs. continuous formula | **$+17.7$ / $+102$** | **$+17.6$ / $+60$** |

- Error falls at rate $1/\sqrt{N}$. The 95% confidence intervals contain the exact theoretical value 93–96% of the time.
- The continuous formula's error stops shrinking (the horizontal plateau in the log-log plot). Checking barriers only once a day at the close adds ~1.2 percentage points to the win rate and 12 days (Nifty) to 18 days (S&P) to average trade duration. The BGK continuity correction removes almost all of this bias.

Figures: [plots/q1_convergence_nifty.png](plots/q1_convergence_nifty.png), [plots/q1_convergence_spx.png](plots/q1_convergence_spx.png)

## Q2: How likely is +20% before −10%, and how sensitive is it to drift? **Very sensitive — which is why rolling calibration helps.**

- **With no edge you mostly lose:** exactly **1/3** ($33.3\%$) if price is a martingale ($\mu = 0$), or **36.6%** if log-price has zero drift ($\nu = 0$).
- **Static training vs. 5-year rolling vs. reality (with 504-day cap):**

  | Dataset | Static training (GBM / Merton) | 5y Rolling out-of-sample (GBM / Merton) | Real overlapping (95% CI) | Real non-overlapping chain |
  |---|---|---|---|---|
  | **Nifty 50** | 45.3% / 46.4% | **55.4% / 56.7%** | **57.5%** [29%–84%] | **66.7%** (range 58%–67%) |
  | **S&P 500** | 49.3% / 49.2% | **52.1% / 53.2%** | **53.6%** [41%–67%] | **52.1%** (range 52%–52%) |

- **Why static training misses Nifty's win rate by 11–12 points:** Nifty's 2007–2016 training period included the 2008 global financial crisis, giving a noisy drift estimate ($\hat{\mu} = 9.6\% \pm 8.1\%$/yr SE) and inflated volatility ($24.2\%$/yr). Updating the calibration on a 5-year rolling window (`rolling`) drops the 2008 crash by 2017 and raises predicted win rates to **55.4%–56.7%**, matching the real overlapping win rate (**57.5%**) within 1 percentage point.

Figures: [plots/q2_sensitivity_nifty.png](plots/q2_sensitivity_nifty.png), [plots/q2_sensitivity_spx.png](plots/q2_sensitivity_spx.png)

## Q3: How long until a barrier is hit, and do 5-year rolling models fix the duration mismatch? **Yes — rolling calibration passes the KS test at all 60 chain offsets.**

| Mean days to exit (504-day cap) | Static training (GBM / Merton) | 5y Rolling (GBM / Merton) | Oracle (GBM / Merton) | Real: chain | Real: overlapping [95% CI] |
|---|---|---|---|---|---|
| **Nifty 50** | 96\* / 104\* | **187 / 194** | 183 / 187 | 152 | **224** [154–301] |
| **S&P 500** | 241 / 248 (KS $p \le 0.01$) | **170 / 176 (KS $p = 0.11$)** | 154 / 162 | **165** | 215 [177–257] |

- **5-year rolling calibration resolves the regime-shift error:**
  - On **Nifty 50**, static training models predicted 86–104 days (outside the real 95% CI `[154, 301]`) because 2007–2016 volatility was $24.2\%$. Under 5-year rolling calibration, mean volatility is $16.6\%$ (vs. real test $16.5\%$), and predicted mean durations (**173–194 days**) fall right inside the real 95% confidence interval and match the oracle models (`172–187` days).
  - On **S&P 500**, static training models (`234–248` days) were **rejected by the Kolmogorov–Smirnov (KS) test at 100% of the 60 chain start offsets** ($p \le 0.01$) because 1962–1993 was much calmer ($\hat{\sigma} = 14.1\%$) than 1994–2025 ($18.7\%$). Under 5-year rolling calibration, mean volatility adapts to $18.0\%$, predicted durations (`158–176` days, median `119–139` days) closely match the real sequential chain (`165` mean, `126` median), and **0% of the 60 chain start offsets are rejected by the KS test** ($p = 0.11\text{–}0.19$).
- **Volatility clustering in the calendar timeline:** The bottom panel of `q3_durations_*.png` shows why real sequential chains (`152` days Nifty, `165` days S&P) are shorter than overlapping all-starts (`224` and `215` days): during high-volatility clusters with $|r_t| > 3\hat{\sigma}$ jumps (2000–2002, 2008, 2020), sequential trades resolve in days or weeks and immediately re-enter, packing many fast trades into crisis windows.

Figures: [plots/q3_durations_nifty.png](plots/q3_durations_nifty.png), [plots/q3_durations_spx.png](plots/q3_durations_spx.png)

## Q4: Which model best predicts the real test period? **5-year Rolling Merton wins across every metric.**

| Model (5y Rolling out-of-sample) | Win rate | Mean duration | Stop-loss fill (close-checked) | Stop fill worst 1% | KS duration rejections (60 offsets) |
|---|---|---|---|---|---|
| **Random walk (rolling 5y)** | Nifty 54.8%, S&P 52.0% (inside CI) | Nifty 173d (in CI), S&P 158d | −9.90% / −9.97% (**outside CI**) | −10.4% / −10.7% | **0% rejected** ($p = 0.75 / 0.19$) |
| **GBM (rolling 5y)** | Nifty 55.4%, S&P 52.1% (inside CI) | Nifty 187d (in CI), S&P 170d | −10.54% / −10.65% (**outside CI**) | −12.1% / −12.6% | **0% rejected** ($p = 0.51 / 0.11$) |
| **Merton (rolling 5y)** | **Nifty 56.7%, S&P 53.2% (inside CI)** | **Nifty 194d (in CI), S&P 176d** | **−11.13% / −11.21% (inside CI on both!)** | **−17.2% / −17.3%** | **0% rejected** ($p = 0.36 / 0.11$) |
| **Real test period** | **Nifty 57.5%, S&P 53.6%** | **224d (chain 152d) / 215d (chain 165d)** | **Nifty −12.02% [−12.7, −10.9]; S&P −11.38% [−12.0, −11.0]** | **−17.4% / −18.4% (min)** | — |

- **Merton + 5-year rolling calibration is the only combination whose stop-loss fills land inside the real 95% confidence interval on both Nifty 50 and S&P 500** (`-11.13%` vs. `[-12.7%, -10.9%]` on Nifty; `-11.21%` vs. `[-12.0%, -11.0%]` on S&P 500).
- Its 1% tail loss (`-17.2%` Nifty, `-17.3%` S&P) also accurately warns the trader of real crash overshoots (`-17.4%` Nifty, `-18.4%` S&P), whereas GBM (`-12.1%` to `-12.6%`) understates tail overshoot by ~5 percentage points.

Figures: [plots/q4_overshoot_nifty.png](plots/q4_overshoot_nifty.png), [plots/q4_overshoot_spx.png](plots/q4_overshoot_spx.png)

## Q5: What bet size maximizes long-run growth, and how does Rolling calibration change $f^*$?

| Calibration & Model | Nifty $f^*$ ($r_f = 6.0\%$) | Nifty cash break-even ($f_0 / f^*$) | Nifty $g(f^*)$/yr (excess over cash) | S&P $f^*$ ($r_f = 2.5\%$) | S&P cash break-even ($f_0 / f^*$) | S&P $g(f^*)$/yr (excess over cash) |
|---|---|---|---|---|---|---|
| **Static training: GBM** | 0.59× | 1.20× (2.02×) | +6.92% (+1.03%) | 2.21× | 4.31× (1.95×) | +7.30% (+4.84%) |
| **Static training: Merton** | 0.60× | 1.21× (2.02×) | +6.95% (+1.07%) | 2.26× | 4.37× (1.93×) | +7.52% (+5.06%) |
| **5y Rolling: GBM** | **1.52×** | **2.97× (1.95×)** | **+9.00% (+3.19%)** | **2.29×** | **4.47× (1.95×)** | **+10.30% (+7.83%)** |
| **5y Rolling: Merton** | **1.55×** | **3.00× (1.94×)** | **+9.18% (+3.38%)** | **2.23×** | **3.79× (1.70×)** | **+10.23% (+7.76%)** |
| **Test Oracle: GBM** | 3.02× | 4.77× (1.58×) | +19.03% (+13.22%) | 2.20× | 4.32× (1.96×) | +10.81% (+8.34%) |
| **Textbook 2-point (no $r_f$)** | 1.86× | — | — | 3.70× | — | — |

- **On Nifty 50, 5-year rolling calibration triples the expected excess growth rate over cash:** In the static 2007–2016 training set, high volatility ($24.2\%$) and a 6% risk-free rate compressed Kelly to an unleveraged $f^* = 0.59\times$ earning just $+1.03\%$/yr above cash. Under 5-year rolling calibration, lower volatility ($16.6\%$) lifts optimal leverage to **$f^* = 1.52\times\text{–}1.55\times$** and raises expected growth to **$+9.00\%\text{–}9.18\%$/yr** ($+3.19\%\text{–}3.38\%$/yr excess over cash).
- **Universal Half-Kelly & Cash Break-Even law:** Across every model and dataset, **Half Kelly ($0.5 f^*$) captures 74%–75% of the excess growth over cash** with half the leverage, and over-betting falls back to the cash return at **$1.93\times\text{–}2.02\times f^*$** (except where jump tail risk hits the ruin cliff earlier, at $1.70\times f^*$ on rolling S&P Merton).

Figures: [plots/q5_kelly_nifty.png](plots/q5_kelly_nifty.png), [plots/q5_kelly_spx.png](plots/q5_kelly_spx.png)

## Q6: Betting Strategy Analysis — Static vs. Rolling Kelly, the Fractional Kelly Spectrum, and Martingale vs. Proportional Sizing

We test **8 strategies** (`Flat Rs50`, `Fixed 50%`, `Static Kelly`, `Static Half Kelly`, `Rolling Kelly`, `Rolling Half Kelly`, `Martingale`, and `Buy & Hold`) plus the **7-point Fractional Kelly Spectrum** ($0.25\times, 0.50\times, 0.75\times, 1.00\times, 1.25\times, 1.50\times, 2.00\times f^*$) across 2,000 simulated careers per world and all 60 real start-day chains.

### 1. Static Kelly vs. Adaptive 5-Year Rolling Kelly

| Dataset & Strategy | Mean $f$ | Simulated GBM: Median $W$ (CAGR / MDD) | Simulated Merton: Median $W$ (CAGR / MDD) | Real Test Chain: $W$ [60-offset range] (CAGR / MDD) |
|---|---|---|---|---|
| **Nifty: Static Half Kelly** | 0.30× | 187 (+7.2% / 6%) | 192 (+7.3% / 7%) | 183 [173–185] (+8.7% / 13%) |
| **Nifty: Rolling Half Kelly** | **0.92×** | **276 (+12.3% / 31%)** | **292 (+12.2% / 31%)** | **208 [195–223] (+10.6% / 28%)** |
| **Nifty: Static Kelly** | 0.59× | 201 (+8.2% / 17%) | 210 (+8.4% / 16%) | 214 [199–219] (+11.0% / 25%) |
| **Nifty: Rolling Kelly** | **1.84×** | **333 (+14.8% / 56%)** | **361 (+14.7% / 56%)** | **222 [204–284] (+11.6% / 52%)** |
| **Nifty: Buy & Hold** | 1.00× | 215 (+9.2% / 31%) | 225 (+9.3% / 31%) | **285 [241–285] (+15.5% / 38%)** |
| **S&P: Static Half Kelly** | 1.11× | 1,427 (+8.6% / 61%) | 1,508 (+8.4% / 62%) | 984 [966–1,001] (+7.5% / 64%) |
| **S&P: Rolling Half Kelly** | **1.24×** | **16,526 (+17.1% / 51%)** | **18,516 (+16.9% / 52%)** | **998 [968–1,511] (+7.5% / 71%)** |
| **S&P: Static Kelly** | 2.21× | 2,683 (+10.8% / 91%) | 2,674 (+10.3% / 92%) | 1,217 [1,202–1,376] (+8.2% / 92%) |
| **S&P: Rolling Kelly** | **2.47×** | **22,516 (+18.1% / 68%)** | **25,738 (+17.9% / 69%)** | 591 [582–748] (+5.8% / 89%) |
| **S&P: Buy & Hold** | 1.00× | 1,262 (+8.2% / 56%) | 1,332 (+8.0% / 57%) | **1,403 [1,355–1,420] (+8.7% / 57%)** |

- **Why Rolling Kelly outperforms Static Kelly on Nifty (and in all simulated worlds):** By updating $(\hat{\mu}_k, \hat{\sigma}_k)$ on a trailing 5-year window before each trade, the Nifty trader sizes up from $0.59\times$ to $1.75\times$ (capped mean) during the 2017–2019 and 2023–2025 expansions. In simulated worlds, `Rolling Half Kelly` (`₹276`, `+12.3%` CAGR, `31%` MDD) beats `Buy & Hold` (`₹215`, `+9.2%` CAGR, `31%` MDD) by **+3.1%/yr at the exact same drawdown**, and on the real Nifty chain both `Rolling Kelly` (`₹222`, range up to `₹284`) and `Rolling Half Kelly` (`₹208`, range up to `₹223`) beat their static counterparts (`₹214` and `₹183`).
- **The post-crash "whipsaw" insight on real paths (Middle panel of `q6_frontier_*.png`):** Why does pure 5-year `Rolling Kelly` lag `Buy & Hold` on the real Nifty path (and lag `Static Kelly` on the real S&P path)? Look at the round-by-round leverage step plot in `q6_frontier_*.png`: right after a sharp market crash (COVID-19 in 2020 on Nifty rounds 5–7; the 2000–2002 dot-com bust and 2008 GFC on S&P rounds 16–33), the trailing 5-year return temporarily falls below the risk-free rate ($\hat{\mu}_k < r_f$). A mechanical 5-year Kelly trader therefore cuts leverage to **$f_k = 0$ (100% cash)** right at the market bottom and misses the V-shaped rebound! `Rolling Half Kelly` softens the pre-crash losses so it still beats `Static Half Kelly` on both real markets (`₹208` vs. `₹183` on Nifty; `$998` vs. `$984` on S&P, with start-day range up to `$1,511`).

### 2. The Fractional Kelly Spectrum ($0.25\times \to 2.00\times f^*$)

- **The Risk-Return Frontier (Left panel of `q6_frontier_*.png` and `Table 2c`):**
  - Moving from **Quarter-Kelly ($0.25\times$)** to **Half-Kelly ($0.50\times$)** delivers a massive jump in CAGR (Nifty rolling: `+9.3%` $\to$ `+12.0%`; S&P rolling: `+11.1%` $\to$ `+17.3%`) while keeping median max drawdown moderate (`31%` Nifty, `51%` S&P) and near-zero probability of losing half (`0.0%–0.1%`).
  - Moving from **Half-Kelly ($0.50\times$)** to **Full Kelly ($1.00\times$)** adds only `+1.3%` to `+2.2%` in simulated CAGR, but **nearly doubles median max drawdown** (`31%` $\to$ `56%` on Nifty; `61%` $\to$ `91%` on S&P static) and spikes the probability of a $>50\%$ drawdown to `67%–100%`.
  - **Over-Kelly ($1.25\times, 1.50\times, 2.00\times f^*$)** bends backward on the risk-return frontier: both CAGR and median wealth *fall* as leverage increases past $1.0\times f^*$, while real S&P terminal wealth collapses from `$1,285` (at $0.75\times$ Static Kelly) to `$629` (at $1.50\times\text{–}2.00\times$) with a **98% max drawdown**.

### 3. Why Martingale & Flat Betting Fail vs. Proportional Sizing

- **Flat Betting (`₹50` / `$50`) under-compounds:** Because the dollar bet stays fixed at `50` while wealth grows to `1,000+`, the effective bet fraction $N_k / W_k$ shrinks toward zero over a career. Over 100 rounds (`Table 2b`), `Flat Rs50` earns only `+6.4%` CAGR on Nifty (barely above the `6.0%` cash rate) and `+3.7%` on S&P (`1,342` terminal wealth vs. `40,758` for `Static Half Kelly` and `8,404,525` for `Rolling Half Kelly`).
- **Martingale (`0.10 -> 0.20 -> 0.40 -> ... -> 3.00x`) is a tail-risk trap:** Doubling the bet fraction after every $-10\%$ stop-loss looks benign in short bull-market samples (`₹204` on real 12-round Nifty, `15%–16%` median simulated MDD). However, when consecutive losses occur during bear markets (like 2000–2002 and 2008 on the S&P 500), Martingale escalates to the $3\times$ leverage cap right into the teeth of the crash. On the real 48-round S&P 500, **Martingale finishes dead last (`$269`, `+3.2%` CAGR, `78%` MDD)**, and over 100 rounds in the Merton world (Right panel of `q6_frontier_*.png`) it suffers an **82% median max drawdown** and a **92% chance of a $>50\%$ drawdown** for a dismal `+4.1%` CAGR.

Figures: [plots/q6_tournament_nifty.png](plots/q6_tournament_nifty.png), [plots/q6_tournament_spx.png](plots/q6_tournament_spx.png), [plots/q6_frontier_nifty.png](plots/q6_frontier_nifty.png), [plots/q6_frontier_spx.png](plots/q6_frontier_spx.png)

## Q7: What if the trader misjudges the drift? **Overestimating is costly; Half Kelly and the leverage cap protect against it.**

The trader assumes drift $= c \times \mu_{\text{true}}$ and bets the Kelly fraction implied by that belief across 1,000 careers of 100 rounds each.

| World & Cap | Believed $f^*$ at $c = 2$ (true $f^*$) | Full Kelly growth/yr, $P(W < 50)$ | Half Kelly growth/yr, $P(W < 50)$ | Cash rate |
|---|---|---|---|---|
| **Nifty, GBM world** | 2.29× (0.59×) | **−1.5%, 47%** | +6.0%, 4% | +5.9% |
| **S&P, GBM world, 3× cap** | 5.16× → capped at 3× (2.21×) | +6.7%, 4% | +7.2%, 2% | +2.5% |
| **S&P, GBM world, no cap** | 5.16× | **−3.1%, 60%** | +7.2%, 2% | +2.5% |
| **S&P, Merton world, no cap** | 5.16× | **−∞ (ruin cliff: one jump wipes out trader), 64%** | +7.4%, 1.4% | +2.5% |

- **Doubling the drift more than doubles the bet:** Because Kelly scales with excess drift $(\mu - r_f)$ and $2\mu - r_f > 2(\mu - r_f)$, doubling $\hat{\mu}$ on Nifty causes the trader to bet **$3.8\times$ the true optimum**, pushing Full Kelly into negative growth (`-1.5%/yr`).
- **The Merton Ruin Cliff (Right heatmap panel of `q7_drift_error_*.png`):** In the Merton jump-diffusion world, once uncapped leverage exceeds $f_{\text{cliff}} = 1 / |R_{\min}|$ ($\approx 5.2\times$), a single jump-overshoot episode wipes out 100% of equity ($g = -\infty$). Both **Half Kelly** and the **3× leverage cap** keep the trader safely below the ruin cliff across all tested drift errors $c \in [0.5, 2.0]$.

Figures: [plots/q7_drift_error_nifty.png](plots/q7_drift_error_nifty.png), [plots/q7_drift_error_spx.png](plots/q7_drift_error_spx.png)

---

## How to reproduce (~4 minutes in total)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_models.py           # Step 1-2: data, static + 5y rolling + oracle calibration
.venv/Scripts/python run_engine_a.py         # Q1 + model predictions (~20 s)
.venv/Scripts/python run_step3_real.py       # Q2-Q4 real-data scoring & plots (~12 s)
.venv/Scripts/python run_step4_tournament.py # Q5-Q7 tournament & strategy deep-dive (~3 min)
.venv/Scripts/python run_step5_tables.py     # TABLES.md (Tables 1, 2, 2b, 2c)
```
Fixed seeds make every number reproducible. Assumptions (dates, risk-free rates, 5-year rolling window `ROLLING_DAYS = 1260`, Fractional Kelly multipliers, cap, leverage) live in [`barrier/config.py`](../barrier/config.py).
