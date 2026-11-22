# Implementation plan: rolling-only rebuild

Hypothesis: GBM, a biased random walk and Merton jump-diffusion describe daily log prices well enough to
(a) predict the +20% / −10% barrier game and (b) size bets in it.

| New | Question | Replaces |
|---|---|---|
| Q1 | Does the code match the theory? | Q1 (extended) |
| Q2 | Do the models fit daily returns? | new (absorbs old Q4's tail evidence) |
| Q3 | Do the models predict barrier outcomes? | old Q2 + Q3, rebuilt |
| Q4 | What bet size does each model recommend? | old Q5 |
| Q5 | Which betting strategy wins, and what does mis-estimation cost? | old Q6 + Q7 |

## Ground rules

- **Rolling calibration only.** No static-training or oracle parameter sets anywhere. The old version is
  in git history at `10dba36`.
- **No look-ahead.** Every estimate at time t uses returns up to the close of t only.
- **Monthly calibration grid.** Models are refit on the first trading day of each test month on the
  trailing window; every start day uses its month's fit.
- **Window length**: default 5 years; Q2 scores 3 / 5 / 7 years on the training period and reports which
  fits best (10 years is infeasible for Nifty in 2017). The training period is burn-in and window selection only.
- Old tests are deleted, not kept as baselines. Reusable tools stay (`scanner.scan`, `scanner.chain`,
  `scanner.block_bootstrap`, `theory.*`, `engine.simulate`).

## Defaults

| Setting | Default |
|---|---|
| Rolling window | 5 years (1,260 trading days) |
| Bootstrap world | stationary bootstrap, mean block 63 days, full real history |
| Borrowing rate | r_f + 1%/yr |
| Trading cost | 0.10% of notional at entry and at exit |
| Margin call | position closed at the close when equity < 25% of position value (levered only) |
| Leverage cap | 3× |
| Careers | 2,000 per world, 100 rounds, metrics reported at rounds 20 and 100 |
| Seeds | one helper, `barrier/rng.py` |

## Steps

### Step 0: infrastructure
- `barrier/rng.py`: collision-free `rng_for` (fixes the step-4 seed collisions).
- `barrier/config.py`: new settings; drop Q6/Q7-only constants.
- `barrier/models.py`: `calibration_grid`, `model_for_day`.
- Renamed scripts: `run_calibrate.py`, `run_q1_theory.py`, `run_q2_returns.py`, `run_q3_barriers.py`,
  `run_q4_kelly.py`, `run_q5_tournament.py`, `run_tables.py`.
- `tests/` with pytest.

### Step 1: Q1, theory vs simulation
- `theory.grid_first_passage`: Markov chain on a log-price grid with the model's daily CDF
  (GBM normal, Merton Poisson-normal mixture). Uncapped (linear solve) and capped (time-stepped, gives the
  joint distribution of exit day and exit return, reused by Q3/Q4).
- Rows: RW vs formula; GBM vs continuous and BGK; **GBM vs exact daily grid**; **Merton vs grid**.
  Printed checks (no plot): Merton λ = 0 vs GBM; grid convergence.
- Parameters: rolling fit at the first test date.

### Step 2: Q2, daily-return fit
- `barrier/diagnostics.py`: predictive log-likelihood (GBM, Merton), PIT-normalised residuals
  z = Φ⁻¹(F_model(r)), Jarque–Bera, tail exceedance, Ljung–Box on z², ARCH-LM, Berkowitz test,
  block-bootstrap CIs for paired differences.
- Random walk excluded from likelihood (point masses), stated.
- Window-length comparison on the training period.

### Step 3: Q3, barrier forecasts
- Per monthly fit, exact capped predictions (RW lattice chain, GBM/Merton grid).
- Per start day: 3-class log score and Brier score vs a no-model baseline (trailing outcome frequencies);
  calibration plot; randomised PIT of exit day with block-bootstrap bands; paired model ranking;
  real vs predicted stop-fill distribution (input to Q4/Q5, not scored).

### Step 4: Q4, bet size
- `barrier/kelly.py`: exact g(f) from the grid's joint (exit day, return) distribution, with borrow
  spread and costs; `KellyTable` for fast f* lookups (GBM: μ, σ; Merton: μ, σ_d, λ with jump sizes
  fixed at the training-period fit).
- Rolling f* time series per model with a drift ±1 SE band; growth curve, Half-Kelly share, cash
  break-even; f* with real fills (diagnostic).

### Step 5: Q5, tournament
- A career is one continuous daily path: 5-year visible burn-in, then sequential non-overlapping trades.
  The trader re-estimates from the path's own past before every round.
- Worlds: RW, GBM, Merton (true parameters = full-history fit), **bootstrap (headline)**, real path.
- Strategies: GBM Kelly spectrum 0.25×–2×, Merton Kelly (full/half), empirical Kelly (full/half),
  shrinkage Kelly, vol-targeted Half Kelly, drawdown-controlled Half Kelly, flat 50, fixed 50%,
  Martingale, anti-martingale, buy & hold.
- Costs, borrow spread, margin call, bankruptcy; daily mark-to-market.
- Metrics at rounds 20 and 100; paired growth difference vs GBM Half Kelly with CIs; mis-estimation
  panel (f_used / f_true and growth loss, ruin cliff) in model worlds.

### Step 6: tables, report, cleanup
- New tables, rewritten `REPORT.md`, stale results removed.

## Later

- **GARCH(1,1)-t as a fourth model.** Prototype (scratchpad) showed it beats GBM and Merton on
  out-of-sample daily log-likelihood (Nifty +0.28/day vs GBM, S&P +0.30/day) and removes most volatility
  clustering. Needs a stateful simulator and is the expensive part of Q5 (≈20k warm MLE fits per world;
  measured 0.022 s each) → multiprocessing.
