"""Project-wide constants. Every other module imports from here so the setup lives in one place."""
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CACHE_DIR = RESULTS_DIR / "cache"     # large regenerable arrays (gitignored)


@dataclass(frozen=True)
class Dataset:
    key: str
    ticker: str
    label: str
    download_start: str
    train: tuple[str, str]
    test: tuple[str, str]
    rf: float                          # constant risk-free rate (annual)
    download_end: str = "2026-01-01"   # fixed so results are reproducible

    @property
    def raw_csv(self) -> Path:
        return DATA_DIR / f"{self.key}_raw.csv"


DATASETS = {
    # Main study. yfinance history for ^NSEI starts 2007-09-17. The training years are only used as
    # burn-in for the rolling window and to compare window lengths; no parameter is frozen from them.
    # rf ~ average 91-day T-bill yield over the test period (assumption; edit here).
    "nifty": Dataset("nifty", "^NSEI", "Nifty 50", "2000-01-01",
                     train=("2007-09-17", "2016-12-31"), test=("2017-01-01", "2025-12-31"),
                     rf=0.060),
    # Validation. ^GSPC High/Low are real only from 1962. rf ~ average 3-month T-bill 1994-2025.
    "spx": Dataset("spx", "^GSPC", "S&P 500", "1960-01-01",
                   train=("1962-01-01", "1993-12-31"), test=("1994-01-01", "2025-12-31"),
                   rf=0.025),
}

# Time convention: 252 trading days per year everywhere.
TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS

S0 = 100.0
UP_RETURN = 0.20     # +20% profit barrier
DOWN_RETURN = -0.10  # -10% ruin barrier
B_LOG = math.log(1 + UP_RETURN)    # b = ln(1.2)  ~ +0.18232
A_LOG = math.log(1 + DOWN_RETURN)  # a = ln(0.9)  ~ -0.10536

# Time cap: a trade still open after 504 trading days (~2 yrs) is a TIMEOUT closed at market.
# Q1 runs uncapped; SAFETY_MAX_DAYS only guarantees termination.
CAP_DAYS = 504
CAPS_REPORTED = (252, 504)
SAFETY_MAX_DAYS = 5000

# Broadie-Glasserman-Kou constant: beta = -zeta(1/2)/sqrt(2*pi)
ZETA_HALF = -1.4603545088095868
BETA_BGK = -ZETA_HALF / math.sqrt(2 * math.pi)  # ~ 0.5826

# Merton jump filter: |r - m| > JUMP_K * s, iterated until the jump set is stable.
JUMP_K = 3.0

# Rolling calibration: trailing window, refit on the first trading day of each month.
ROLLING_YEARS = 5
ROLLING_DAYS = ROLLING_YEARS * TRADING_DAYS
WINDOW_YEARS_SENSITIVITY = (3, 5, 7)

# Monte Carlo / numerics
N_CONVERGENCE_POOL = 500_000   # Q1: pool split into disjoint replications at each N
GRID_N = 1_200                 # cells on [a, b] for the Markov-chain solver
SEED = 20240601

# Bootstrap confidence intervals over (overlapping) start days / daily scores
BOOT_REPS = 2_000
BOOT_BLOCK_DAYS = 63           # daily-score blocks (Q2)
BOOT_BLOCK_STARTS = 252        # start-day blocks (Q3)
CHAIN_OFFSETS = 60             # real chains re-run from each of the first 60 days

# Betting game (Q4, Q5)
W0 = 100.0
LEVERAGE_CAP = 3.0
BORROW_SPREAD = 0.01           # borrowing costs r_f + 1%/yr
TRADE_COST = 0.001             # 0.10% of notional at entry and at exit
MARGIN_MAINT = 0.25            # levered position closed when equity < 25% of position value
N_CAREERS = 2_000
K_ROUNDS = 100
K_REPORT = (20, 100)           # metrics are reported at these rounds
BOOT_WORLD_MEAN_BLOCK = 63     # stationary bootstrap mean block length (trading days)
FLAT_BET = 50.0
FIXED_FRACTION = 0.50
MARTINGALE_START = 0.25
KELLY_SPECTRUM = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
VOL_TARGET_DAYS = 60           # short volatility window for vol-targeted Half Kelly
DD_TRIGGER = 0.20              # drawdown control: halve the bet below a 20% drawdown
F_GRID_MAX = 6.0               # Kelly search range (growth curves are evaluated on [0, F_GRID_MAX])
