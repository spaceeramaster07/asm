"""Project-wide constants. Every other module imports from here so the setup lives in one place."""
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


@dataclass(frozen=True)
class Dataset:
    key: str
    ticker: str
    label: str
    download_start: str
    train: tuple[str, str]
    test: tuple[str, str]
    rf: float                          # constant risk-free / borrowing rate (annual)
    download_end: str = "2026-01-01"   # fixed so results are reproducible

    @property
    def raw_csv(self) -> Path:
        return DATA_DIR / f"{self.key}_raw.csv"


DATASETS = {
    # Main study. yfinance history for ^NSEI starts 2007-09-17 -> ~9.3 training years.
    # rf ~ average 91-day T-bill yield over the test period (assumption; edit here).
    "nifty": Dataset("nifty", "^NSEI", "Nifty 50", "2000-01-01",
                     train=("2007-09-17", "2016-12-31"), test=("2017-01-01", "2025-12-31"),
                     rf=0.060),
    # Validation. ^GSPC High/Low are real only from 1962 (before that H=L=C).
    # Open is a placeholder (= previous close) until ~2006; flagged per day by data.clean().
    # rf ~ average 3-month T-bill yield 1994-2025 (assumption; edit here).
    "spx": Dataset("spx", "^GSPC", "S&P 500", "1960-01-01",
                   train=("1962-01-01", "1993-12-31"), test=("1994-01-01", "2025-12-31"),
                   rf=0.025),
}

# Time convention. Markets trade ~245-252 days/yr; we use 252 everywhere for consistency.
# p_win and E[tau] in *days* are invariant to this choice; only displayed annual numbers move.
TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS

S0 = 100.0
UP_RETURN = 0.20     # +20% profit barrier
DOWN_RETURN = -0.10  # -10% ruin barrier
B_LOG = math.log(1 + UP_RETURN)    # b = ln(1.2)  ~ +0.18232
A_LOG = math.log(1 + DOWN_RETURN)  # a = ln(0.9)  ~ -0.10536

# Time cap. 504 trading days (~2 yrs) is used for every model-vs-reality comparison and the
# tournament; a trade still open then is a TIMEOUT (third outcome, closed at market).
# Q1 (simulation vs closed form) runs uncapped; SAFETY_MAX_DAYS only guarantees termination.
CAP_DAYS = 504
CAPS_REPORTED = (252, 504)
SAFETY_MAX_DAYS = 5000

# Broadie-Glasserman-Kou constant: beta = -zeta(1/2)/sqrt(2*pi)
ZETA_HALF = -1.4603545088095868
BETA_BGK = -ZETA_HALF / math.sqrt(2 * math.pi)  # ~ 0.5826

# Merton jump filter: |r - m| > JUMP_K * s, iterated until the jump set is stable.
JUMP_K = 3.0

# Monte Carlo sizes
N_EPISODES = 50_000          # per model per parameter set (spec)
N_CONVERGENCE_POOL = 500_000  # Q1: pool split into disjoint replications at each N
SEED = 20240601

# Step 3: real-data scanner
BOOT_REPS = 2_000
BOOT_BLOCKS = (126, 252, 504)  # block lengths in start-days; 252 is the headline
BOOT_BLOCK_MAIN = 252
CHAIN_OFFSETS = 60             # non-overlapping chain re-run from each of the first 60 days

# Rolling calibration window (5 trading years before each trade entry in test period)
ROLLING_YEARS = 5
ROLLING_DAYS = ROLLING_YEARS * TRADING_DAYS

# Step 4: tournament
W0 = 100.0
N_CAREERS = 2_000
K_LONG = 100                   # long-run career length (K_real comes from the real chain)
LEVERAGE_CAP = 3.0
FLAT_BET = 50.0
FIXED_FRACTION = 0.50
MARTINGALE_START = 0.25
FRACTIONAL_KELLY_MULTS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
Q7_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
Q7_UNCAPPED_MAX = 10.0         # leverage ceiling for the "no cap" Q7 variant
