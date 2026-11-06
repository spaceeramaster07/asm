"""Step 1a: download, clean and split daily OHLC index data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .config import Dataset

OHLC = ["Open", "High", "Low", "Close"]


def download(ds: Dataset, force: bool = False) -> pd.DataFrame:
    """Return raw OHLC data, cached at data/<key>_raw.csv."""
    if ds.raw_csv.exists() and not force:
        return pd.read_csv(ds.raw_csv, index_col=0, parse_dates=True)
    import yfinance as yf

    df = yf.download(ds.ticker, start=ds.download_start, end=ds.download_end,
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds.raw_csv)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop bad rows and flag unreliable intraday fields. Returns (clean_df, quality_report)."""
    df = df[OHLC].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    n0 = len(df)
    df = df.dropna()
    df = df[(df[OHLC] > 0).all(axis=1)]

    # OHLC consistency: High must bound Open/Close from above, Low from below.
    bad = (df.High < df[["Open", "Close"]].max(axis=1) - 1e-6) | \
          (df.Low > df[["Open", "Close"]].min(axis=1) + 1e-6)
    df = df[~bad]

    # Open == previous Close exactly -> placeholder, not a real opening auction price.
    # (Nifty: a few days in 2008-10. S&P: almost every day before ~2006.)
    # Gap-fill logic must fall back to the barrier price on these days.
    df["open_suspect"] = df.Open == df.Close.shift(1)
    # High == Low == Close -> no intraday range recorded (S&P before 1962).
    df["range_suspect"] = (df.High == df.Low) & (df.Low == df.Close)

    gaps = df.index.to_series().diff().dt.days
    years = df.groupby(df.index.year).size()
    report = {
        "rows_raw": int(n0),
        "rows_clean": int(len(df)),
        "rows_dropped": int(n0 - len(df)),
        "ohlc_inconsistent_dropped": int(bad.sum()),
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
        "open_suspect_share": round(float(df.open_suspect.mean()), 4),
        "range_suspect_share": round(float(df.range_suspect.mean()), 4),
        "calendar_gaps_gt_5d": {str(k.date()): int(v) for k, v in gaps[gaps > 5].items()},
        "trading_days_per_year_actual": float(years.iloc[1:-1].mean()),  # full years only
    }
    return df, report


def split(df: pd.DataFrame, ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split. Nothing from TEST may be used to set model parameters."""
    train = df.loc[ds.train[0]:ds.train[1]]
    test = df.loc[ds.test[0]:ds.test[1]]
    assert train.index.max() < test.index.min(), "train/test overlap"
    return train, test


def log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily close-to-close log returns computed *within* one split (no cross-boundary return)."""
    return np.log(df.Close).diff().dropna()


def load(ds: Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    full, report = clean(download(ds).loc[ds.train[0]:ds.test[1]])
    train, test = split(full, ds)
    return full, train, test, report
