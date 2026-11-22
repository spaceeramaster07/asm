"""Step 4 of the plan: run the +20% / -10% barrier rule on real price history.

Entry is always at the Close of the start day. Two exit rules:
  close    : barrier checked on daily Closes only, exit at that Close (same as the models).
  intraday : barrier checked on daily High/Low. If the day *opens* beyond a barrier (and the
             Open is a genuine print) the fill is the Open - this is how real stop-loss gaps
             produce losses worse than -10%. Otherwise the fill is the barrier price. If High
             and Low both cross on the same day the order is unknown; we assume the stop hit
             first (conservative) and count these as 'ambiguous'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import config as C
from .engine import RUIN, TIMEOUT, WIN


def _ahead(x: np.ndarray, cap: int) -> np.ndarray:
    """Row i = x[i+1 .. i+cap], NaN past the end of the data."""
    pad = np.concatenate([x.astype(float), np.full(cap, np.nan)])
    return sliding_window_view(pad[1:], cap)[: len(x)]


def scan(df: pd.DataFrame, cap: int = C.CAP_DAYS, mode: str = "close") -> pd.DataFrame:
    """One row per possible start day. `complete` is False when the data ends before the
    trade resolves (neither barrier hit and fewer than `cap` days of data left)."""
    n = len(df)
    logc = np.log(df.Close.to_numpy())
    rows = np.arange(n)
    Xc = _ahead(logc, cap) - logc[:, None]
    a, b = C.A_LOG, C.B_LOG

    if mode == "close":
        up, dn = Xc >= b, Xc <= a
    elif mode == "intraday":
        Xh = _ahead(np.log(df.High.to_numpy()), cap) - logc[:, None]
        Xl = _ahead(np.log(df.Low.to_numpy()), cap) - logc[:, None]
        Xo = _ahead(np.log(df.Open.to_numpy()), cap) - logc[:, None]
        open_ok = _ahead((~df.open_suspect).to_numpy(), cap) == 1.0
        up, dn = Xh >= b, Xl <= a
    else:
        raise ValueError(mode)

    hit = up | dn
    any_hit = hit.any(axis=1)
    j = np.where(any_hit, hit.argmax(axis=1), cap - 1)
    tau = np.where(any_hit, j + 1, cap)
    in_data = rows + cap <= n - 1                      # a full cap window exists
    complete = any_hit | in_data

    ambiguous = np.zeros(n, bool)
    gap = np.zeros(n, bool)
    exit_open_ok = np.ones(n, bool)
    if mode == "close":
        outcome = np.where(any_hit, np.where(up[rows, j], WIN, RUIN), TIMEOUT)
        exit_log = Xc[rows, j]
    else:
        u, d, o, ok = up[rows, j], dn[rows, j], Xo[rows, j], open_ok[rows, j]
        gap_dn, gap_up = ok & (o <= a), ok & (o >= b)
        ambiguous = any_hit & u & d & ~gap_dn & ~gap_up
        is_ruin = gap_dn | (d & ~gap_up)
        outcome = np.where(any_hit, np.where(is_ruin, RUIN, WIN), TIMEOUT)
        fill = np.where(gap_dn | gap_up, o, np.where(is_ruin, a, b))
        exit_log = np.where(any_hit, fill, Xc[rows, j])
        gap = any_hit & (gap_dn | gap_up)
        exit_open_ok = ok

    out = pd.DataFrame({
        "start": df.index, "outcome": outcome.astype(np.int8), "tau": tau.astype(int),
        "exit_log": exit_log, "ret": np.expm1(exit_log), "complete": complete,
        "gap": gap, "ambiguous": ambiguous, "exit_open_ok": exit_open_ok,
    })
    out.loc[~complete, ["outcome", "exit_log", "ret"]] = [TIMEOUT, np.nan, np.nan]
    out["overlap_ok"] = in_data            # start rule for overlapping windows
    return out


def overlapping(sc: pd.DataFrame) -> pd.DataFrame:
    """Starts allowed for overlapping windows: at least `cap` days of data remain."""
    return sc[sc.overlap_ok].reset_index(drop=True)


def chain(sc: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    """Non-overlapping sequential trades: next entry at the Close of the day after the exit.
    The final trade is dropped if the data ends before it resolves (censored)."""
    idx, i, n = [], offset, len(sc)
    while i < n and sc.complete.iat[i]:
        idx.append(i)
        i = i + int(sc.tau.iat[i]) + 1
    out = sc.iloc[idx].copy()
    out["entry_idx"] = idx
    out["exit_idx"] = out.entry_idx + out.tau
    return out.reset_index(drop=True)
