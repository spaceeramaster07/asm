"""Step 3 (Engine A): single-episode Monte Carlo first-passage simulator.

Every path starts at X=0 (price 100), adds one daily log-return increment per trading day,
and stops on the first daily close with X >= b (win) or X <= a (ruin). Paths are run
*uncapped* (only SAFETY_MAX_DAYS for termination); while running we snapshot each path's
log price at every day in CAPS_REPORTED, so any time cap can be applied afterwards without
re-simulating (see apply_cap).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

WIN, RUIN, TIMEOUT = 1, 0, -1


@dataclass
class Episodes:
    outcome: np.ndarray        # WIN / RUIN (uncapped; TIMEOUT only if SAFETY_MAX_DAYS hit)
    tau: np.ndarray            # exit day (1-based); SAFETY_MAX_DAYS if never exited
    exit_log: np.ndarray       # X at exit day
    snap: dict                 # cap day -> X at that day (NaN if exited on/before it)

    def __len__(self) -> int:
        return len(self.tau)

    def subset(self, idx) -> "Episodes":
        return Episodes(self.outcome[idx], self.tau[idx], self.exit_log[idx],
                        {k: v[idx] for k, v in self.snap.items()})


def sim_barriers(model) -> tuple[float, float]:
    """Barriers in log space as the simulator applies them.

    Random walk: lattice barriers -A*dx and +B*dx, placed half a step inside so float
    rounding in the cumulative sum can never decide a crossing. Other models: true a, b."""
    if model.name == "RandomWalk":
        return -(model.A - 0.5) * model.dx, (model.B - 0.5) * model.dx
    return C.A_LOG, C.B_LOG


def simulate(model, n: int, rng: np.random.Generator, max_days: int = C.SAFETY_MAX_DAYS,
             caps=C.CAPS_REPORTED, block: int = 128, chunk: int = 50_000) -> Episodes:
    a, b = sim_barriers(model)
    outcome = np.full(n, TIMEOUT, dtype=np.int8)
    tau = np.full(n, max_days, dtype=np.int32)
    exit_log = np.full(n, np.nan)
    snap = {c: np.full(n, np.nan) for c in caps}

    for start in range(0, n, chunk):
        ids = np.arange(start, min(start + chunk, n))
        x = np.zeros(len(ids))
        alive = np.arange(len(ids))            # positions within this chunk
        day = 0
        while alive.size and day < max_days:
            L = min(block, max_days - day)
            path = x[alive, None] + np.cumsum(model.increments(rng, (alive.size, L)), axis=1)
            up, dn = path >= b, path <= a
            hit = up | dn
            any_hit = hit.any(axis=1)
            first = np.where(any_hit, hit.argmax(axis=1), L)   # L = not hit in this block

            for c in caps:                      # snapshot X at day c for paths still open
                j = c - day - 1
                if 0 <= j < L:
                    still_open = first > j
                    snap[c][ids[alive[still_open]]] = path[still_open, j]

            rows = np.nonzero(any_hit)[0]
            f = first[rows]
            g = ids[alive[rows]]
            tau[g] = day + f + 1
            exit_log[g] = path[rows, f]
            outcome[g] = np.where(up[rows, f], WIN, RUIN)

            keep = ~any_hit
            x[alive[keep]] = path[keep, -1]
            alive = alive[keep]
            day += L
        if alive.size:                          # survived SAFETY_MAX_DAYS: close at market
            exit_log[ids[alive]] = x[alive]
    return Episodes(outcome, tau, exit_log, snap)


def load(path) -> Episodes:
    """Load episodes saved by run_engine_a.py (results/sims/*.npz)."""
    z = np.load(path)
    snap = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("snap_")}
    return Episodes(z["outcome"], z["tau"], z["exit_log"], snap)


def apply_cap(ep: Episodes, cap: int | None) -> Episodes:
    """View of the same paths under a time cap: anything still open at `cap` is a TIMEOUT
    closed at that day's price."""
    if cap is None:
        return ep
    late = ep.tau > cap
    outcome = np.where(late, TIMEOUT, ep.outcome).astype(np.int8)
    tau = np.where(late, cap, ep.tau)
    exit_log = np.where(late, ep.snap[cap], ep.exit_log)
    return Episodes(outcome, tau, exit_log, {})


def summarize(ep: Episodes) -> dict:
    """Everything Table 1 needs from one set of episodes."""
    n = len(ep)
    win, ruin, to = ep.outcome == WIN, ep.outcome == RUIN, ep.outcome == TIMEOUT
    ret = np.expm1(ep.exit_log)
    t = ep.tau.astype(float)
    p = win.mean()

    def mean_or_nan(x):
        return float(x.mean()) if x.size else float("nan")

    return {
        "n": n,
        "p_win": float(p),
        "p_ruin": float(ruin.mean()),
        "p_timeout": float(to.mean()),
        "p_win_se": float(np.sqrt(p * (1 - p) / n)),
        # win share among resolved trades (comparable to the uncapped formulas)
        "p_win_resolved": float(win.sum() / max(win.sum() + ruin.sum(), 1)),
        "tau_mean": float(t.mean()),
        "tau_se": float(t.std(ddof=1) / np.sqrt(n)),
        "tau_median": float(np.median(t)),
        "tau_sd": float(t.std(ddof=1)),
        "tau_p05": float(np.percentile(t, 5)),
        "tau_p95": float(np.percentile(t, 95)),
        "tau_mean_win": mean_or_nan(t[win]),
        "tau_mean_ruin": mean_or_nan(t[ruin]),
        "ret_mean_win": mean_or_nan(ret[win]),
        "ret_mean_ruin": mean_or_nan(ret[ruin]),          # stop-loss overshoot
        "ret_p01_ruin": float(np.percentile(ret[ruin], 1)) if ruin.any() else float("nan"),
        "ret_mean_timeout": mean_or_nan(ret[to]),
        "ret_mean_all": float(ret.mean()),
    }
