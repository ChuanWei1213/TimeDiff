"""Training-only empirical sampling of paired history and future conditions.

The sampling unit is a whole (past 63 days, next 21 days' R and sigma) pair.
The full pool includes the tail. Pool membership is not an outcome label.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_provider.dow10_cond import (
    COLUMNS, PRED_LEN, SEQ_LEN, window_conditions,
)


@dataclass
class JointPlan:
    population: pd.DataFrame
    draws: pd.DataFrame
    metadata: dict


def training_population(features, train_start, train_end, tail_quantile=0.30):
    """Include only complete target windows; past-only context may predate start."""
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    if start > end or not 0 < tail_quantile < 1:
        raise ValueError("Need train_start <= train_end and 0 < tail_quantile < 1")
    dates = pd.DatetimeIndex(features.index)
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Feature dates must be unique and strictly increasing")
    # Slice first: neither quantiles nor conditions inspect validation/test rows.
    eligible = features.loc[:end, COLUMNS]
    raw = eligible.to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Training features must be finite")
    dates = pd.DatetimeIndex(eligible.index)
    starts = np.array([
        s for s in range(SEQ_LEN, len(dates) - PRED_LEN + 1)
        if dates[s] >= start
    ], dtype=np.int64)
    if not len(starts):
        raise ValueError("No complete 63-day history / 21-day target windows")
    conditions = window_conditions(raw, starts)
    cutoff = float(np.quantile(conditions[:, 0], tail_quantile, method="linear"))
    population = pd.DataFrame({
        "window_id": starts,
        "history_start": dates[starts - SEQ_LEN].strftime("%Y-%m-%d"),
        "anchor_date": dates[starts - 1].strftime("%Y-%m-%d"),
        "target_start": dates[starts].strftime("%Y-%m-%d"),
        "target_end": dates[starts + PRED_LEN - 1].strftime("%Y-%m-%d"),
        "target_return": conditions[:, 0],
        "target_volatility": conditions[:, 1],
        "is_tail": conditions[:, 0] <= cutoff,
    })
    return population, cutoff


def make_joint_plan(features, *, train_start="2010-01-01", train_end="2017-12-31",
                    n_draws=500, full_weight=0.70, tail_quantile=0.30, seed=20261017):
    """Stratify pool counts, sample within each pool with replacement, then shuffle.

    Exact 70/30 allocation requires n_draws divisible by 10. Otherwise the full
    count is rounded half-up and the realized allocation is recorded explicitly.
    """
    if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    if not np.isfinite(full_weight) or not 0 <= full_weight <= 1:
        raise ValueError("full_weight must be finite and in [0, 1]")
    population, cutoff = training_population(features, train_start, train_end, tail_quantile)
    rng = np.random.default_rng(seed)
    n_full = int(np.floor(n_draws * full_weight + 0.5))
    n_tail = n_draws - n_full
    tail = np.flatnonzero(population.is_tail.to_numpy())
    picks = np.r_[rng.choice(len(population), n_full, replace=True),
                  rng.choice(tail, n_tail, replace=True)]
    pools = np.array(["full"] * n_full + ["tail"] * n_tail)
    order = rng.permutation(n_draws)
    draws = population.iloc[picks[order]].reset_index(drop=True).copy()
    draws.insert(0, "draw_id", np.arange(n_draws))
    draws.insert(1, "source_pool", pools[order])
    actual_full = n_full / n_draws
    p = actual_full / len(population) + (1 - actual_full) * population.is_tail / len(tail)
    population["mixture_probability"] = p
    tail_share = len(tail) / len(population)
    metadata = {
        "sampler": "paired_empirical_full_plus_lower_tail",
        "train_start": str(pd.Timestamp(train_start).date()),
        "train_end": str(pd.Timestamp(train_end).date()),
        "history_days": SEQ_LEN, "target_days": PRED_LEN,
        "n_population": len(population), "n_tail_population": len(tail),
        "tail_quantile": tail_quantile, "tail_return_cutoff": cutoff,
        "tail_rule": "target_return <= training quantile (linear interpolation)",
        "requested_full_weight": full_weight, "actual_full_weight": actual_full,
        "n_draws": n_draws, "n_full_draws": n_full, "n_tail_draws": n_tail,
        "plan_seed": int(seed), "population_tail_share": tail_share,
        "expected_sampled_tail_share": actual_full * tail_share + 1 - actual_full,
        "sampled_tail_share": float(draws.is_tail.mean()),
        "n_unique_windows": int(draws.window_id.nunique()),
        "sampling_unit": "paired original history and its own following return/volatility",
        "replacement": True,
        "caveats": [
            "Overlapping windows are dependent; unique windows are not independent events.",
            "The full pool also contains tail windows; source_pool is not is_tail.",
            "Past-only context may predate train_start; every target day is inside training.",
            "Future conditions and source labels must never enter the PPO observation.",
        ],
    }
    return JointPlan(population, draws, metadata)


def paired_inputs(features, draws):
    """Return original paired H/Y/C arrays; Y is only for diagnostics, not inference."""
    raw = features[COLUMNS].to_numpy(dtype=np.float64)
    starts = draws.window_id.to_numpy(dtype=np.int64)
    histories = np.stack([raw[s - SEQ_LEN:s] for s in starts])
    reference_futures = np.stack([raw[s:s + PRED_LEN] for s in starts])
    targets = draws[["target_return", "target_volatility"]].to_numpy(dtype=np.float64)
    # Do not accept a plan whose source history or target was silently changed.
    if not np.allclose(window_conditions(raw, starts), targets, rtol=1e-12, atol=1e-12):
        raise ValueError("The plan's conditions no longer match its original future windows")
    return histories, reference_futures, targets
