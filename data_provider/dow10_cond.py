"""Dow 10 close/high/low features + equal-weight portfolio conditions (63 -> 21).

Feature layout (30 channels, ticker-major, same as the original Dow10 pipeline):
    AAPL_rclose, AAPL_rhigh, AAPL_rlow, AMGN_rclose, ..., WMT_rlow
Condition (2 numbers) = statistics of the daily-rebalanced equal-weight portfolio:
    r_p,d = log( mean_i exp(r_i,d) )
    R     = exp(sum_d r_p,d) - 1                     (21-day cumulative return)
    sigma = std(r_p,d; ddof=1) * sqrt(252)           (annualized volatility)
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN, PRED_LEN = 63, 21
TICKERS = ["AAPL", "AMGN", "CRM", "CSCO", "IBM", "INTC", "MSFT", "NKE", "VZ", "WMT"]
FEATS = ["rclose", "rhigh", "rlow"]
COLUMNS = [f"{t}_{f}" for t in TICKERS for f in FEATS]
CLOSE_INDICES = [i * len(FEATS) for i in range(len(TICKERS))]       # 0, 3, ..., 27
N_VARS = len(COLUMNS)                                                # 30


# ───────────────────────────── statistics ─────────────────────────────
def portfolio_log_returns(close_log_returns):
    """[..., T, n_assets] daily log returns -> [..., T] equal-weight portfolio log returns."""
    r = np.asarray(close_log_returns, dtype=np.float64)
    m = r.max(axis=-1, keepdims=True)                               # stable log-mean-exp
    return (m + np.log(np.exp(r - m).mean(axis=-1, keepdims=True)))[..., 0]


def portfolio_statistics(close_log_returns):
    """[..., T, n_assets] -> [..., 2] = (21-day cumulative return, annualized vol)."""
    rp = portfolio_log_returns(close_log_returns)
    if rp.shape[-1] < 2 or not np.isfinite(rp).all():
        raise ValueError("Need >= 2 finite daily returns")
    return np.stack([np.expm1(rp.sum(axis=-1)), rp.std(axis=-1, ddof=1) * np.sqrt(252)], axis=-1)


# ───────────────────────────── prices ─────────────────────────────
def download_dow10(path, start, end):
    """Split/dividend-adjusted OHLC (auto_adjust=True, as the original Dow10 pipeline).

    Unadjusted prices would turn AAPL's 2014 7:1 / 2020 4:1 splits into fake crashes.
    Output is long format date,tic,close,high,low -- the exact file FinRL also reads.
    """
    import yfinance as yf
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; use a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(path.parent / "yfinance_cache"))
    raw = yf.download(TICKERS, start=start, end=end, interval="1d", auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo returned no data")
    frames = []
    for t in TICKERS:
        f = pd.DataFrame({"close": raw["Close"][t], "high": raw["High"][t], "low": raw["Low"][t]})
        f.index = pd.to_datetime(f.index).tz_localize(None)
        f["tic"] = t
        frames.append(f.rename_axis("date").reset_index())
    prices = pd.concat(frames, ignore_index=True)[["date", "tic", "close", "high", "low"]]
    prices = prices.sort_values(["date", "tic"]).reset_index(drop=True)
    validate_prices(prices)
    prices.to_csv(path, index=False, date_format="%Y-%m-%d")
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "tickers": TICKERS, "start_inclusive": start, "end_exclusive": end,
        "auto_adjust": True, "yfinance_version": yf.__version__,
        "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": len(prices), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return prices


def validate_prices(prices):
    v = prices[["close", "high", "low"]].to_numpy(float)
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Prices must be positive and finite (no forward-fill)")
    if (prices.high < prices.close * (1 - 1e-9)).any() or (prices.low > prices.close * (1 + 1e-9)).any():
        raise ValueError("Need low <= close <= high")
    counts = prices.groupby("date").tic.nunique()
    if (counts != len(TICKERS)).any():
        raise ValueError("Every date must have all 10 tickers")
    if set(prices.tic) != set(TICKERS):
        raise ValueError(f"Tickers must be {TICKERS}")


def load_prices(path):
    prices = pd.read_csv(path, parse_dates=["date"])
    validate_prices(prices)
    return prices


def wide_prices(prices):
    """long -> dict(close/high/low) of [dates x TICKERS] frames."""
    return {c: prices.pivot(index="date", columns="tic", values=c)[TICKERS] for c in ("close", "high", "low")}


def load_features(path):
    prices = load_prices(path)
    w = wide_prices(prices)
    rclose = np.log(w["close"] / w["close"].shift(1))
    rhigh = np.log(w["high"] / w["close"])
    rlow = np.log(w["low"] / w["close"])
    parts = {"rclose": rclose, "rhigh": rhigh, "rlow": rlow}
    features = pd.DataFrame({f"{t}_{f}": parts[f][t] for t in TICKERS for f in FEATS},
                            index=w["close"].index).iloc[1:]
    features = features[COLUMNS]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Non-finite features")
    return prices, features


# ───────────────────────────── scaling / windows ─────────────────────────────
class Scaler:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float64)
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8))

    def transform(self, values):
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values):
        return np.asarray(values) * self.scale + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def window_conditions(raw, starts):
    """raw: [T, 30] raw features; starts: target start rows -> [n, 2] raw conditions."""
    return np.array([portfolio_statistics(raw[s:s + PRED_LEN][:, CLOSE_INDICES]) for s in starts])


class Dow10Windows(Dataset):
    def __init__(self, features, starts, feature_scaler, condition_scaler):
        self.raw = features[COLUMNS].to_numpy(dtype=np.float64)
        self.values = feature_scaler.transform(self.raw)
        self.starts = list(starts)
        self.conditions = condition_scaler.transform(window_conditions(self.raw, self.starts))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return (torch.from_numpy(self.values[s - SEQ_LEN:s].copy()),
                torch.from_numpy(self.values[s:s + PRED_LEN].copy()),
                torch.from_numpy(self.conditions[i].copy()))


def split_windows(features, target_start, train_end, val_end):
    """Target windows by date. Train targets lie fully in [target_start, train_end];
    validation targets fully in (train_end, val_end]. Straddling windows are dropped.
    History (63 days before a target) may precede target_start; it is input only."""
    target_start, train_end, val_end = map(pd.Timestamp, (target_start, train_end, val_end))
    if not target_start < train_end < val_end:
        raise ValueError("Need target_start < train_end < val_end")
    dates = features.index
    groups = {"train": [], "validation": []}
    for s in range(SEQ_LEN, len(features) - PRED_LEN + 1):
        first, last = dates[s], dates[s + PRED_LEN - 1]
        if first >= target_start and last <= train_end:
            groups["train"].append(s)
        elif first > train_end and last <= val_end:
            groups["validation"].append(s)
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Not enough train/validation windows")
    return groups
