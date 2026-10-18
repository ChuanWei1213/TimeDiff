"""SPX close/high/low features and leakage-safe 63 -> 21 windows."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_LEN, PRED_LEN = 63, 21
FEATURES = ['rclose', 'rhigh', 'rlow']


def market_statistics(log_returns):
    """Last axis contains daily close log returns; volatility uses ddof=1."""
    r = np.asarray(log_returns, dtype=np.float64)
    if r.shape[-1] < 2 or not np.isfinite(r).all():
        raise ValueError('Need at least two finite daily log returns')
    return np.stack([np.expm1(r.sum(axis=-1)), r.std(axis=-1, ddof=1) * np.sqrt(252)], axis=-1)


def download_spx(path, start, end):
    import yfinance as yf
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} already exists; use a new path to refresh the cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(path.parent / 'yfinance_cache'))
    raw = yf.download('^GSPC', start=start, end=end, interval='1d',
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError('Yahoo returned no SPX data')
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs('^GSPC', axis=1, level=1)
    prices = raw[['Close', 'High', 'Low']].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = 'date'
    validate_prices(prices)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    path.with_suffix('.metadata.json').write_text(json.dumps({
        'ticker': '^GSPC', 'start_inclusive': start, 'end_exclusive': end,
        'auto_adjust': False, 'interval': '1d', 'yfinance_version': yf.__version__,
        'downloaded_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'rows': len(prices), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    return prices


def validate_prices(prices):
    if prices.empty or not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError('Price dates must be nonempty, sorted and unique')
    values = prices[['Close', 'High', 'Low']].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('SPX prices must be positive and finite; do not forward-fill missing prices')
    if (prices.High < prices.Close).any() or (prices.Low > prices.Close).any():
        raise ValueError('SPX requires Low <= Close <= High')


def load_features(path):
    prices = pd.read_csv(path, parse_dates=['date']).set_index('date')
    validate_prices(prices)
    features = pd.DataFrame({
        'rclose': np.log(prices.Close / prices.Close.shift(1)),
        'rhigh': np.log(prices.High / prices.Close),
        'rlow': np.log(prices.Low / prices.Close),
    }, index=prices.index).iloc[1:]
    return prices, features


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
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}


class SPXWindows(Dataset):
    def __init__(self, features, starts, feature_scaler, condition_scaler):
        self.raw = features[FEATURES].to_numpy(dtype=np.float64)
        self.values = feature_scaler.transform(self.raw)
        self.starts = list(starts)
        self.condition_scaler = condition_scaler

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = self.starts[index]
        history = self.values[s-SEQ_LEN:s]
        target = self.values[s:s+PRED_LEN]
        condition = self.condition_scaler.transform(market_statistics(self.raw[s:s+PRED_LEN, 0]))
        return tuple(torch.from_numpy(v.copy()) for v in (history, target, condition))


def split_windows(features, train_end, val_end):
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end >= val_end:
        raise ValueError('train_end must precede val_end')
    dates = features.index
    groups = {'train': [], 'validation': [], 'test': []}
    for s in range(SEQ_LEN, len(features) - PRED_LEN + 1):
        first, last = dates[s], dates[s+PRED_LEN-1]
        if last <= train_end:
            groups['train'].append(s)
        elif first > train_end and last <= val_end:
            groups['validation'].append(s)
        elif first > val_end:
            groups['test'].append(s)
        # Targets straddling a split boundary are deliberately excluded.
    if not groups['train'] or not groups['validation']:
        raise ValueError('Not enough data for train/validation windows')
    return groups
