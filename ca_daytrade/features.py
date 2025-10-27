from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for a single ticker dataframe.

    Expects columns: open, high, low, close, volume; index is datetime.
    Returns a new dataframe with feature columns and the original columns.
    """
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # Returns
    out["ret_1d"] = close.pct_change()
    out["ret_3d"] = close.pct_change(3)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)

    # Moving averages and ratios
    for w in (5, 10, 20, 50):
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"sma_ratio_{w}"] = close / out[f"sma_{w}"]

    # Exponential moving averages and MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # RSI (14)
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(14).mean()
    roll_down = pd.Series(down, index=close.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Bollinger Bands width (20, 2)
    sma20 = out["sma_20"]
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_width"] = (upper - lower) / sma20

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_ratio"] = out["atr_14"] / close

    # Volume features
    out["vol_ma_20"] = volume.rolling(20).mean()
    out["rel_volume_20"] = volume / (out["vol_ma_20"] + 1e-9)

    # Day of week cyclical encoding
    dow = out.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return out


def add_labels_next_day_spike(df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Add binary label: 1 if next-day return >= threshold.

    threshold: e.g., 0.10 for +10% next day.
    """
    out = df.copy()
    next_close = out["close"].shift(-1)
    out["target_spike"] = (next_close / out["close"] - 1.0) >= threshold
    out["target_spike"] = out["target_spike"].astype(int)
    return out
