#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

try:
    import ta  # noqa: F401
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
except Exception as e:  # pragma: no cover
    raise SystemExit("The 'ta' package is required. Try: pip install ta") from e


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)


def _map_barchart_to_yahoo(symbol: str) -> Optional[str]:
    """Convert Barchart symbol like 'MKA.VN' or 'URM.CN' to Yahoo Finance.

    Returns None if mapping not possible.
    """
    if not symbol or "." not in symbol:
        return None
    base, exch = symbol.split(".", 1)
    exch_map = {
        "VN": "V",   # TSX Venture
        "CN": "CN",  # CSE
        "TO": "TO",  # TSX
        "NE": "NE",  # NEO Exchange
    }
    yahoo_exch = exch_map.get(exch.upper())
    if not yahoo_exch:
        return None
    return f"{base}.{yahoo_exch}"


def fetch_barchart_top_canadian(limit: int = 120) -> List[str]:
    """Scrape Barchart Top Canadian stocks and return Yahoo tickers.

    Falls back to empty list if the page structure blocks scraping.
    """
    url = (
        "https://www.barchart.com/ca/stocks/top-100-stocks"
        "?viewName=main&orderBy=percentChange&orderDir=desc"
    )
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        html = resp.text
        # Two common patterns observed: /quotes/TICKER:EXCH and /quotes/TICKER.EXCH
        pattern_colon = re.findall(r"/ca/stocks/quotes/([A-Z0-9]+):([A-Z]{2})", html)
        pattern_dot = re.findall(r"/ca/stocks/quotes/([A-Z0-9]+)\.([A-Z]{2})", html)
        raw = {f"{s}.{e}" for s, e in [*pattern_colon, *pattern_dot]}
        tickers: List[str] = []
        for sym in raw:
            ysym = _map_barchart_to_yahoo(sym)
            if ysym:
                tickers.append(ysym)
        tickers = list(dict.fromkeys(tickers))  # de-duplicate, stable order
        if limit and len(tickers) > limit:
            tickers = tickers[:limit]
        return tickers
    except Exception:
        return []


def fetch_tsx60_from_wikipedia() -> List[str]:
    """Fetch S&P/TSX 60 constituents from Wikipedia, return Yahoo tickers.

    This is a robust fallback universe if Barchart scraping fails.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P/TSX_60")
        for table in tables:
            cols = [c.lower() for c in table.columns]
            if any("symbol" in c or "ticker" in c for c in cols):
                # Normalize symbol column name
                for c in table.columns:
                    if str(c).lower() in ("symbol", "ticker", "ticker symbol"):
                        sym_col = c
                        break
                else:
                    continue
                syms = (
                    table[sym_col]
                    .astype(str)
                    .str.replace("TSX:", "", regex=False)
                    .str.replace("TSE:", "", regex=False)
                    .str.replace(".TO", "", regex=False)
                    .str.strip()
                )
                # remove rows that look like notes
                syms = syms[syms.str.fullmatch(r"[A-Za-z\.\-]+")]
                tickers = [f"{s}.TO" for s in syms if s and s.upper() == s]
                tickers = list(dict.fromkeys(tickers))
                if tickers:
                    return tickers
        return []
    except Exception:
        return []


def default_canadian_universe() -> List[str]:
    """Static large-cap TSX list as last-resort fallback."""
    return [
        # Banks
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
        # Energy & materials
        "CNQ.TO", "SU.TO", "ENB.TO", "TRP.TO", "CVE.TO", "TOU.TO", "MEG.TO",
        "ABX.TO", "AEM.TO", "NTR.TO", "LUN.TO", "TECK-B.TO",
        # Industrials & rails
        "CNR.TO", "CP.TO", "TFII.TO", "WSP.TO",
        # Tech & communications
        "SHOP.TO", "BCE.TO", "T.TO", "QBR-B.TO", "BYD.TO",
        # Utilities & pipelines
        "EMA.TO", "AQN.TO", "FTS.TO", "BEP-UN.TO", "BEPC.TO",
        # Consumer
        "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "WN.TO",
        # Real estate & others
        "BAM.TO", "BN.TO", "GWO.TO", "MFC.TO", "SLF.TO",
    ]


def get_universe() -> List[str]:
    tickers = fetch_barchart_top_canadian(limit=120)
    if not tickers:
        tickers = fetch_tsx60_from_wikipedia()
    if not tickers:
        tickers = default_canadian_universe()
    # Ensure unique and reasonable length
    tickers = list(dict.fromkeys(tickers))
    return tickers


@dataclass
class BuildConfig:
    lookback_days: int = 400
    target_threshold: float = 0.10  # 10% next-day gain to be considered top performer
    min_history: int = 120          # minimum bars required to use a ticker
    max_tickers: int = 150          # hard cap to control runtime


FEATURE_COLUMNS = [
    # Price action
    "return_1d", "return_5d", "volatility_10", "intraday_range", "gap_pct",
    # Trend
    "ema10_dist", "ema20_dist", "ema50_dist",
    "sma10_dist", "sma20_dist", "sma50_dist",
    # Momentum
    "rsi_14", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_diff",
    # Volatility / liquidity
    "atr_14", "bb_high_dist", "bb_low_dist", "bb_pct_b", "volume_z",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical features for a single-ticker OHLCV DataFrame.

    Assumes columns: [Open, High, Low, Close, Adj Close, Volume]
    Index is DatetimeIndex.
    """
    data = df.copy()
    if data.isna().all().all():
        return pd.DataFrame()

    # yfinance can sometimes return single-column DataFrames; ensure Series
    def as_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(obj, pd.DataFrame):
            return obj.iloc[:, 0].astype(float)
        return obj.astype(float)

    close = as_series(data["Adj Close"])  # type: ignore[index]
    high = as_series(data["High"])        # type: ignore[index]
    low = as_series(data["Low"])          # type: ignore[index]
    open_ = as_series(data["Open"])       # type: ignore[index]
    vol = as_series(data["Volume"])       # type: ignore[index]

    # Basic returns and volatility
    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)
    data["volatility_10"] = data["return_1d"].rolling(10).std() * np.sqrt(252)

    # Intraday measures
    data["intraday_range"] = (high - low) / close
    # Gap between today's open and yesterday's close
    data["gap_pct"] = (open_ - close.shift(1)) / close.shift(1)

    # Trend indicators
    ema10 = EMAIndicator(close, window=10).ema_indicator()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    sma10 = SMAIndicator(close, window=10).sma_indicator()
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()

    for name, series in [("ema10", ema10), ("ema20", ema20), ("ema50", ema50),
                         ("sma10", sma10), ("sma20", sma20), ("sma50", sma50)]:
        data[f"{name}_dist"] = (close - series) / close

    # Momentum
    rsi14 = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    macd = MACD(close)
    data["rsi_14"] = rsi14
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()

    # Volatility
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    data["atr_14"] = atr
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    data["bb_high_dist"] = (bb_high - close) / close
    data["bb_low_dist"] = (close - bb_low) / close
    denom = (bb.bollinger_hband() - bb.bollinger_lband())
    denom = denom.replace(0, np.nan)
    data["bb_pct_b"] = (close - bb.bollinger_lband()) / denom

    # Liquidity: volume z-score
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    data["volume_z"] = (vol - vol_mean) / vol_std

    return data


def add_target(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add binary target: next-day percent change >= threshold."""
    out = data.copy()
    close = out["Adj Close"].astype(float)
    next_close = close.shift(-1)
    out["target"] = ((next_close / close - 1.0) >= threshold).astype(int)
    return out


def download_history(ticker: str, lookback_days: int) -> pd.DataFrame:
    start = (dt.datetime.utcnow() - dt.timedelta(days=lookback_days * 2)).date()
    try:
        df = yf.download(
            ticker,
            start=str(start),
            end=None,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Drop timezone-naive duplicates and ensure DatetimeIndex
            df = df.reset_index()
            if "Date" in df.columns:
                df.rename(columns={"Date": "Date"}, inplace=True)
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                df.set_index("Date", inplace=True)
            # Flatten any MultiIndex columns from yfinance single-ticker quirk
            if isinstance(df.columns, pd.MultiIndex):
                # If only one ticker level, keep the price level (Open, High, ...)
                if df.columns.nlevels >= 2 and len(df.columns.get_level_values(1).unique()) == 1:
                    df.columns = df.columns.get_level_values(0)
                else:
                    # Fall back to joining names with '_'
                    df.columns = ["_".join([str(c) for c in col]).strip("_") for col in df.columns]
            # Remove duplicate columns, keep first occurrence
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            # Ensure expected columns exist; fallback to Close if Adj Close missing
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            # Keep only the last 'lookback_days'
            df = df.tail(lookback_days)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def build_dataset(tickers: List[str], cfg: BuildConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    latest_rows: List[pd.DataFrame] = []

    limited = tickers[: cfg.max_tickers]
    print(f"Building dataset from {len(limited)} tickers...")
    for t in tqdm(limited):
        hist = download_history(t, cfg.lookback_days)
        if hist.empty or len(hist) < cfg.min_history:
            continue
        feats = compute_features(hist)
        feats = add_target(feats, cfg.target_threshold)
        feats["ticker"] = t
        # model rows drop NaNs
        model_rows = feats.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
        if model_rows.empty:
            continue
        rows.append(model_rows)
        # latest row for prediction (last date with all features not NaN)
        last = model_rows.tail(1).copy()
        latest_rows.append(last)

    if not rows:
        raise RuntimeError("No usable data built. Try expanding the universe or lookback.")

    dataset = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    latest = pd.concat(latest_rows).reset_index().rename(columns={"index": "date"})
    return dataset, latest


def _positive_class_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Return probability of class 1 for any classifier even if only one class seen.

    If the model was trained on a single class, scikit-learn returns a single
    probability column. In that case, emit all-zeros (class 1 absent) or
    all-ones (class 1 only) as appropriate.
    """
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([]))
    if proba.ndim == 2 and proba.shape[1] == 2:
        # Map to index of class 1
        idx = np.where(classes == 1)[0]
        col = int(idx[0]) if len(idx) else 1
        return proba[:, col]
    # Single-class model fallback
    if len(classes) == 1 and int(classes[0]) == 1:
        return np.ones(X.shape[0])
    return np.zeros(X.shape[0])


def train_and_predict(dataset: pd.DataFrame, latest: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    X = dataset[FEATURE_COLUMNS].values
    y = dataset["target"].values.astype(int)

    # Compute class weights to mitigate rare-event imbalance
    y_unique = np.unique(y)
    if len(y_unique) >= 2:
        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {int(c): w for c, w in zip(classes, weights)}
    else:
        class_weight = None

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=42,
    )

    # TimeSeries cross-validation for a quick sanity metric
    tscv = TimeSeriesSplit(n_splits=5)
    aucs: List[float] = []
    for train_idx, test_idx in tscv.split(X):
        # Skip folds without both classes in training or testing
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        proba = _positive_class_proba(model, X[test_idx])
        try:
            aucs.append(roc_auc_score(y[test_idx], proba))
        except ValueError:
            pass
    if aucs:
        print(f"TimeSeries CV ROC-AUC: mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}")

    # Fit on full history and predict for the latest day per ticker
    model.fit(X, y)

    latest_X = latest[FEATURE_COLUMNS].values
    latest_proba = _positive_class_proba(model, latest_X)

    latest_out = latest.copy()
    latest_out["pred_prob_big_up_move"] = latest_proba
    latest_out.sort_values("pred_prob_big_up_move", ascending=False, inplace=True)

    # Select output columns
    keep_cols = [
        "date", "ticker", "pred_prob_big_up_move", "Adj Close", "Volume",
        "rsi_14", "atr_14", "bb_pct_b", "return_1d", "return_5d",
    ]
    missing_keep = [c for c in keep_cols if c not in latest_out.columns]
    for c in missing_keep:
        latest_out[c] = np.nan
    latest_out = latest_out[keep_cols]

    latest_out.rename(columns={"Adj Close": "last_adj_close"}, inplace=True)
    latest_out.reset_index(drop=True, inplace=True)

    latest_out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

    # Also display top 20
    print(latest_out.head(20))
    return latest_out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict Canadian stocks likely to be top performers tomorrow "
            "using technical-feature RandomForest on recent history."
        )
    )
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Next-day percent gain threshold for 'top performer' label. Default 0.10 (10%).")
    parser.add_argument("--max-tickers", type=int, default=150,
                        help="Max number of tickers to include for speed.")
    parser.add_argument("--lookback", type=int, default=400,
                        help="Lookback days of daily history to build features.")
    parser.add_argument("--universe", type=str, default="auto",
                        help="Comma-separated Yahoo tickers, or 'auto' to scrape Barchart with fallbacks.")
    parser.add_argument("--output", type=str, default="predictions_canada.csv",
                        help="Output CSV path for ranked predictions.")

    args = parser.parse_args()

    if args.universe == "auto":
        tickers = get_universe()
    else:
        tickers = [t.strip() for t in args.universe.split(',') if t.strip()]

    if not tickers:
        raise SystemExit("No tickers available to build a dataset.")

    cfg = BuildConfig(
        lookback_days=int(args.lookback),
        target_threshold=float(args.threshold),
        max_tickers=int(args.max_tickers),
    )

    print(f"Universe size: {len(tickers)} tickers")
    dataset, latest = build_dataset(tickers, cfg)

    print(
        f"Built dataset with {len(dataset)} rows across {latest['ticker'].nunique()} tickers."
    )

    _ = train_and_predict(dataset, latest, out_csv=args.output)


if __name__ == "__main__":
    main()
