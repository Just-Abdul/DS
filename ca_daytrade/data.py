from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def download_history_for_symbol(symbol: str, period: str = "3y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        # Standardize column names
        hist = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]
        hist.index.name = "date"
        hist["symbol"] = symbol
        # Drop duplicate dates if present
        hist = hist[~hist.index.duplicated(keep="last")]
        return hist
    except Exception:
        return None


def download_histories(symbols: List[str], period: str = "3y", interval: str = "1d", max_workers: int = 10) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_history_for_symbol, s, period, interval): s for s in symbols}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading OHLCV"):
            sym = futs[fut]
            df = fut.result()
            if df is not None and len(df) >= 60:
                results[sym] = df
    return results
