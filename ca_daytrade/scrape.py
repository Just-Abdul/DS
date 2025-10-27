from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import requests


@dataclass
class TopListRow:
    symbol: str
    name: Optional[str]
    percent_change: Optional[float]
    price: Optional[float]
    exchange: Optional[str]


BC_API = "https://www.barchart.com/proxies/core-api/v1/quotes/get"
BC_TOP100_PARAMS = {
    "lists": "stocks.top100.ca",
    "orderBy": "percentChange",
    "orderDir": "desc",
    "limit": "100",
    "page": "1",
    "quoteType": "stocks",
    "country": "CA",
    "meta": "field.shortName,field.type",
    "fields": ",".join(
        [
            "symbol",
            "symbolCode",
            "exchange",
            "shortName",
            "lastPrice",
            "percentChange",
            "priceChange",
            "tradeTimestamp",
            "volume",
            "averageVolume",
        ]
    ),
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.barchart.com/ca/stocks/top-100-stocks",
}


YAHOO_SUFFIX_MAP = {
    ".VN": ".V",  # TSX Venture on Barchart -> Yahoo
    ".V": ".V",   # already Yahoo style (rare)
    ".TO": ".TO",  # TSX
    ".CN": ".CN",  # CSE
    ".NE": ".NE",  # NEO Exchange (Cboe CA)
}


def map_barchart_symbol_to_yahoo(symbol: str) -> str:
    for bc, yf in YAHOO_SUFFIX_MAP.items():
        if symbol.endswith(bc):
            return symbol[: -len(bc)] + yf
    # Heuristic: if no suffix is present, assume TSX (rare in CA lists)
    if "." not in symbol:
        return symbol + ".TO"
    return symbol


def fetch_top100_rows(session: Optional[requests.Session] = None) -> List[TopListRow]:
    """Fetch top-100 CA stocks list from Barchart.

    Tries the JSON API first; falls back to parsing HTML if needed.
    """
    sess = session or requests.Session()

    # Try the JSON API endpoint
    try:
        resp = sess.get(BC_API, params=BC_TOP100_PARAMS, headers=DEFAULT_HEADERS, timeout=20)
        if resp.ok:
            payload = resp.json()
            data = payload.get("data", [])
            rows: List[TopListRow] = []
            for item in data:
                rows.append(
                    TopListRow(
                        symbol=item.get("symbol"),
                        name=item.get("shortName"),
                        percent_change=(
                            float(item.get("percentChange")) if item.get("percentChange") is not None else None
                        ),
                        price=(float(item.get("lastPrice")) if item.get("lastPrice") is not None else None),
                        exchange=item.get("exchange"),
                    )
                )
            if rows:
                return rows
    except Exception:
        pass

    # Fallback: try to parse HTML for a table
    try:
        url = (
            "https://www.barchart.com/ca/stocks/top-100-stocks?viewName=main&orderBy=percentChange&orderDir=desc"
        )
        html = sess.get(url, headers={**DEFAULT_HEADERS, "Accept": "text/html"}, timeout=25).text

        # Try to find embedded JSON (window.__BC_DATA__ or Next.js data)
        m = re.search(r"window\.__BC_DATA__\s*=\s*(\{.*?\});", html, flags=re.S)
        if m:
            try:
                bc_data = json.loads(m.group(1))
                # Attempt to locate quotes list
                entries = (
                    bc_data.get("state", {})
                    .get("scans", {})
                    .get("table", {})
                    .get("rows", [])
                )
                rows = []
                for e in entries:
                    rows.append(
                        TopListRow(
                            symbol=e.get("symbol"),
                            name=e.get("name"),
                            percent_change=e.get("percentChange"),
                            price=e.get("lastPrice"),
                            exchange=e.get("exchange"),
                        )
                    )
                if rows:
                    return rows
            except Exception:
                pass

        # As a last resort, try pandas.read_html to locate an HTML table with a Symbol column
        tables = pd.read_html(html)
        for df in tables:
            if {"Symbol", "Name"}.issubset(set(df.columns)):
                rows = []
                for _, r in df.iterrows():
                    rows.append(
                        TopListRow(
                            symbol=str(r.get("Symbol")),
                            name=str(r.get("Name")),
                            percent_change=float(r.get("%Change")) if "%Change" in df.columns else None,
                            price=float(r.get("Last")) if "Last" in df.columns else None,
                            exchange=None,
                        )
                    )
                if rows:
                    return rows
    except Exception:
        pass

    raise RuntimeError(
        "Unable to fetch Top 100 CA stocks from Barchart. If this persists, "
        "pass --tickers or --tickers-file to the CLI."
    )


def get_top100_yahoo_symbols(session: Optional[requests.Session] = None) -> List[str]:
    rows = fetch_top100_rows(session=session)
    symbols = []
    for r in rows:
        if not r.symbol:
            continue
        symbols.append(map_barchart_symbol_to_yahoo(r.symbol))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for s in symbols:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq
