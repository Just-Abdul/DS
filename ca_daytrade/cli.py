from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .scrape import get_top100_yahoo_symbols
from .data import download_histories
from .features import compute_features, add_labels_next_day_spike
from .model import build_training_frame, train_classifier, predict_for_latest


def _load_tickers_from_file(path: Path) -> List[str]:
    text = Path(path).read_text().strip()
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    # Otherwise treat as newline or comma separated
    tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    return tokens


def run(threshold: float, period: str, interval: str, out_csv: Path, tickers: Optional[List[str]], topk: int) -> None:
    # 1) Universe
    if not tickers:
        print("Fetching Top 100 CA tickers from Barchart ...")
        try:
            tickers = get_top100_yahoo_symbols()
        except Exception as e:
            raise SystemExit(
                f"Failed to fetch Barchart Top 100: {e}. Provide --tickers or --tickers-file."
            )

    print(f"Universe size: {len(tickers)}")

    # 2) Download data
    histories = download_histories(tickers, period=period, interval=interval)
    if not histories:
        raise SystemExit("No OHLCV histories were downloaded. Exiting.")

    # 3) Feature engineering + labels
    for sym, df in list(histories.items()):
        fe = compute_features(df)
        fe = add_labels_next_day_spike(fe, threshold=threshold)
        histories[sym] = fe

    # 4) Assemble training frame
    train_df = build_training_frame(histories)

    # Drop the last row per symbol from training (no known label for final row)
    last_rows = train_df.groupby("symbol").tail(1).index
    train_df_no_last = train_df.drop(index=last_rows)

    # 5) Train
    model, ap = train_classifier(train_df_no_last)
    print(f"Validation Average Precision (PR AUC): {ap:.4f}")

    # 6) Predict latest
    preds = predict_for_latest(histories, model)
    preds["rank"] = preds.reset_index().index + 1

    # 7) Save and print top-k
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_csv, index=False)

    print("Top candidates:")
    print(preds.head(topk).to_string(index=False))
    print(f"\nSaved full predictions to: {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Predict next-day Canadian big gainers from technicals.")
    p.add_argument("--threshold", type=float, default=0.10, help="Next-day return threshold, e.g., 0.10 for +10%.")
    p.add_argument("--period", type=str, default="3y", help="History period for yfinance (e.g., 1y, 2y, 3y, max).")
    p.add_argument("--interval", type=str, default="1d", help="History interval (1d recommended).")
    p.add_argument("--out", type=Path, default=Path("predictions_ca_top100.csv"), help="Output CSV path.")
    p.add_argument("--tickers", type=str, nargs="*", help="Optional space-separated list of tickers (Yahoo format).")
    p.add_argument("--tickers-file", type=Path, default=None, help="Optional file with tickers (csv, txt, or json list).")
    p.add_argument("--topk", type=int, default=20, help="Show top-k candidates in the console.")

    args = p.parse_args()

    provided_tickers: Optional[List[str]] = None
    if args.tickers_file:
        provided_tickers = _load_tickers_from_file(args.tickers_file)
    elif args.tickers:
        provided_tickers = args.tickers

    run(
        threshold=args.threshold,
        period=args.period,
        interval=args.interval,
        out_csv=args.out,
        tickers=provided_tickers,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
