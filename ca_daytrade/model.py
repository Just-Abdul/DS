from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    # returns
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    # MAs
    "sma_ratio_5",
    "sma_ratio_10",
    "sma_ratio_20",
    "sma_ratio_50",
    # MACD
    "macd",
    "macd_signal",
    "macd_hist",
    # RSI
    "rsi_14",
    # Bollinger
    "bb_width",
    # Volatility
    "atr_ratio",
    # Volume
    "rel_volume_20",
    # day-of-week encodings
    "dow_sin",
    "dow_cos",
]


@dataclass
class TrainedModel:
    model: RandomForestClassifier
    scaler: StandardScaler
    features: List[str]


def build_training_frame(histories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sym, df in histories.items():
        frames.append(df.copy())
    all_df = pd.concat(frames, axis=0, ignore_index=False)
    # Drop rows with any NaNs in feature set later
    return all_df


def train_classifier(df: pd.DataFrame) -> Tuple[TrainedModel, float]:
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_spike"])  # keep only rows with all features
    if df.empty:
        raise ValueError("No training data after dropping NaNs")

    X = df[FEATURE_COLUMNS].values
    y = df["target_spike"].values.astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Split by time (approx) using index order; but many symbols interleaved.
    # Simple random split to keep things straightforward.
    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_probs)

    return TrainedModel(model=clf, scaler=scaler, features=FEATURE_COLUMNS), float(ap)


def predict_for_latest(histories: Dict[str, pd.DataFrame], trained: TrainedModel) -> pd.DataFrame:
    records = []
    for sym, df in histories.items():
        last = df.dropna(subset=trained.features).iloc[-1:]
        if last.empty:
            continue
        X = last[trained.features].values
        Xs = trained.scaler.transform(X)
        prob = trained.model.predict_proba(Xs)[0, 1]
        records.append({
            "symbol": sym,
            "date": last.index[0],
            "prob_spike": float(prob),
        })
    return pd.DataFrame.from_records(records).sort_values("prob_spike", ascending=False).reset_index(drop=True)
