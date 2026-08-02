"""
data_prep.py
============
Loads raw CSV stock data, computes rolling features and target, then saves
processed train/test splits for a given iteration.

Usage:
    python src/data_prep.py --iteration 1   # v0 data only
    python src/data_prep.py --iteration 2   # v0 + v1 data merged
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Config ────────────────────────────────────────────────────────────────────

RAW_BASE = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

V0_STOCKS = ["AARTIIND", "ABCAPITAL"]
V1_STOCKS = ["ABFRL", "ADANIENT", "ADANIGAS"]
FILE_SUFFIX = "__EQ__NSE__NSE__MINUTE.csv"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_stock_csv(filepath: Path, stock_name: str) -> pd.DataFrame:
    """Load a single stock CSV, parse timestamp, add stock_name column."""
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    # Ensure timezone-aware (strip tz if present to normalise)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
    df["stock_name"] = stock_name
    # Sort chronologically — data is NOT guaranteed to be sorted
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per stock_name, compute rolling features using the last 10 minutes of data.

    Features:
        rolling_avg_10  — 10-min rolling mean of close price
        volume_sum_10   — 10-min rolling sum of volume

    Target:
        target — 1 if close[t+5 rows] > close[t], else 0
                 (uses available data; shift(-5) per stock, chronologically)
    """
    results = []

    for stock, grp in df.groupby("stock_name", sort=False):
        grp = grp.sort_values("timestamp").copy()
        grp = grp.set_index("timestamp")

        # Forward-fill missing values
        grp.ffill(inplace=True)

        # Rolling features on time-based window (10 minutes)
        grp["rolling_avg_10"] = (
            grp["close"].rolling(window="10min", min_periods=1).mean()
        )
        grp["volume_sum_10"] = (
            grp["volume"].rolling(window="10min", min_periods=1).sum()
        )

        # Target: 1 if price is higher 5 rows in future (chronological)
        grp["close_5min_future"] = grp["close"].shift(-5)
        grp["target"] = (grp["close_5min_future"] > grp["close"]).astype(int)
        grp.drop(columns=["close_5min_future"], inplace=True)

        grp = grp.reset_index()
        results.append(grp)

    return pd.concat(results, ignore_index=True)


def split_train_test(df: pd.DataFrame, test_rows_per_stock: int = 20):
    """
    Chronological train/test split per stock.
    Last `test_rows_per_stock` rows (after removing NaN targets) per stock → test.
    """
    train_parts, test_parts = [], []

    for stock, grp in df.groupby("stock_name", sort=False):
        grp = grp.sort_values("timestamp")
        grp_clean = grp.dropna(subset=["target", "rolling_avg_10", "volume_sum_10"]).copy()
        test_parts.append(grp_clean.tail(test_rows_per_stock))
        train_parts.append(grp_clean.iloc[:-test_rows_per_stock])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration", type=int, choices=[1, 2], required=True,
        help="1 = v0 data only | 2 = v0 + v1 merged"
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which stocks to load
    stocks_to_load = V0_STOCKS.copy()
    data_versions = ["v0"]
    if args.iteration == 2:
        stocks_to_load += V1_STOCKS
        data_versions = ["v0", "v1"]

    print(f"[data_prep] Iteration {args.iteration} — Loading stocks: {stocks_to_load}")

    dfs = []
    for version in data_versions:
        version_dir = RAW_BASE / version
        for stock in (V0_STOCKS if version == "v0" else V1_STOCKS):
            csv_path = version_dir / f"{stock}{FILE_SUFFIX}"
            if not csv_path.exists():
                raise FileNotFoundError(f"Expected data file not found: {csv_path}")
            print(f"  Loading {csv_path} ...")
            df = load_stock_csv(csv_path, stock)
            dfs.append(df)

    raw_df = pd.concat(dfs, ignore_index=True)
    print(f"[data_prep] Total raw rows: {len(raw_df):,}")

    # Compute features + target
    featured_df = compute_features(raw_df)
    print(f"[data_prep] Featured rows: {len(featured_df):,}")

    # Train / test split
    train_df, test_df = split_train_test(featured_df, test_rows_per_stock=20)
    print(f"[data_prep] Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")

    # Save
    iter_tag = f"iter{args.iteration}"
    train_path = PROCESSED_DIR / f"train_{iter_tag}.csv"
    test_path = PROCESSED_DIR / f"test_{iter_tag}.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[data_prep] Saved → {train_path}")
    print(f"[data_prep] Saved → {test_path}")

    # Also save a combined test set (latest iteration becomes the canonical test set)
    test_df.to_csv(PROCESSED_DIR / "test_latest.csv", index=False)
    print(f"[data_prep] Saved → {PROCESSED_DIR / 'test_latest.csv'}")


if __name__ == "__main__":
    main()
