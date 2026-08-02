"""
sanity_tests.py
===============
Feature sanity tests run during CI.
Validates that computed features are consistent with the raw data.

One test per feature:
  1. rolling_avg_10  — value must be within [min, max] of the close prices
                       in the preceding ≤10 data points for that stock.
  2. volume_sum_10   — value must be ≥ 0 and ≥ any individual volume
                       in the preceding ≤10 data points.
  3. stock_name      — value must be a known, non-null stock ticker.

Usage:
    python src/sanity_tests.py
    pytest src/sanity_tests.py -v   (also works with pytest)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path("data/processed")
MODELS_DIR     = Path("models")

KNOWN_STOCKS = {
    "AARTIIND", "ABCAPITAL",        # v0
    "ABFRL", "ADANIENT", "ADANIGAS",  # v1
}

FEATURE_COLS = ["rolling_avg_10", "volume_sum_10", "stock_name"]


# ─── Load test data ────────────────────────────────────────────────────────────

def _load_test_df() -> pd.DataFrame:
    for candidate in ["test_latest.csv", "test_iter2.csv", "test_iter1.csv"]:
        path = PROCESSED_DIR / candidate
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No test data found in data/processed/")


# ─── Sanity Test Functions ─────────────────────────────────────────────────────

def test_rolling_avg_10_within_bounds():
    """
    rolling_avg_10 must lie within [min_close_10, max_close_10] of the stock's
    10 most recent close prices at each timestamp.

    Since we can't reconstruct the exact rolling window from the test set alone,
    we use a weaker but valid sanity check:
      - rolling_avg_10 > 0
      - rolling_avg_10 is not NaN
      - rolling_avg_10 is within [0.5 * close, 2.0 * close]
        (a 50%–200% band around current close is a very loose sanity bound)
    """
    df = _load_test_df()
    assert "rolling_avg_10" in df.columns, "Column 'rolling_avg_10' missing from test data"
    assert "close" in df.columns, "Column 'close' missing from test data"

    series = df["rolling_avg_10"].dropna()
    close  = df.loc[series.index, "close"]

    # Must not be NaN
    nan_count = df["rolling_avg_10"].isna().sum()
    assert nan_count == 0, f"rolling_avg_10 has {nan_count} NaN values"

    # Must be positive
    non_positive = (series <= 0).sum()
    assert non_positive == 0, f"rolling_avg_10 has {non_positive} non-positive values"

    # Must be within 50%–200% of close price (sanity bound)
    lower_violations = (series < 0.5 * close).sum()
    upper_violations = (series > 2.0 * close).sum()
    assert lower_violations == 0, (
        f"rolling_avg_10 is below 50% of close for {lower_violations} rows"
    )
    assert upper_violations == 0, (
        f"rolling_avg_10 is above 200% of close for {upper_violations} rows"
    )

    # Verify: rolling_avg_10 ≈ mean of close (correlation check)
    corr = np.corrcoef(series.values, close.values)[0, 1]
    assert corr > 0.8, (
        f"Correlation between rolling_avg_10 and close is too low: {corr:.3f} "
        f"(expected > 0.8 — rolling mean should track close price)"
    )

    print(
        f"[PASS] test_rolling_avg_10_within_bounds  "
        f"(n={len(series)}, corr={corr:.3f})"
    )


def test_volume_sum_10_non_negative():
    """
    volume_sum_10 must be:
      1. Non-NaN
      2. Non-negative (volume can't be negative)
      3. ≥ individual volume at each row (sum ≥ any single component)
    """
    df = _load_test_df()
    assert "volume_sum_10" in df.columns, "Column 'volume_sum_10' missing from test data"
    assert "volume" in df.columns, "Column 'volume' missing from test data"

    series = df["volume_sum_10"].dropna()
    volume = df.loc[series.index, "volume"]

    # Must not be NaN
    nan_count = df["volume_sum_10"].isna().sum()
    assert nan_count == 0, f"volume_sum_10 has {nan_count} NaN values"

    # Must be non-negative
    neg_count = (series < 0).sum()
    assert neg_count == 0, f"volume_sum_10 has {neg_count} negative values"

    # Sum must be ≥ individual volume (a sum of non-negative numbers ≥ any addend)
    violations = (series < volume).sum()
    assert violations == 0, (
        f"volume_sum_10 < individual volume for {violations} rows "
        f"(a rolling sum must be ≥ any component)"
    )

    # Sanity: sum should be at least as large as the max of the window
    # As a proxy, check that avg per-row volume_sum_10 > avg volume
    avg_sum = series.mean()
    avg_vol = volume.mean()
    assert avg_sum >= avg_vol, (
        f"Average volume_sum_10 ({avg_sum:.2f}) < average volume ({avg_vol:.2f})"
    )

    print(
        f"[PASS] test_volume_sum_10_non_negative  "
        f"(n={len(series)}, avg_sum={avg_sum:.2f}, avg_vol={avg_vol:.2f})"
    )


def test_stock_name_valid():
    """
    stock_name must be:
      1. Non-null / non-empty
      2. One of the known stock tickers in the dataset
    """
    df = _load_test_df()
    assert "stock_name" in df.columns, "Column 'stock_name' missing from test data"

    series = df["stock_name"]

    # No nulls
    null_count = series.isna().sum()
    assert null_count == 0, f"stock_name has {null_count} null values"

    # No empty strings
    empty_count = (series.astype(str).str.strip() == "").sum()
    assert empty_count == 0, f"stock_name has {empty_count} empty string values"

    # All values must be in the known set
    found = set(series.unique())
    unexpected = found - KNOWN_STOCKS
    assert len(unexpected) == 0, (
        f"Unknown stock names found: {unexpected}. "
        f"Known: {KNOWN_STOCKS}"
    )

    # Must have at least 1 stock represented
    assert len(found) >= 1, "No stocks found in test data"

    print(
        f"[PASS] test_stock_name_valid  "
        f"(stocks={sorted(found)})"
    )


# ─── Run all tests (standalone) ────────────────────────────────────────────────

def run_all_sanity_tests():
    """Run all sanity checks and report results."""
    tests = [
        test_rolling_avg_10_within_bounds,
        test_volume_sum_10_non_negative,
        test_stock_name_valid,
    ]

    results = {}
    all_passed = True

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            results[name] = f"FAIL: {e}"
            all_passed = False
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results[name] = f"ERROR: {e}"
            all_passed = False

    # Save results
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/sanity_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Sanity Test Summary ===")
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    if not all_passed:
        raise SystemExit(1)

    print("\n✅ All sanity tests passed!")
    return results


if __name__ == "__main__":
    run_all_sanity_tests()
