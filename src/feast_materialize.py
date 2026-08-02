"""
feast_materialize.py
====================
1. Reads processed training data (from data_prep.py output).
2. Writes a parquet file used as Feast's offline FileSource.
3. Runs `feast apply` to register feature definitions.
4. Runs `feast materialize` to populate the online store.
5. Demonstrates a point-in-time correct feature retrieval.

Usage (run from repo root):
    python src/feast_materialize.py --iteration 1
    python src/feast_materialize.py --iteration 2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore


# ─── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path("data/processed")
FEAST_REPO_DIR = Path("feature_store")
FEAST_DATA_DIR = FEAST_REPO_DIR / "data"


# ─── Build parquet for FileSource ─────────────────────────────────────────────

def build_feast_parquet(iteration: int) -> Path:
    """
    Combine processed train files for the given iteration and write a parquet
    file that Feast's FileSource will read.

    Feast requires columns:
        event_timestamp   — datetime with tz
        stock_name        — entity join key
        rolling_avg_10    — feature
        volume_sum_10     — feature
    """
    FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for it in range(1, iteration + 1):
        csv_path = PROCESSED_DIR / f"train_iter{it}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {csv_path}\n"
                f"Run: python src/data_prep.py --iteration {it}"
            )
        frames.append(pd.read_csv(csv_path))

    df = pd.concat(frames, ignore_index=True)

    # Feast requires a timezone-aware event_timestamp column
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")

    feast_df = df[["timestamp", "stock_name", "rolling_avg_10", "volume_sum_10"]].copy()
    feast_df = feast_df.rename(columns={"timestamp": "event_timestamp"})
    feast_df = feast_df.dropna(subset=["rolling_avg_10", "volume_sum_10"])
    feast_df = feast_df.sort_values("event_timestamp")

    parquet_path = FEAST_DATA_DIR / "stock_features.parquet"
    feast_df.to_parquet(parquet_path, index=False)
    print(f"[feast] Parquet written → {parquet_path}  ({len(feast_df):,} rows)")
    return parquet_path


# ─── Feast Apply & Materialize ─────────────────────────────────────────────────

def feast_apply():
    """Run `feast apply` from the feature store repo directory."""
    result = subprocess.run(
        [sys.executable, "-m", "feast", "apply"],
        cwd=str(FEAST_REPO_DIR),
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("feast apply failed")
    print("[feast] feast apply completed.")


def feast_materialize(store: FeatureStore):
    """Materialize all features into the online store."""
    import datetime
    from dateutil.parser import parse as dtparse

    # Determine time range from parquet
    parquet_path = FEAST_DATA_DIR / "stock_features.parquet"
    meta = pd.read_parquet(parquet_path, columns=["event_timestamp"])
    start_dt = meta["event_timestamp"].min().to_pydatetime()
    end_dt   = meta["event_timestamp"].max().to_pydatetime()

    # Ensure timezone-aware
    if start_dt.tzinfo is None:
        import pytz
        tz = pytz.timezone("Asia/Kolkata")
        start_dt = tz.localize(start_dt)
        end_dt   = tz.localize(end_dt)

    print(f"[feast] Materializing from {start_dt} → {end_dt} ...")
    store.materialize(start_date=start_dt, end_date=end_dt)
    print("[feast] Materialization complete.")


# ─── Point-in-time Feature Retrieval Demo ─────────────────────────────────────

def retrieve_training_features(store: FeatureStore, iteration: int) -> pd.DataFrame:
    """
    Retrieve a point-in-time correct feature dataset from Feast.
    Returns a DataFrame with features joined to the training entity timestamps.
    """
    train_csv = PROCESSED_DIR / f"train_iter{iteration}.csv"
    train_df  = pd.read_csv(train_csv)

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], utc=False)
    if train_df["timestamp"].dt.tz is None:
        train_df["timestamp"] = train_df["timestamp"].dt.tz_localize("Asia/Kolkata")

    entity_df = train_df[["timestamp", "stock_name", "target"]].copy()
    entity_df = entity_df.rename(columns={"timestamp": "event_timestamp"})
    entity_df = entity_df.dropna(subset=["target"])

    print(f"[feast] Retrieving features for {len(entity_df):,} entity rows ...")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_features:rolling_avg_10",
            "stock_features:volume_sum_10",
        ],
    ).to_df()

    out_path = PROCESSED_DIR / f"feast_train_iter{iteration}.csv"
    training_data.to_csv(out_path, index=False)
    print(f"[feast] Feast training dataset saved → {out_path}")
    print(f"        Shape: {training_data.shape}")
    print(training_data.head(3))
    return training_data


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration", type=int, choices=[1, 2], required=True,
        help="Which iteration's processed data to load"
    )
    args = parser.parse_args()

    # 1. Build parquet
    build_feast_parquet(args.iteration)

    # 2. feast apply
    feast_apply()

    # 3. Load store & materialize
    store = FeatureStore(repo_path=str(FEAST_REPO_DIR))
    feast_materialize(store)

    # 4. Retrieve point-in-time correct training dataset
    retrieve_training_features(store, args.iteration)

    print("\n[feast] All done ✓")


if __name__ == "__main__":
    main()
