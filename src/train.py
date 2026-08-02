"""
train.py
========
Train a RandomForest classifier on processed stock data.
Logs parameters, metrics, and model artifact to MLflow.

Usage:
    python src/train.py --iteration 1
    python src/train.py --iteration 2

Environment variables (set before running):
    MLFLOW_TRACKING_URI          — e.g. sqlite:///mlflow.db
    MLFLOW_ARTIFACT_ROOT         — e.g. gs://23f2004644-mlops-oppe/mlflow-artifacts
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


# ─── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
REPORTS_DIR   = Path("reports")

FEATURE_COLS  = ["rolling_avg_10", "volume_sum_10"]
ENTITY_COL    = "stock_name"
TARGET_COL    = "target"

MLFLOW_EXPERIMENT = "stock_movement_predictor"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_data(iteration: int):
    """Load train/test CSVs for the given iteration."""
    train_path = PROCESSED_DIR / f"train_iter{iteration}.csv"
    test_path  = PROCESSED_DIR / f"test_iter{iteration}.csv"

    # Fall back to Feast-retrieved training data if available
    feast_path = PROCESSED_DIR / f"feast_train_iter{iteration}.csv"
    if feast_path.exists():
        print(f"[train] Using Feast training data: {feast_path}")
        train_df = pd.read_csv(feast_path)
    else:
        train_df = pd.read_csv(train_path)

    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_features(df: pd.DataFrame, le: LabelEncoder = None, fit_le: bool = False):
    """Encode stock_name + select features. Returns X, y, encoder."""
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    if fit_le:
        le = LabelEncoder()
        le.fit(df[ENTITY_COL].astype(str))

    df["stock_name_enc"] = le.transform(df[ENTITY_COL].astype(str))

    X = df[FEATURE_COLS + ["stock_name_enc"]].values
    y = df[TARGET_COL].astype(int).values
    return X, y, le


# ─── Training ──────────────────────────────────────────────────────────────────

def train(iteration: int, n_estimators: int = 100, max_depth: int = 10):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    artifact_root = os.getenv(
        "MLFLOW_ARTIFACT_ROOT",
        "gs://23f2004644-mlops-oppe/mlflow-artifacts"
    )
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment if not exists
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=artifact_root
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"iteration_{iteration}_rf_n{n_estimators}_d{max_depth}"
    print(f"[train] Starting run: {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        # Load & prepare data
        train_df, test_df = load_data(iteration)
        X_train, y_train, le = prepare_features(train_df, fit_le=True)
        X_test,  y_test,  _  = prepare_features(test_df,  le=le)

        # Log data info
        mlflow.log_param("iteration",    iteration)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",    max_depth)
        mlflow.log_param("train_rows",   len(X_train))
        mlflow.log_param("test_rows",    len(X_test))
        mlflow.log_param("stocks",       list(le.classes_))

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        roc_auc  = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1       : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  roc_auc)

        # Save classification report as artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = REPORTS_DIR / f"classification_report_iter{iteration}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Save metrics JSON for CI
        metrics = {
            "iteration": iteration,
            "accuracy":  accuracy,
            "f1_score":  f1,
            "roc_auc":   roc_auc,
        }
        metrics_path = REPORTS_DIR / f"metrics_iter{iteration}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Log & register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="StockMovementPredictor",
        )

        # Save LabelEncoder classes for CI
        le_path = MODELS_DIR / f"label_encoder_iter{iteration}.json"
        with open(le_path, "w") as f:
            json.dump(list(le.classes_), f)
        mlflow.log_artifact(str(le_path))

        run_id = run.info.run_id
        print(f"[train] Run ID: {run_id}")

        # Persist run_id so CI can look up the model
        run_info = {
            "run_id":    run_id,
            "iteration": iteration,
            "accuracy":  accuracy,
            "f1_score":  f1,
            "roc_auc":   roc_auc,
        }
        run_info_path = MODELS_DIR / f"run_info_iter{iteration}.json"
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)

    return run_id, accuracy, f1, roc_auc


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, choices=[1, 2], required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    args = parser.parse_args()

    train(args.iteration, args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
