"""
hyperparameter_tune.py
======================
Grid-search hyperparameter sweep with MLflow tracking.
Selects the best run based on ROC-AUC and promotes it to
the "Staging" stage in the MLflow Model Registry.

Usage:
    python src/hyperparameter_tune.py --iteration 2

Environment variables:
    MLFLOW_TRACKING_URI     — e.g. sqlite:///mlflow.db
    MLFLOW_ARTIFACT_ROOT    — e.g. gs://23f2004644-mlops-oppe/mlflow-artifacts
"""

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

# Allow importing from sibling scripts in src/
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from train import (
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET_COL,
    FEATURE_COLS,
    ENTITY_COL,
    load_data,
    prepare_features,
)


# ─── Hyperparameter Grid ───────────────────────────────────────────────────────

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [5, 10, 20],
    "min_samples_split": [2, 5],
}


# ─── Single Run ────────────────────────────────────────────────────────────────

def run_single(iteration, n_estimators, max_depth, min_samples_split,
               X_train, y_train, X_test, y_test, le):
    """Execute one hyperparameter configuration and log to MLflow."""
    run_name = (
        f"iter{iteration}_rf_n{n_estimators}_d{max_depth}_mss{min_samples_split}"
    )

    with mlflow.start_run(run_name=run_name, nested=False) as run:
        mlflow.log_param("iteration",         iteration)
        mlflow.log_param("n_estimators",      n_estimators)
        mlflow.log_param("max_depth",         max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("train_rows",        len(X_train))
        mlflow.log_param("test_rows",         len(X_test))
        mlflow.log_param("stocks",            list(le.classes_))

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, zero_division=0)
        roc_auc  = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  roc_auc)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        run_id = run.info.run_id
        print(
            f"  [{run_id[:8]}] n_est={n_estimators:3d}  depth={max_depth:2d}"
            f"  mss={min_samples_split}  → acc={accuracy:.4f}  f1={f1:.4f}"
            f"  auc={roc_auc:.4f}"
        )
        return run_id, accuracy, f1, roc_auc


# ─── Sweep ─────────────────────────────────────────────────────────────────────

def sweep(iteration: int):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    artifact_root = os.getenv(
        "MLFLOW_ARTIFACT_ROOT",
        "gs://23f2004644-mlops-oppe/mlflow-artifacts"
    )
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifact_root)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load data once
    train_df, test_df = load_data(iteration)
    X_train, y_train, le = prepare_features(train_df, fit_le=True)
    X_test,  y_test,  _  = prepare_features(test_df,  le=le)

    print(
        f"\n[tune] Iteration {iteration} — Sweep over"
        f" {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_split'])}"
        f" configurations ...\n"
    )

    results = []
    for n_est, depth, mss in product(
        PARAM_GRID["n_estimators"],
        PARAM_GRID["max_depth"],
        PARAM_GRID["min_samples_split"],
    ):
        run_id, acc, f1, auc = run_single(
            iteration, n_est, depth, mss,
            X_train, y_train, X_test, y_test, le
        )
        results.append({
            "run_id": run_id, "n_estimators": n_est,
            "max_depth": depth, "min_samples_split": mss,
            "accuracy": acc, "f1_score": f1, "roc_auc": auc,
        })

    # Select best by ROC-AUC
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n[tune] Best run: {best['run_id'][:8]}  ROC-AUC={best['roc_auc']:.4f}")

    # Register best model in Model Registry
    best_model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(
        model_uri=best_model_uri,
        name="StockMovementPredictor",
    )
    print(f"[tune] Registered model version: {mv.version}")

    # Transition to Staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="StockMovementPredictor",
        version=mv.version,
        stage="Staging",
    )
    print(f"[tune] Model v{mv.version} → Staging")

    # Also tag as Production (latest iteration wins)
    client.transition_model_version_stage(
        name="StockMovementPredictor",
        version=mv.version,
        stage="Production",
    )
    print(f"[tune] Model v{mv.version} → Production")

    # Save best model info for CI
    best_info = {
        "run_id":          best["run_id"],
        "model_version":   mv.version,
        "model_name":      "StockMovementPredictor",
        "iteration":       iteration,
        "accuracy":        best["accuracy"],
        "f1_score":        best["f1_score"],
        "roc_auc":         best["roc_auc"],
        "n_estimators":    best["n_estimators"],
        "max_depth":       best["max_depth"],
        "min_samples_split": best["min_samples_split"],
        "label_classes":   list(le.classes_),
    }
    info_path = MODELS_DIR / "best_model_info.json"
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"[tune] Best model info saved → {info_path}")

    # Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    print(f"[tune] Tuning results saved → {REPORTS_DIR / 'tuning_results.csv'}")

    return best_info


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    sweep(args.iteration)


if __name__ == "__main__":
    main()
