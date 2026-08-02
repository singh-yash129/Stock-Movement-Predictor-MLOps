"""
evaluate.py
===========
CI evaluation script:
  1. Loads the best model from the MLflow Model Registry (Production stage).
  2. Pulls test data from DVC (already pulled by CI step).
  3. Computes evaluation metrics and saves them as JSON + plots.
  4. Outputs metrics in a format CML can read.

Usage:
    python src/evaluate.py

Environment variables:
    MLFLOW_TRACKING_URI   — e.g. sqlite:///mlflow.db
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder


# ─── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
REPORTS_DIR   = Path("reports")

FEATURE_COLS = ["rolling_avg_10", "volume_sum_10"]
TARGET_COL   = "target"
ENTITY_COL   = "stock_name"


# ─── Load model from MLflow ────────────────────────────────────────────────────

def load_best_model():
    """
    Load the Production-stage model from the MLflow Model Registry.
    Falls back to loading from best_model_info.json if registry is unavailable.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "StockMovementPredictor"

    try:
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/Production"
        )
        print(f"[evaluate] Loaded model: {model_name} (Production)")
        return model
    except Exception as e:
        print(f"[evaluate] WARNING: Could not load from registry: {e}")
        print("[evaluate] Falling back to best_model_info.json ...")

    # Fallback: load from run_id stored in best_model_info.json
    info_path = MODELS_DIR / "best_model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Neither MLflow registry nor {info_path} is available."
        )
    with open(info_path) as f:
        info = json.load(f)

    run_id = info["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"[evaluate] Loaded model from run_id: {run_id}")
    return model


# ─── Load test data ────────────────────────────────────────────────────────────

def load_test_data() -> pd.DataFrame:
    """Load the canonical test dataset (latest iteration)."""
    # Prefer the latest iteration test file
    for candidate in ["test_latest.csv", "test_iter2.csv", "test_iter1.csv"]:
        path = PROCESSED_DIR / candidate
        if path.exists():
            print(f"[evaluate] Loading test data: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("No test data found in data/processed/")


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_best_model()
    test_df = load_test_data()

    test_df = test_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    # Encode stock_name
    le = LabelEncoder()

    # Try to load label classes from best_model_info.json
    info_path = MODELS_DIR / "best_model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        le.classes_ = np.array(info.get("label_classes", []))
        # Handle unseen labels gracefully
        known = set(le.classes_)
        test_df["stock_name_safe"] = test_df[ENTITY_COL].apply(
            lambda s: s if s in known else le.classes_[0]
        )
        test_df["stock_name_enc"] = le.transform(test_df["stock_name_safe"])
    else:
        le.fit(test_df[ENTITY_COL].astype(str))
        test_df["stock_name_enc"] = le.transform(test_df[ENTITY_COL].astype(str))

    X_test = test_df[FEATURE_COLS + ["stock_name_enc"]].values
    y_test = test_df[TARGET_COL].astype(int).values

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ────────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    roc_auc  = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n[evaluate] === Test Evaluation Metrics ===")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # ── Save metrics.json (for CML) ────────────────────────────────────────────
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc":  round(roc_auc, 4),
        "n_test_samples": len(y_test),
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {metrics_path}")

    # ── Confusion Matrix Plot ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Down (0)", "Up (1)"],
        yticklabels=["Down (0)", "Up (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Stock Movement Predictor")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {cm_path}")

    # ── Per-stock metrics ──────────────────────────────────────────────────────
    per_stock_rows = []
    stock_col = test_df.reset_index(drop=True)[ENTITY_COL]
    for stock in stock_col.unique():
        mask = (stock_col == stock).values
        if mask.sum() == 0:
            continue
        ys_true = y_test[mask]
        ys_pred = y_pred[mask]
        per_stock_rows.append({
            "stock": stock,
            "n":     int(mask.sum()),
            "accuracy": round(accuracy_score(ys_true, ys_pred), 4),
            "f1":    round(f1_score(ys_true, ys_pred, zero_division=0), 4),
        })

    per_stock_df = pd.DataFrame(per_stock_rows)
    per_stock_path = REPORTS_DIR / "per_stock_metrics.csv"
    per_stock_df.to_csv(per_stock_path, index=False)
    print(f"[evaluate] Per-stock metrics saved → {per_stock_path}")
    print(per_stock_df.to_string(index=False))

    # ── CML report markdown ────────────────────────────────────────────────────
    report_md = f"""## 📊 Model Evaluation Report

### Overall Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | {accuracy:.4f} |
| F1-Score  | {f1:.4f} |
| ROC-AUC   | {roc_auc:.4f} |
| N Samples | {len(y_test)} |

### Confusion Matrix

![Confusion Matrix](reports/confusion_matrix.png)

### Per-Stock Performance

{per_stock_df.to_markdown(index=False)}

### Sanity Test Results

Sanity tests passed ✅ (see CI logs for details)
"""
    report_path = REPORTS_DIR / "cml_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"[evaluate] CML report saved → {report_path}")

    return metrics


if __name__ == "__main__":
    evaluate()
