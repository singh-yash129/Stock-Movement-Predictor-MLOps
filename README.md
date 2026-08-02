# 23F2004644 — Stock Movement Predictor (MLOps OPPE-1, MAY 2026)

End-to-end MLOps pipeline predicting stock price direction (up/down) 5 minutes in the future.

## Pipeline Overview

```
Raw CSVs (v0/v1)
     │
     ▼
[DVC] data versioning → GCS remote (gs://23f2004644-mlops-oppe/dvc-store)
     │
     ▼
[data_prep.py] → rolling_avg_10, volume_sum_10, target
     │
     ▼
[Feast] feature store → apply + materialize → point-in-time retrieval
     │
     ▼
[train.py] Iteration 1 (v0) → MLflow run
[train.py] Iteration 2 (v0+v1) → MLflow run
     │
     ▼
[hyperparameter_tune.py] → grid sweep → best model → MLflow Model Registry (Production)
     │
     ▼
[CI / CML] sanity_tests + evaluate → CML PR comment with metrics + confusion matrix
```

## Deliverables

| # | Deliverable | Tools |
|---|-------------|-------|
| D2 | Data versioning (v0 & v1) | DVC + GCS |
| D3 | Feature store (rolling features) | Feast |
| D4 | Training — 2 iterations | scikit-learn + MLflow |
| D5 | Hyperparameter tuning + registry | MLflow |
| D6 | CI pipeline + CML report | GitHub Actions + CML |

## GCS Bucket

`gs://23f2004644-mlops-oppe`

| Path | Contents |
|------|----------|
| `/dvc-store/` | DVC cache (raw CSV files) |
| `/mlflow-artifacts/` | MLflow model artifacts |
| `/mlflow/mlflow.db` | MLflow SQLite tracking DB |

## Features

| Feature | Description |
|---------|-------------|
| `rolling_avg_10` | 10-min rolling mean of close price |
| `volume_sum_10` | 10-min rolling sum of volume |
| `stock_name` | NSE stock ticker |

**Target**: `1` if `close[t+5] > close[t]`, else `0`

## Quick Start (GCP Cloud Shell)

```bash
# Clone and set up
git clone https://github.com/singh-yash129/23F2004644_MLOPS_OPPE1_MAY_2026.git
cd 23F2004644_MLOPS_OPPE1_MAY_2026

# Clone data source repo alongside
git clone https://github.com/IITMBSMLOps/MLOPS_MAY_2026_OPPE1.git ../MLOPS_MAY_2026_OPPE1

# Run the full pipeline
chmod +x setup_gcp.sh
./setup_gcp.sh
```

## Manual Step-by-Step

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy raw data
```bash
mkdir -p data/raw/v0 data/raw/v1
cp ../MLOPS_MAY_2026_OPPE1/StockAnalyticaData/v0/*.csv data/raw/v0/
cp ../MLOPS_MAY_2026_OPPE1/StockAnalyticaData/v1/*.csv data/raw/v1/
```

### 3. DVC — Track v0 data
```bash
dvc init
dvc remote add -d gcs_remote gs://23f2004644-mlops-oppe/dvc-store
dvc add data/raw/v0
git add data/raw/v0.dvc .dvc/config
git commit -m "feat: track v0 data with DVC"
git tag v0-data
dvc push
```

### 4. DVC — Track v1 data
```bash
dvc add data/raw/v1
git add data/raw/v1.dvc
git commit -m "feat: track v1 data with DVC"
git tag v1-data
dvc push
```

### 5. Data preparation
```bash
python src/data_prep.py --iteration 1   # v0 only
python src/data_prep.py --iteration 2   # v0 + v1 merged
```

### 6. Feast feature store
```bash
# Build parquet + apply + materialize
python src/feast_materialize.py --iteration 1
python src/feast_materialize.py --iteration 2
```

### 7. Training — Iteration 1 & 2
```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_ROOT="gs://23f2004644-mlops-oppe/mlflow-artifacts"

python src/train.py --iteration 1
python src/train.py --iteration 2
```

### 8. Hyperparameter tuning
```bash
python src/hyperparameter_tune.py --iteration 2
```

### 9. Track test data with DVC
```bash
dvc add data/processed/test_latest.csv
git add data/processed/test_latest.csv.dvc
git commit -m "feat: track test data with DVC"
dvc push
```

### 10. Upload MLflow DB to GCS
```bash
gsutil cp mlflow.db gs://23f2004644-mlops-oppe/mlflow/mlflow.db
```

### 11. Push to GitHub
```bash
git push origin main --tags
```

### 12. Add GitHub Secret for CI
In GitHub repository → **Settings → Secrets and variables → Actions → New repository secret**:
- Name: `GCP_SA_KEY`
- Value: Contents of your GCP service account JSON key

### 13. Trigger CI
Create a PR or push a commit to `main` to trigger the workflow.

## Restoring a DVC Snapshot

```bash
# Restore v0 data only
git checkout v0-data -- data/raw/v0.dvc
dvc pull data/raw/v0

# Restore v1 data
git checkout v1-data -- data/raw/v1.dvc
dvc pull data/raw/v1
```

## MLflow UI (local)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open: http://localhost:5000
```

## Project Structure

```
.
├── data/
│   ├── raw/v0/              # DVC-tracked: AARTIIND, ABCAPITAL
│   ├── raw/v1/              # DVC-tracked: ABFRL, ADANIENT, ADANIGAS
│   └── processed/           # train/test CSVs (DVC-tracked)
├── feature_store/
│   ├── feature_store.yaml   # Feast config
│   ├── features.py          # Entity + FeatureView definitions
│   └── data/                # Parquet offline source
├── src/
│   ├── data_prep.py         # Feature engineering + target computation
│   ├── feast_materialize.py # Feast apply + materialize + retrieval
│   ├── train.py             # Model training + MLflow logging
│   ├── hyperparameter_tune.py  # Sweep + model registry
│   ├── evaluate.py          # CI evaluation + CML report
│   └── sanity_tests.py      # Feature sanity checks
├── models/                  # best_model_info.json
├── reports/                 # metrics.json, plots
├── .github/workflows/ci.yml # CI + CML workflow
├── requirements.txt
└── setup_gcp.sh             # Full pipeline script
```
