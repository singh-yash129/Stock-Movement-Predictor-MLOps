#!/usr/bin/env bash
# =============================================================================
# setup_gcp.sh
# =============================================================================
# Complete step-by-step pipeline to run in GCP Cloud Shell after cloning.
#
# Usage:
#   git clone https://github.com/<YOUR_USERNAME>/23F2004644_MLOPS_OPPE1_MAY_2026.git
#   cd 23F2004644_MLOPS_OPPE1_MAY_2026
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh
#
# Or run step-by-step (recommended for first run).
# =============================================================================

set -euo pipefail

# ─── Variables ────────────────────────────────────────────────────────────────
REPO_ROOT="$(pwd)"
GCS_BUCKET="23f2004644-mlops-oppe"
GCS_DVC_REMOTE="gs://${GCS_BUCKET}/dvc-store"
GCS_MLFLOW_ARTIFACTS="gs://${GCS_BUCKET}/mlflow-artifacts"
GCS_MLFLOW_DB="gs://${GCS_BUCKET}/mlflow/mlflow.db"
GCS_FEAST_REGISTRY="gs://${GCS_BUCKET}/feast/registry.db"

# ─── Helper ───────────────────────────────────────────────────────────────────
step() { echo; echo "══════════════════════════════════════════════"; echo "  STEP: $1"; echo "══════════════════════════════════════════════"; }

# =============================================================================
# STEP 0: Install Python dependencies
# =============================================================================
step "0 — Install dependencies"
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# =============================================================================
# STEP 1: Copy raw data into the repo data directory
# =============================================================================
step "1 — Copy raw data (v0 and v1)"

mkdir -p data/raw/v0 data/raw/v1

# Adjust path if you have the source repo in a sibling directory
SRC_DATA="${REPO_ROOT}/../MLOPS_MAY_2026_OPPE1/StockAnalyticaData"
if [ -d "$SRC_DATA" ]; then
  cp "${SRC_DATA}/v0/"*.csv data/raw/v0/
  cp "${SRC_DATA}/v1/"*.csv data/raw/v1/
  echo "✓ Data copied from ${SRC_DATA}"
else
  echo "⚠  Source data not found at ${SRC_DATA}"
  echo "   Please manually copy the CSVs:"
  echo "     v0 → data/raw/v0/   (AARTIIND, ABCAPITAL)"
  echo "     v1 → data/raw/v1/   (ABFRL, ADANIENT, ADANIGAS)"
  echo "   Then re-run from STEP 2."
  exit 1
fi

ls -lh data/raw/v0/ data/raw/v1/

# =============================================================================
# STEP 2: DVC — Initialise and track v0 data
# =============================================================================
step "2 — DVC init + track v0 data"

dvc init --no-scm 2>/dev/null || dvc init  # idempotent

# Configure GCS remote
dvc remote add -d gcs_remote "${GCS_DVC_REMOTE}" 2>/dev/null || \
  dvc remote modify gcs_remote url "${GCS_DVC_REMOTE}"

echo "[dvc] Remote configured: ${GCS_DVC_REMOTE}"

# Track v0 data
dvc add data/raw/v0
git add data/raw/v0.dvc data/raw/.gitignore .dvc/config
git commit -m "feat: track v0 stock data with DVC (AARTIIND, ABCAPITAL)"
git tag -a "v0-data" -m "DVC snapshot: v0 data only"

# Push v0 to GCS
dvc push
echo "✓ v0 data pushed to ${GCS_DVC_REMOTE}"

# =============================================================================
# STEP 3: DVC — Track v1 data (v0 + v1 together)
# =============================================================================
step "3 — DVC track v1 data"

dvc add data/raw/v1
git add data/raw/v1.dvc data/raw/.gitignore
git commit -m "feat: track v1 stock data with DVC (ABFRL, ADANIENT, ADANIGAS)"
git tag -a "v1-data" -m "DVC snapshot: v1 data added (v0+v1 available)"

# Push v1 to GCS
dvc push
echo "✓ v1 data pushed to ${GCS_DVC_REMOTE}"

# Demonstrate checkout (restore v0 snapshot)
echo
echo "─── Demo: checkout v0 snapshot ───"
git checkout v0-data -- data/raw/v0.dvc
dvc pull data/raw/v0
echo "✓ v0 snapshot restored from GCS"
git checkout HEAD -- data/raw/v0.dvc   # restore to latest

# =============================================================================
# STEP 4: Data preparation — Iteration 1 (v0 only)
# =============================================================================
step "4 — Data prep: Iteration 1 (v0 only)"

python src/data_prep.py --iteration 1
echo "✓ Iteration 1 data prepared"
ls -lh data/processed/

# =============================================================================
# STEP 5: Feast — Materialise features (Iteration 1)
# =============================================================================
step "5 — Feast feature store: Iteration 1"

cd feature_store
feast apply
cd "${REPO_ROOT}"

python src/feast_materialize.py --iteration 1
echo "✓ Feast features materialised (Iteration 1)"

# =============================================================================
# STEP 6: MLflow — Train Iteration 1
# =============================================================================
step "6 — Train: Iteration 1 (v0 only)"

export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_ARTIFACT_ROOT="${GCS_MLFLOW_ARTIFACTS}"

python src/train.py --iteration 1 --n_estimators 100 --max_depth 10
echo "✓ Training Iteration 1 complete"

# =============================================================================
# STEP 7: Data preparation — Iteration 2 (v0 + v1)
# =============================================================================
step "7 — Data prep: Iteration 2 (v0 + v1 merged)"

python src/data_prep.py --iteration 2
echo "✓ Iteration 2 data prepared"
ls -lh data/processed/

# =============================================================================
# STEP 8: Feast — Materialise features (Iteration 2)
# =============================================================================
step "8 — Feast feature store: Iteration 2"

python src/feast_materialize.py --iteration 2
echo "✓ Feast features materialised (Iteration 2)"

# =============================================================================
# STEP 9: MLflow — Hyperparameter Tuning on Iteration 2 data
# =============================================================================
step "9 — Hyperparameter tuning: Iteration 2 (v0+v1)"

python src/hyperparameter_tune.py --iteration 2
echo "✓ Hyperparameter sweep complete — best model registered in MLflow"
cat models/best_model_info.json

# =============================================================================
# STEP 10: DVC — Track processed test data for CI
# =============================================================================
step "10 — DVC track processed test data"

dvc add data/processed/test_latest.csv
dvc add data/processed/test_iter1.csv
dvc add data/processed/test_iter2.csv

git add data/processed/*.dvc data/processed/.gitignore
git commit -m "feat: track processed test data with DVC"

dvc push
echo "✓ Test data pushed to GCS for CI use"

# =============================================================================
# STEP 11: Upload MLflow DB to GCS (so CI can access it)
# =============================================================================
step "11 — Upload MLflow DB to GCS"

gsutil cp mlflow.db "${GCS_MLFLOW_DB}"
echo "✓ MLflow DB uploaded to ${GCS_MLFLOW_DB}"

# =============================================================================
# STEP 12: Commit all generated files and push to GitHub
# =============================================================================
step "12 — Commit and push to GitHub"

# Add non-DVC tracked files
git add \
  models/best_model_info.json \
  reports/ \
  feature_store/data/ \
  feature_store/registry.db \
  .mlflow/ 2>/dev/null || true

git add -A
git commit -m "feat: complete MLOps pipeline — DVC, Feast, MLflow, models" \
  --allow-empty

git push origin main
git push origin --tags
echo "✓ All changes pushed to GitHub"

# =============================================================================
# STEP 13: Local sanity test + evaluation run
# =============================================================================
step "13 — Local sanity tests + evaluation"

python src/sanity_tests.py
python src/evaluate.py
echo "✓ Sanity tests and evaluation complete"
cat reports/metrics.json

# =============================================================================
# DONE
# =============================================================================
step "✅ Pipeline complete!"
echo
echo "Summary:"
echo "  DVC remote  : ${GCS_DVC_REMOTE}"
echo "  MLflow DB   : ${GCS_MLFLOW_DB}"
echo "  Artifacts   : ${GCS_MLFLOW_ARTIFACTS}"
echo "  Reports     : $(ls reports/)"
echo
echo "Next steps:"
echo "  1. Add GCP_SA_KEY secret to your GitHub repository settings"
echo "     (Settings → Secrets → Actions → New repository secret)"
echo "  2. Create a PR or push to main to trigger the CI workflow"
echo "  3. Check the Actions tab for the CML report"
