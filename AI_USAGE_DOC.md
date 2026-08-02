# OPPE AI Usage Documentation

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam
> Provide **public share links** to AI chats or attach conversation files if links are not available

- Claude (Anthropic)
    - Purpose: Generated pipeline code (feature engineering, Feast definitions, training/tuning scripts, CI workflow), setup scripts for GCP/DVC/GitHub, README.md, and this AI_Usage.md, based on the OPPE-1 problem statement I provided.
    - Shared Chat Link: <PASTE YOUR SHARE LINK HERE — click Share in the Claude UI on this conversation>
    - Notes: I reviewed, adapted column-name assumptions to the actual dataset schema, and ran all commands myself in my GCP environment.

---

## Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks
> Include **all responses** in case public share links are not available to share

### Tool Name #1: Claude (Anthropic)

- Prompt 1:
    - "Build a stock movement predictor with end-to-end MLOps tooling (DVC, Feast, MLflow, and CI with CML) on GCP" [full OPPE-1 problem statement pasted]
    - Response Log: Generated full project structure — `src/prepare_features.py` (rolling_avg_10, volume_sum_10, target computation), `feature_repo/stock_repo.py` and `feature_store.yaml` (Feast entity/feature view), `src/train.py` (iteration 1 & 2 training via Feast historical features), `src/hyperparameter_tuning.py` (MLflow sweep + model registry), `src/merge_versions.py`, `src/prepare_test_set.py`, `src/ci_evaluate.py` (sanity tests + evaluation), `.github/workflows/ci.yml` (CI + CML), setup scripts (`01_gcp_setup.sh` through `06_make_subsample.py`), README.md, and this AI_Usage.md template fill-in.

- Prompt 2:
    - "For pipeline we got the data is 120-150 MB CSV what we do"
    - Response Log: Recommended not committing large CSVs directly to GitHub; suggested Git LFS or GCS + DVC as remote storage instead, chunked reading for memory safety, and converting to Parquet for downstream steps. I chose the DVC + GCS remote approach, which is reflected in `scripts/02_dvc_setup.sh`.

- Prompt 3:
    - "For MLOps pipeline I listen that we must use sub data"
    - Response Log: Explained the rationale for using a data subsample during development (faster iteration, avoids OOM on small VMs) while keeping the full dataset as the source of truth in GCS/DVC for final runs. Recommended sampling by entity (stock) rather than by random row to preserve time-series continuity needed for rolling-window features. This became `scripts/06_make_subsample.py`.

*(Add further prompts here as you iterate/debug during the actual exam — e.g. any error you hit and the follow-up prompt you used to fix it.)*
