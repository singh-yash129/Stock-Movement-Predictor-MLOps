# Video Script — Stock Movement Predictor (MLOps OPPE-1)
Total time: approximately 7-8 minutes

## A — Problem Statement (30 seconds)
"Hi, I'm [Your Name], and this is my OPPE-1 submission for the Stock
Movement Predictor assignment.

The task is to predict, at every single minute, whether a stock will
trade up or down 5 minutes from now — using only the last 10 minutes of
its price and volume history. On top of the actual prediction model, the
assignment requires a full MLOps setup around it: DVC for data
versioning, Feast for feature storage and serving, MLflow for
experiment tracking and a model registry, and a CI pipeline with CML
that automatically evaluates and reports on every push to main.

The reason all of this tooling matters — a raw model that predicts well
once, on one snapshot of data, isn't enough for a real trading system.
You need to know exactly which data produced which model, you need
training and serving to compute features identically, and you need every
change to main to be automatically validated before anyone trusts it."

## B — Approach (30 seconds)
"My approach follows the six deliverables in order.

First, I version both provided data snapshots — v0 and v1 — with DVC,
backed by a GCS remote. Then I compute two rolling features,
`rolling_avg_10` and `volume_sum_10`, plus the 5-minute-ahead target
label, and register those as a Feast feature view keyed on `stock_name`.
I train in two iterations — first on v0 only, then on the merged v0+v1
data — pulling features from Feast both times to guarantee point-in-time
correctness. I run a hyperparameter sweep tracked in MLflow and register
the best model to the Model Registry. Finally, a GitHub Actions CI
workflow fetches that registered model, pulls the DVC-tracked test set,
runs sanity tests and evaluation, and posts the results as a CML comment
on the pull request."

## C — Cloud Compute Setup (45 seconds)
*Show GCP Console / Cloud Shell on screen*

"For compute, I'm running this on a GCP project, project ID
[your-project-id]. Here's my Cloud Shell — I've enabled the Compute
Engine, Cloud Storage, and BigQuery APIs, and created a GCS bucket,
[bucket-name], which serves as the remote storage backend for DVC and
also holds MLflow's artifact store.

I provisioned an e2-standard-4 VM to actually run the pipeline, since the
raw data is 120-150 MB and I wanted enough memory headroom. I'm SSH'd
into that VM right here.

And per the assignment's mandatory step, I've already granted the course
team viewer and container-viewer IAM roles on this project — you can see
that reflected in the IAM bindings here."

## D — Input Files (45 seconds)
*Show the CSV / df.head() on screen*

"Let me walk through the input data.

This is NSE minute-level data from the `StockAnalyticaData` folder in the
`IITMBSMLOps/MLOPS_MAY_2026_OPPE1` repository — one CSV per stock, named
like `AARTIIND__EQ__NSE__NSE__MINUTE.csv`, with columns timestamp, open,
high, low, close, and volume. The stock ticker itself isn't a column in
the raw file — I derive it from the filename, exactly as the reference
notebook does.

For Iteration 1, `v0` gives me two stocks — AARTIIND and ABCAPITAL. For
Iteration 2, `v1` adds three more — ABFRL, ADANIENT, and ADANIGAS — on
top of v0.

Since the raw files are large, I didn't commit them directly to GitHub.
Instead they live in a GCS bucket and are tracked through DVC pointer
files, and for fast local iteration while building the pipeline, I
worked off a contiguous recent-rows subsample per stock, since random row
sampling would break the 10-minute rolling-window calculations."

## E — Sequence of Actions Performed (1 minute)
*Show terminal on screen, step by step*

"Here's the exact sequence, in order.

First, I ran the GCP setup script — created the bucket, enabled APIs,
granted IAM access, and stood up the VM.

Second, I initialized DVC, added both `v0` and `v1` raw CSVs, pushed
them to the GCS remote, and committed the `.dvc` pointer files to git.

Third, I ran feature engineering on `v0` — computing the rolling
features and the target label — then applied and materialized my Feast
feature view against that.

Fourth, I ran training for Iteration 1, pulling historical features
straight from Feast rather than the raw CSV.

Fifth, I repeated feature engineering on `v1`, merged it with `v0`,
re-pointed Feast at the merged parquet file, re-applied and
re-materialized, then ran Iteration 2 training on the combined data.

Sixth, I ran the hyperparameter sweep, which logs every run to MLflow
and registers the best one to the Model Registry.

Seventh, I carved out a fixed held-out test set, tracked it with DVC too,
and pushed it to the remote — this is what CI evaluates against on every
run.

And finally, I pushed the whole repository to a private GitHub repo,
added the course team as a collaborator, and opened a pull request to
trigger the CI workflow."

## F — Exhaustive Explanation of Scripts/Code (1 minute)
*Show feature_store.yaml, then stock_repo.py, then train.py on screen*

"Let me walk through the code.

`feature_store.yaml` is Feast's config — local provider, a SQLite online
store for fast lookups, and a registry file tracking my definitions.

`stock_repo.py` defines three things: the entity `stock_name`, which
Feast uses to key every feature; the `FileSource`, pointing at my
feature-engineered parquet file with `event_timestamp` explicitly set so
Feast doesn't have to guess; and the feature view itself, mapping
`rolling_avg_10`, `volume_sum_10`, close, and volume to that entity.

`prepare_features.py` is where the actual feature math happens — for
each stock's CSV, I derive the ticker from the filename, forward-fill any
missing values, then take a real 10-minute time-window rolling mean of
close price and rolling sum of volume — a time-based window rather than
a row-count window, since minute bars can have gaps. `min_periods=1`
means if fewer than 10 minutes of history exist yet, it still processes
whatever's available, per the assignment's instruction. The target label
comes from shifting the close price 5 rows ahead and comparing, exactly
as in the reference notebook.

`train.py` never touches the raw CSV for feature values — it builds an
entity dataframe with just the stock name, timestamp, and label, and
calls `get_historical_features` on the Feast store, which does a
point-in-time correct join. Everything after that is a standard
RandomForest trained and logged to MLflow.

`hyperparameter_tuning.py` sweeps over `n_estimators` and `max_depth`,
logs every combination as its own MLflow run, and registers whichever
run had the best F1 score to the Model Registry.

And `ci_evaluate.py` is what actually runs inside GitHub Actions — it
loads the latest registered model from MLflow, loads the DVC-pulled test
set, runs four sanity tests — one per feature, checking things like
non-negative rolling values and non-null stock names — then computes
evaluation metrics and writes them out as JSON and a plot, which the CI
workflow hands to CML to post as a PR comment."

## G — Errors I Encountered (45 seconds)
"A few things went wrong along the way, worth mentioning honestly.

[Fill this in with what actually happened when you ran it — for
example: Feast couldn't infer the timestamp column and needed it set
explicitly; a `dvc push` failed until the service account had Storage
Object Admin rather than just Viewer; the CI workflow initially failed
because the GitHub Actions runner didn't have GCP credentials configured
for `dvc pull`, fixed by adding the service-account key as a repo
secret and writing it to a file in the workflow step; or `mlflow.register_model`
failed until I set `MLFLOW_TRACKING_URI` correctly in the training
shell.]

Document these here honestly and specifically once you've actually run
the pipeline — this section is evaluated on genuine troubleshooting, not
a made-up story."

## H — Working Demonstration of the Pipeline in GCP (1 minute)
*Show terminal, running live*

"Now let me run through the pipeline live.

Here's `feast apply` — created entity `stock_name`, created feature view
`stock_features`. Registration done.

Here's `feast materialize-incremental` — pushing feature values into the
SQLite online store, no errors.

Now `train.py --iteration 1` — here's the historical features pulled
from Feast, and here's the accuracy and F1 printed at the end.

Now the merge step and `train.py --iteration 2` — same code, more data,
and you can see the metrics shift.

Here's the hyperparameter sweep running through each combination, and at
the end, the model getting registered — here it is in the MLflow Model
Registry UI, version 1, stage None, ready for CI to pick up.

And here's the GitHub Actions run — you can see it pull the model from
the registry, pull the test set via `dvc pull`, run the sanity tests, and
at the bottom, the CML comment posted directly on the pull request, with
the metrics table and the plot."

## I — Explaining Output Files/Data (30 seconds)
*Show file explorer / GitHub repo / GCS or MLflow console*

"Here's what actually got produced.

`registry.db` and `online_store.db` are Feast's definition registry and
materialized feature values. The `features_v0.parquet`,
`features_v1.parquet`, and `features_merged.parquet` files are my
feature-engineered datasets at each stage. `model_iteration_1.joblib`
and `model_iteration_2.joblib` are the two trained models, saved
locally, with full copies also logged as MLflow artifacts.

Every tuning run is visible here in the MLflow Tracking UI, and the best
one is registered here in the Model Registry. And `metrics.json` and
`metrics_plot.png` are what CI generates and CML turns into that PR
comment I showed a moment ago."

## J — Learnings from This Assignment (30 seconds)
"A few takeaways from building this end to end.

Keeping training and serving both pull features from Feast, rather than
computing them separately in two places, is really the whole point of a
feature store — it removes an entire class of subtle bugs where training
and inference quietly disagree.

Point-in-time correctness matters a lot more with financial time-series
data than it might first seem — it's very easy to accidentally leak
future prices into a feature if you're not careful with timestamps, and
Feast's historical retrieval is built specifically to prevent that.

Splitting the pipeline into versioned data, a feature store, tracked
experiments, and an automated CI check means each piece can be trusted
independently — I don't have to re-verify the whole pipeline by hand
every time I retrain, because CI does that automatically on every push.

And practically — treating the raw 120-150 MB files as DVC-tracked data
rather than trying to commit them to git directly, and working off a
smaller per-stock subsample during development, made the whole build
loop dramatically faster without changing anything about the final
pipeline logic.

Thank you!"

Total time: approximately 7-8 minutes ✅

---

## Quick recording checklist
- [ ] GCP Cloud Shell / VM authenticated before hitting record
- [ ] Both v0 and v1 CSVs already pulled locally (or via `dvc pull`)
- [ ] Terminal font bumped up so it's readable
- [ ] Run through Deliverables 2-6 once already, so there are no live surprises
- [ ] GCP project ID and bucket name ready to reference on screen
- [ ] MLflow UI and GitHub PR/CML comment open in browser tabs, ready to switch to
- [ ] Screen + audio recording tested beforehand
- [ ] Final file named correctly: `<IITM_BS_ID>_OPPE1_<TERM>_<YEAR>_MLOps.<file_type>`
- [ ] Collaborator invitation to course team confirmed ACCEPTED before ending session
