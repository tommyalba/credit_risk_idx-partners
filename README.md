# Credit Risk Modeling — Lending Club

Binary credit-risk pipeline: predict whether a loan ends **GOOD** (Fully Paid)
or **BAD** (Charged Off / Default / Late 31–120) using only application-time
features, with a fixed performance window, WoE binning, temporal
out-of-time evaluation, probability calibration and cost-based thresholds.

## Layout

```
notebooks/   1_EDA (exploratory only) · 2_Preparation_Preprocessing (canonical
             prep: performance window, target, temporal split, WoE binning)
             · 3_Modeling_LogisticRegression · 4_Modeling_Tree-based
src/         workflow modules: utils, performance_window, binning_woe,
             metrics, evaluation, thresholding, calibration, training,
             ensemble, expected_cost_threshold, calibration_selection,
             ks_statistic_evaluation, population_stability_index,
             scorecard, shap_interpretability, manifest
data/        raw/ → interim/ → processed/   (DVC-tracked)
models/      serialized winners               (DVC-tracked)
artifacts/   binning_artifacts.pkl, manifest.json   (DVC-tracked)
reports/     figures/, reject_inference.md + results
logs/        pipeline logs
configs/     (reserved for shared config)
tests/       pytest suite
experiments/ one-off studies: survival hazard, adverse impact,
             bureau-feature retest, reject inference
autoresearch/ agent experiment loop: program.md protocol,
             experiment.py (temporal-validation eval only),
             configs/, results/trials.jsonl
pipeline/    Scenario A+B .py pipelines (nb2->nb4 equivalents):
             core.py shared engine; scenario_a/ and scenario_b/ each
             with prepare.py / train.py / score.py; infer.py runs the
             unified application->funding inference workflow
```

## Environment

Canonical env is conda `credit-risk_VENV` (Python 3.13):

```bash
conda create -n credit-risk_VENV python=3.13
conda activate credit-risk_VENV
pip install -r requirements.txt        # pinned
pip install -r requirements-dev.txt    # pytest
```

`environment.yml` is provided for a full conda reproduction.

## Workflow

```bash
jupyter notebook            # run notebooks in order: 1 → 2 → 3 → 4
pytest                      # unit tests (no data required)
dvc pull                    # fetch data/models/artifacts from remote
```

Notebook 2 produces `data/interim` + `data/processed` pickles and
`artifacts/manifest.json` (env versions, split config, artifact SHA-256).
Notebooks 3–4 consume them and log every run to MLflow
(`sqlite:///mlflow.db`, experiment names `logistic-regression-scorecard` and
`tree-based-models`).

## Versioning

- **DVC** tracks `data/`, `models/`, `artifacts/` (`*.dvc` files in git).
  Remote: local dir `~/dvc_remotes/credit_risk` — change with
  `dvc remote modify local url <path>` (swap for S3/GDrive for a team).
- **MLflow** tracks experiments/metrics/params; `mlflow.db` + `mlartifacts/`
  are gitignored local state.
- **Git** tracks code, notebooks, tests, requirements, reports.

## Agent / contributor orientation

If you are an agent starting a fresh session in this repo:

1. **Memory**: run `~/.optmem/memo wake` first (required by global rules) —
   it holds cross-session decisions and experiment outcomes.
2. **Environment**: `conda activate credit-risk_VENV` before running
   notebooks, `src/` modules, or `dashboard/app.py` (pickle artifacts are
   env-sensitive — never load them in another env).
3. **Read order**: this README → `notebooks/2_Preparation_Preprocessing.ipynb`
   (canonical prep) → `src/` modules as needed → `artifacts/manifest.json`
   (exact artifact versions/hashes).
4. **Notebook roles**: nb1 is exploratory-only (its in-notebook drops are NOT
   the modeling pipeline); nb2 is the single canonical prep that feeds
   nb3/nb4; nb3 = LR scorecard, nb4 = tree models + ensemble.
5. **Hard rules**: never touch the OOT test split during model selection;
   post-origination columns (`pymnt_time_month`, `recoveries`, `out_prncp`,
   `total_pymnt`, …) are leakage and must never become features; keep
   `pytest` green (`15 passed`).
6. **Current state**: CatBoost won Stage A on the temporal validation split
   (valid AUC ≈ 0.647, OOT ≈ 0.644); LR scorecard valid ≈ 0.639 / OOT ≈ 0.639.
   Streamlit scoring app in `dashboard/app.py` uses the nb3 scorecard +
   `binning_artifacts.pkl`. `experiments/` holds one-off studies
   (survival hazard, adverse impact, bureau-feature retest — all documented
   as rejected or v2 candidates).
7. **Deferred work**: cloud DVC remote (needs credentials),
   dashboard v2 (batch scoring, PSI panel, segment views), hazard-model
   recalibration, rolling-origin evaluation.
   **Done via OpenResearch agent** (branch `reject-inference`): reject
   inference — strict RI impossible (funded-only data, no reject labels);
   sensitivity/bounds analysis in `reports/reject_inference.md`. Key result:
   applicant-pool PD plausibly 1.4–1.9× the funded estimate and τ=0.288 is
   likely too loose (oracle ≈ 0.12–0.18) — the bias is invisible to model
   scores and lives entirely in realized outcomes. Merge the branch when
   ready.
   **Also done** (branch `class-imbalance-sampling`): oversampling AND
   undersampling study — `reports/class_imbalance_sampling.md`. Verdict:
   SMOTE hurts monotonically (−0.006 @30% minority, −0.020 @50%), RUS is
   AUC-neutral but destroys calibration; nothing beats cat_raw 0.6468.
   `autoresearch/experiment.py` now supports a `sampling` config key
   (fit split only; SMOTENC required — 4 of 12 features are categorical).

## Key methodology decisions

- Performance window M = 33 months (90th percentile of months-to-bad);
  eligibility is asymmetric — bads and resolved goods always count, only
  'Current' loans must survive M months.
- Safe horizon cutoff = snapshot − M months → OOT test = 2013Q1.
- Validation is **temporal** (latest ~11% of historical issues), OOT is the
  untouched final check; thresholds are tuned on validation only.
- Tree models use the best *consistent* hyperparameter run (no re-search);
  a soft-voting `ensemble_avg` of XGB+CatBoost+LGBM competes in Stage A.
- Cost threshold uses per-observation EAD (`loan_amnt`) × LGD and
  term-derived duration, compared against scalar-EAD scenarios.
