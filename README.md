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
reports/     figures/
logs/        pipeline logs
configs/     (reserved for shared config)
tests/       pytest suite
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
