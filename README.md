# Credit Risk Modeling: Lending Club

A reproducible machine-learning system for retail-installment default risk,
built on the Lending Club public loan tape (2007–2014). The system produces
two complementary models: an application-time score that drives the
approve/reject decision, and a post-origination score that monitors the
funded book. Both share one canonical preprocessing pipeline, one temporal
validation protocol, and one production inference path.

## Key results at a glance

| Metric | Scenario A (application) | Scenario B (post-funding) |
|---|---|---|
| Validation ROC-AUC | 0.6468 | 0.6684 |
| Out-of-time ROC-AUC (2013Q1) | 0.6442 | 0.6687 |
| Validation KS | 0.215 | 0.253 |
| Validation LogLoss / Brier | 0.402 / 0.122 | 0.397 / 0.120 |
| Decision threshold τ (expected cost) | 0.288 | 0.282 |
| Approved-book bad rate (valid) | n/a | 12.9% vs 14.8% base |

Scenario B outperforms Scenario A by roughly two AUC points, as expected:
it additionally observes lender pricing (`grade`, `int_rate`,
`installment`, `funded_amnt`) that is unknown at application time.

## Problem formulation

The target is binary: a loan is BAD if it ends Charged Off, Default, or
Late (31–120 days); GOOD if Fully Paid. Loans still `Current` are handled
through an asymmetric maturity rule rather than dropped outright. The
modeling population is 117,841 funded loans drawn from 466,285 raw records
after deduplication, date parsing, and a 33-month performance window
(90th percentile of months-to-bad on the 2011–2012 mature cohort).

Two deployment scenarios define feature eligibility:

- **Scenario A (application-time).** Only borrower and bureau fields
  available before funding. Used for the credit decision.
- **Scenario B (post-origination).** Adds funding and pricing fields
  (`funded_amnt`, `grade`, `int_rate`, `installment`,
  `investor_funding_ratio`, `payment_to_income`,
  `mths_since_last_*`). Used for funded-book monitoring.

Repayment and collection outcomes (`recoveries`, `out_prncp`,
`total_pymnt`, `last_pymnt_*`, …) are leakage in both scenarios and are
rejected at scoring time.

## Methodology

- **Performance window.** M = 33 months. Bad and resolved-good loans are
  always eligible; `Current` loans must reach age M. The safe-horizon
  cutoff (snapshot 2016-01-01 minus M) removes unresolved young vintages.
- **Temporal validation.** Train = 2007-08 to 2012-11 (84,764 loans),
  validation = latest ~11% of history (10,596), OOT test = 2013Q1
  (22,481), evaluated once at the end. No random splits anywhere.
- **Leakage controls.** Post-origination columns are excluded by spec and
  hard-rejected by the scoring API. `pymnt_time_month`-class fields never
  enter features.
- **Modeling.** CatBoost, XGBoost, and LightGBM trained on an
  85/15 fit/early-stop split with a 50/50 early-stop/calibration split.
  Winner selected on validation AUC; probability calibration (raw vs
  sigmoid vs isotonic) chosen on validation Brier; decision threshold set
  by expected cost, not F1.
- **Economics.** C_FN = 8.25M IDR (EAD 15M × LGD 0.55), C_FP = 2.7M IDR
  (EAD × spread × duration). The cost-optimal τ on validation is 0.253
  for A and 0.277 for B; deployed τ values sit close (0.288 / 0.282).

## Results and interpretation

CatBoost won both scenarios. XGBoost and LightGBM finished within ~0.002
AUC of the winner in every split, so the ensemble adds little; the single
CatBoost artifact is the production candidate.

Discrimination is moderate, not strong: valid KS of 0.21–0.25 is typical
for retail unsecured credit on thin bureau files, and consistent with the
literature on Lending Club-style tapes where pricing variables
(`int_rate`, `grade`) carry much of the signal in Scenario B.

Calibration is adequate without post-hoc correction: raw probabilities
beat sigmoid and isotonic on validation Brier in both scenarios, so the
bundles ship uncalibrated models.

### Bias and fairness findings

- **Reject inference** (`reports/reject_inference.md`): the extract holds
  funded loans only, so Scenario A PDs are optimistic for the true
  applicant pool. A sensitivity analysis bounds the bias: applicant-pool
  PD is plausibly 1.4–1.9× the funded estimate, and the funded-tuned
  τ = 0.288 is likely too loose (oracle ≈ 0.12–0.18 under plausible
  reject-risk assumptions). The bias is invisible to model scores; it
  lives in realized outcomes.
- **Adverse impact** (`experiments/adverse_impact.py`): adverse-impact
  ratio < 0.80 on 60-month terms, `small_business` purpose, and the top
  loan-to-income tercile. Income itself is clean.
- **Class imbalance** (`reports/class_imbalance_sampling.md`): SMOTE
  hurts monotonically; random undersampling is AUC-neutral but breaks
  calibration. Resampling is not a lever for gradient-boosted trees here.

## Production artifacts

Each scenario ships a self-contained bundle
(`models/scenario_{a,b}/scenario_{a,b}_bundle.joblib`) containing the
fitted model, calibrator (or none), feature list, categorical levels and
fill values, decision threshold, validation decile reference for PSI
drift monitoring, and split metadata.

```bash
# reproduce the full pipeline
python -m pipeline.scenario_a.prepare && python -m pipeline.scenario_a.train
python -m pipeline.scenario_b.prepare && python -m pipeline.scenario_b.train

# score batches
python -m pipeline.scenario_a.score applications.csv -o scores_a.csv --psi
python -m pipeline.scenario_b.score funded.csv -o scores_b.csv --psi
# add --intervals for Venn-Abers PD intervals (see reports/conformal_coverage.md)

# unified lifecycle scoring (one CSV -> both PDs)
python -m pipeline.infer records.csv -o lifecycle_scores.csv

# explanations: SHAP, expected-cost curve, LR coefficients
python -m pipeline.explain --scenario a
python -m pipeline.explain --scenario b --shap --n 1000
```

Scenario A's `prepare.py` reproduces the canonical nb2 interim pickles
byte-for-byte in values, so `autoresearch/`, the notebooks, and the
pipeline all consume identical artifacts.

## Repository layout

```
notebooks/    1-EDA · 2-Preparation_Preprocessing · 3-Modeling_LogisticRegression
              · 4-Modeling_Tree-based   (nb2 is canonical prep)
src/          workflow modules: performance_window, binning_woe, metrics,
              evaluation, thresholding, calibration, training, ensemble,
              expected_cost_threshold, population_stability_index,
              scorecard, shap_interpretability, manifest
pipeline/     .py production pipeline: core.py engine, scenario_a/ and
              scenario_b/ (prepare/train/score), infer.py, explain.py
data/         raw -> interim -> processed          (DVC-tracked)
models/       serialized bundles                    (DVC-tracked)
artifacts/    binning_artifacts.pkl, manifest.json  (DVC-tracked)
reports/      model cards, metrics JSON/CSV, study reports
autoresearch/ agent experiment loop (temporal-validation eval only)
experiments/  one-off studies (hazard, adverse impact, reject inference)
docs/         method.md (module map) · reproduce.md (full recipe) · data.md (provenance)
tests/        pytest suite (47 tests)
dashboard/    Streamlit Scenario-A scorecard app
```

## Reproducibility

- Environment: conda `credit-risk_VENV`, Python 3.13; `requirements.txt`
  is pinned, `environment.yml` reproduces the full env.
- Versioning: DVC tracks `data/`, `models/`, `artifacts/`; MLflow tracks
  notebook experiments (`sqlite:///mlflow.db`); git tracks code.
- `artifacts/manifest.json` records environment versions, split config,
  and artifact SHA-256 hashes.
- `pytest` runs 47 tests covering performance-window logic (including the
  M-month eligibility boundary), WoE round-trips, cost-threshold argmin,
  PSI/calibration/metric claims on synthetic data, leakage guards, feature
  derivation, reject-augmentation weights, bootstrap CIs,
  and scoring round-trips.

## Agent / contributor orientation

For a fresh agent session:

1. Run `~/.optmem/memo wake` first; it holds cross-session decisions.
2. `conda activate credit-risk_VENV` before running anything that loads
   pickles; they are environment-sensitive.
3. Read order: this README, `notebooks/2-Preparation_Preprocessing.ipynb`,
   `src/` modules, `artifacts/manifest.json`.
4. Notebook roles: nb1 is exploratory only; nb2 is canonical prep;
   nb3 is the LR scorecard track; nb4 is tree models.
5. Hard rules: never touch OOT during model selection; never add
   post-origination columns as features; keep `pytest` green.

## Future directions

- **Reject-level data.** The largest known bias is Scenario A's funded-only
  training population. The ingestion path is implemented
  (`src/reject_inference.py`, `experiments/reject_augmentation.py`):
  given a rejects file it trains on funded loans plus rejects, using
  real outcome labels when available (`--label-col`) or model-PD fuzzy
  labels otherwise. What remains is the data itself — even a small
  sample of rejected applications with bureau attributes. Until then
  the sensitivity bounds in `reports/reject_inference.md` are the
  honest statement of uncertainty.
- **Scheduled retraining.** Rolling-origin evaluation
  (`reports/rolling_oot_evaluation.md`) shows discrimination decays
  gradually (~0.004–0.006 AUC per quarter over 2011–2013) rather than
  breaking suddenly. A semi-annual refit cadence is likely sufficient;
  `experiments/rolling_oot.py` re-validates any retrained bundle cheaply.
  Per-vintage recalibration was tested and rejected
  (`reports/recalibration_drift.md`): quarter-ahead raw probabilities
  are already calibrated (ECE ~0.02), and recalibrators fit on the
  adjacent tail degrade ECE/Brier in most windows — recalibrate only
  at retrain time on a large validation window.
- **Hazard/survival formulation.** Revisited with calibration and a
  gradient-boosted learner (`reports/hazard_model.md`): the discrete-time
  hazard model still trails the binary CatBoost by ~0.008 OOT AUC, but
  calibration fixes its PD level and it uniquely provides PD(t)
  curves. Retained as the right tool if term-structure or
  early-warning use cases appear.
- **Dashboard v2.** Batch scoring, PSI panel, and segment views on top of
  `pipeline/infer.py` outputs; the antislop skill family is installed for
  UI work.
- **Threshold governance.** The gap between statistical optima
  (τ ≈ 0.25) and deployed τ (≈ 0.28–0.29) should be an explicit risk-policy
  decision, revisited quarterly with fresh OOT data.
- **DVC remote.** Move the local `~/dvc_remotes` remote to cloud storage
  once credentials are available.
