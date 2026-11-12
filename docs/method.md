# Method

What each component does and where it lives. The README covers results and
business interpretation; this file is the engineering map.

## The two scenarios

| | Scenario A | Scenario B |
|---|---|---|
| Decision point | Before approval/funding | After funding |
| Feature spec | `pipeline/scenario_a/features.py` | `pipeline/scenario_b/features.py` |
| Added signal | none (application fields only) | `grade`, `int_rate`, `installment`, `funded_amnt`, `payment_to_income`, `investor_funding_ratio`, `mths_since_last_*` |
| Production use | approve/reject | funded-book monitoring |

Both specs reject repayment and collection fields (`recoveries`,
`out_prncp`, `total_pymnt`, `last_pymnt_*`, `collection_recovery_fee`).
Scenario B's scoring entry point hard-rejects payloads containing those
columns; `pipeline/infer.py` strips them per scenario when routing one
record through both models.

## Pipeline stages

Each scenario has the same four stages under `pipeline/scenario_{a,b}/`:

- `features.py`: raw-column selection, category normalization, derived
  features. Scenario B's file is the larger one (222 lines) because it
  reconstructs the nb4 funding/pricing derivations (`payment_to_income`,
  `investor_funding_ratio`, loan-cut variables).
- `prepare.py`: thin wrapper over `pipeline/core.py::prepare`; writes
  `data/interim/scenario_{a,b}/{X_train_raw,X_valid_raw,X_test_raw,
  y_train,y_valid,y_test}.pkl`.
- `train.py`: wrapper over `core.py::train`; fits CatBoost/XGBoost/
  LightGBM on an 85/15 fit/early-stop split, tunes probability
  calibration and the expected-cost threshold on validation, writes
  `models/scenario_{a}/scenario_{a}_bundle.joblib` plus
  `reports/scenario_{a}_{metrics.json,leaderboard.csv,model_card.md}`.
- `score.py`: batch scorer. Loads the bundle, aligns categories to the
  stored levels, applies the threshold. `--psi` emits the full drift
  report (`core.drift_report`): per-feature characteristic PSI plus
  score-distribution PSI and an overall Stable/Moderate/Significant
  verdict; `--intervals` adds Venn-Abers PD intervals.

`pipeline/core.py` is the shared engine: split construction, model
family dispatch, calibration selection, threshold selection, report
writing, and bundle serialization. Scenario wrappers pass a spec object;
new scenarios should extend the spec, not copy core.

## Shared modules (`src/`)

| Module | Role |
|---|---|
| `performance_window.py` | M=33 eligibility rule; decides which `Current` loans are mature enough to label |
| `utils.py` | pickle I/O, temporal `split_train_test`, date parsing |
| `binning_woe.py` | WoE binning + derived features; feeds the nb3 scorecard and autoresearch configs |
| `training.py` | estimator construction (`build_estimator`), fit strategies |
| `metrics.py`, `evaluation.py`, `ks_statistic_evaluation.py` | AUC, KS, log-loss, Brier |
| `calibration.py`, `calibration_selection.py` | raw vs sigmoid vs isotonic, chosen on validation Brier |
| `expected_cost_threshold.py` | E[C(τ)] curve, argmin, and comparison against Youden/F1 optima |
| `thresholding.py` | generic threshold utilities used by the cost module |
| `population_stability_index.py` | PSI for drift monitoring |
| `scorecard.py` | nb3 LogisticRegression on WoE features (Scenario A interpretability track) |
| `shap_interpretability.py` | SHAP on tree models; values reported on the margin/log-odds scale |
| `ensemble.py` | stacking/ensemble helpers (evaluated; single CatBoost won) |
| `manifest.py` | `artifacts/manifest.json`: env versions, split config, artifact hashes |
| `reject_inference.py` | reject-file ingestion, pseudo-label weights, augmented training set builder |
| `conformal.py` | Venn-Abers PD intervals (`--intervals` on score/infer) |

## Entry points beyond prepare/train/score

- `pipeline/infer.py`: one input CSV, two outputs. Routes each record
  through Scenario A (application PD) and Scenario B (funded PD).
- `pipeline/explain.py`: SHAP importance, expected-cost curve,
  LR coefficients. `--input` accepts an arbitrary CSV for out-of-sample
  explanation; without it, uses the temporal validation split.
- `autoresearch/experiment.py`: evaluates one `configs/*.json` on the
  temporal validation split only. Protocol and keep/discard rule live in
  `autoresearch/program.md`; trials append to
  `autoresearch/results/trials.jsonl`. Configs may set `sampling`
  (fit-split resampling) and `counts` (raw / two-part re-expression of
  heavy-zero count features; see `experiments/reports/feature_representation.md`).
- `experiments/` (+ `experiments/reports/` outputs): one-off studies, not part of the production path:
  `reject_inference.py`, `adverse_impact.py`,
  `calibration_parity.py` (per-segment calibration gap + slope; see
  `experiments/reports/calibration_parity.md`),
  `mths_features_and_lr_interactions.py`, `rolling_oot.py`
  (expanding-window quarterly OOT evaluation; see
  `experiments/reports/rolling_oot_evaluation.md`), `reject_augmentation.py`
  (trains Scenario A with a rejects file; see
  `src/reject_inference.py`), `survival_hazard.py`
  (discrete-time hazard, `--learner lr|hgb`; emits PD(t) cohort curves,
  EL term structure, and time-to-event summaries; see
  `experiments/reports/hazard_model.md` and `experiments/reports/hazard_pd_curves.md`),
  `metric_ci.py`
  (bootstrap CIs on the temporal splits; see `experiments/reports/metric_ci.csv`),
  `recalibration_drift.py` (per-quarter recalibration transfer test;
  see `experiments/reports/recalibration_drift.md`), `calibration_slope.py`
  (logit slope/intercept + ECE + OOT transfer test; see
  `experiments/reports/calibration_slope.md`), `tau_ci.py`
  (bootstrap CI on the cost-optimal threshold; see
  `experiments/reports/threshold_ci.md`).
- `dashboard/app.py`: Streamlit Scenario-A scorecard preview (no
  production scoring path yet).

## Data flow

```
data/raw/loan_data_2007_2014.csv
  -> pipeline/scenario_{a,b}/prepare.py   (dedup, dates, performance window, splits)
  -> data/interim/scenario_{a,b}/*.pkl     (DVC-tracked)
  -> pipeline/scenario_{a,b}/train.py      (models + calibration + tau + reports)
  -> models/scenario_{a,b}/*_bundle.joblib (DVC-tracked)
  -> pipeline/{scenario_*,infer,explain}.py
```
