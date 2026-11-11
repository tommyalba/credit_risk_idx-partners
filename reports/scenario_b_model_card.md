# Scenario B Model Card — Post-Origination Default Risk

## Scope

Predicts P(bad) for **funded** loans at/after funding, when pricing and
loan-structure fields are known. Feature eligibility follows
`notebooks/1_EDA.ipynb` §3.1 "Scenario B": funding/pricing fields allowed;
repayment/collection outcomes excluded.

This is the production-track counterpart to Scenario A (application-time).
Scenario A remains the pre-funding screening model; Scenario B scores the
book that was actually funded.

## Pipeline (`pipeline/scenario_b/`)

| Step | File | Notebook equivalent |
|---|---|---|
| Prep | `prepare.py` | nb2 cells 2–37 (clean → M=33 window → target → temporal split) |
| Train | `train.py` | nb4 Stage A/B/C (xgb/cat/lgbm, calibration, cost threshold) |
| Score | `score.py` | production inference + PSI drift report |

```bash
python -m pipeline.scenario_b.prepare   # raw CSV -> data/interim/scenario_b/
python -m pipeline.scenario_b.train     # -> models/scenario_b/scenario_b_bundle.joblib
python -m pipeline.scenario_b.score batch.csv -o scores.csv --psi
```

## Data & splits

- Population: 117,841 funded loans after M=33-month performance window
  (identical to Scenario A — same `apply_performance_window_filter`).
- Temporal split: train 2007-08→2012-11 (84,764), valid 2012-11→12
  (10,596), OOT test 2013Q1 (22,481). OOT evaluated once at train time.
- Bad rate: train 15.3% / valid 14.8% / OOT 13.9%.

## Features (28)

Application-time bureau/demographic set **plus** funding-time fields:
`funded_amnt`, `grade`, `initial_list_status`, `installment`, `int_rate`,
`investor_funding_ratio`, `payment_to_income`, `mths_since_last_{delinq,
major_derog,record}`, plus nb2-derived `credit_age_years`,
`emp_length_ordinal`, `has_derogatory`, `loan_to_income`, `open_acc_ratio`.
Categoricals: grade, has_derogatory, home_ownership, initial_list_status,
purpose, term, verification_status.

Excluded by design (enforced at scoring time — `FORBIDDEN_FEATURES`):
`recoveries`, `out_prncp`, `total_pymnt`, `total_rec_*`, `last_pymnt_*`,
`collection_recovery_fee`, `pymnt_plan`. `loan_amount_cut`/`loan_cut_ratio`
from the nb1 list were never implemented and are omitted.

## Results (canonical run)

| Model | Valid AUC | OOT AUC |
|---|---|---|
| cat (winner) | **0.6684** | **0.6687** |
| xgb | 0.6666 | — |
| lgbm | 0.6658 | — |

- Calibration: raw probabilities won on validation Brier (0.1202);
  sigmoid/isotonic offered no gain — bundle stores `calibrator=None`.
- Calibration diagnostic (`reports/calibration_slope.md`): logit slope 0.865 valid / 0.827 OOT (b<1 = mildly overconfident), ECE 0.0119/0.0184; valid-fit logit recal transfers to OOT (+~0.001 log-loss).
- Decision threshold: expected-cost τ = **0.2823** on validation
- Threshold robustness (`reports/threshold_ci.md`): τ bootstrap 95% CI [0.238, 0.302]; cost curve flat (±0.05 costs ~2%); regret at deployed τ <0.5% mean.
  (C_FN=8.25M, C_FP=2.7M); approved-book bad rate on valid = **12.9%**
  (vs 14.8% base rate).
- vs Scenario A cat (valid 0.6468 / OOT 0.644): +0.02 AUC — expected, since
  Scenario B sees lender pricing (`grade`, `int_rate`) which is highly
  informative post-funding.

## Artifacts

`models/scenario_b/scenario_b_bundle.joblib` contains: fitted model,
calibrator (or None), feature list, categorical levels, threshold,
validation decile reference for PSI, split metadata. Gitignored — manage
via DVC like other model artifacts.

## Production use

- Input: CSV/parquet of new funded-loan applications with the raw columns
  in `SCENARIO_B_FEATURE_COLUMNS`. `score.py` rejects payloads containing
  post-origination columns.
- Output: `pd_score`, `decision` (APPROVE if pd < τ).
- Monitoring: `python -m pipeline.scenario_b.score batch.csv --psi` reports
  per-feature PSI vs the validation reference (thresholds: <0.10 stable,
  <0.25 investigate, ≥0.25 significant shift).

## Known limitations

- OOT is only 2013Q1 — same limitation as Scenario A.
- Scenario B does not suffer the reject-inference selection bias that
  affects Scenario A (population = funded loans by construction), but
  `grade`/`int_rate` partially encode Lending Club's own underwriting —
  performance may degrade if LC's grading policy changes.
- Categorical levels unseen at scoring time degrade to NaN codes.
- nb3-equivalent LR/scorecard path not ported: trees are the production
  candidate; WoE binning spec is Scenario-A-specific.
