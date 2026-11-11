# Scenario A Model Card — Application-Time Default Risk

## Scope

Predicts P(bad) **before loan approval/funding**, using only
application-time information. Feature eligibility follows
`notebooks/1_EDA.ipynb` §3.1 "Scenario A" and the canonical nb2
`features_to_selected` list. This is the approve/reject decision model;
Scenario B (`reports/scenario_b_model_card.md`) monitors the funded book.

## Pipeline (`pipeline/scenario_a/`)

| Step | File | Notebook equivalent |
|---|---|---|
| Prep | `prepare.py` | nb2 — writes the SAME `data/interim/` pickles nb2 produces |
| Train | `train.py` | nb4 Stage A/B/C on the canonical interim pickles |
| Score | `score.py` | production inference + PSI drift report |

```bash
python -m pipeline.scenario_a.prepare   # reproduces data/interim/*.pkl
python -m pipeline.scenario_a.train     # -> models/scenario_a/scenario_a_bundle.joblib
python -m pipeline.scenario_a.score applications.csv -o scores.csv --psi
python -m pipeline.infer full_records.csv -o lifecycle_scores.csv   # A + B
```

## Data & splits

Identical to nb2: 117,841 loans after M=33 performance window; train
2007-08→2012-11 (84,764), valid 2012-11→12 (10,596), OOT 2013Q1 (22,481).
Bad rate: 15.3% / 14.8% / 13.9%.

## Features (12 after derivation)

Canonical nb2 set: `annual_inc`, `collections_12_mths_ex_med`,
`delinq_2yrs`, `dti`, `credit_age_years`, `emp_length_ordinal`,
`has_derogatory`, `home_ownership`, `inq_last_6mths`, `loan_to_income`,
`open_acc_ratio`, `purpose`, `revol_util`, `term` (categoricals:
has_derogatory, home_ownership, purpose, term).

Forbidden at scoring time (`FORBIDDEN_FEATURES`): all funding/pricing
fields (`funded_amnt`, `grade`, `int_rate`, `installment`,
`initial_list_status`, …) and all servicing outcomes (`recoveries`,
`out_prncp`, `total_pymnt`, `last_pymnt_*`, …) plus `loan_status`/`target`.

## Results (canonical run — reproduces nb4/autoresearch exactly)

| Model | Valid AUC | OOT AUC |
|---|---|---|
| cat (winner) | **0.6468** | **0.6442** |
| lgbm | 0.6456 | — |
| xgb | 0.6442 | — |

- Calibration: raw won on validation Brier (0.1217).
- Calibration diagnostic (`reports/calibration_slope.md`): logit slope 0.897 valid / 0.881 OOT (b<1 = mildly overconfident), ECE 0.0130/0.0190; valid-fit logit recal transfers to OOT (+~0.001 log-loss).
- Threshold: expected-cost τ = **0.2883**; see bundle for approved-book
- Threshold robustness (`reports/threshold_ci.md`): τ bootstrap 95% CI [0.244, 0.292]; cost curve flat (±0.05 costs ~2%); regret at deployed τ <0.5% mean.
  bad rate.
- Matches `autoresearch` baseline_cat to 6 decimals — the .py pipeline is
  a verified nb4 reproduction.

## Artifacts

`models/scenario_a/scenario_a_bundle.joblib`: model, calibrator,
feature list, category levels + fill values, threshold, PSI reference
deciles, split metadata.

## Relationship to Scenario B

- Scenario A decides **whether to fund**; Scenario B scores the funded
  book for monitoring/early-warning.
- `pipeline/infer.py` runs both on one record set: it strips each
  scenario's forbidden columns and returns `pd_score_application`,
  `decision_application`, `pd_score_funded`, `decision_funded`.
- NB: the standalone `scenario_a.score` CLI **rejects** payloads
  containing funding fields — use `pipeline.infer` for post-funding
  records that legitimately carry them.

## Known limitations

- Reject-inference bias applies here (see `reports/reject_inference.md`):
  funded-only training → applicant-pool PD plausibly 1.4–1.9× higher;
  τ=0.288 likely too loose for the true applicant population.
- OOT is only 2013Q1.
- nb3 LR/scorecard path remains notebook-driven (`dashboard/app.py`
  uses it); the tree bundle is the production candidate.
