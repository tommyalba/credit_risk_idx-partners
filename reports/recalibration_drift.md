# Per-vintage recalibration drift

Session 7 of the robustness roadmap. Script:
`experiments/recalibration_drift.py`; data:
`reports/recalibration_drift_{a,b}.csv` (16 quarterly windows per
scenario, same expanding-window CatBoost refits as
`experiments/rolling_oot.py`).

## Question

Rolling-origin evaluation showed discrimination (AUC) decays
gradually. Does *calibration* decay faster — i.e., should the
recalibration cadence be shorter than the retraining cadence?

## Method

For each quarter: refit production CatBoost on all prior history,
score the quarter raw, and score it again after two recalibrators
fit on the temporally last 15% of that window's training data
(isotonic and logit `a + b*logit(p)`). Reported per window: AUC,
raw slope/intercept/ECE/Brier, and recalibrated ECE/Brier/log-loss.

## Findings

- **Raw probabilities are already well calibrated quarter-ahead.**
  Mean raw ECE is 0.021 (A) / 0.019 (B); mean Brier 0.118 / 0.116.
- **Per-quarter recalibration makes things worse on average.** A
  recalibrator fit on the adjacent tail improved ECE in only 4/16
  (A, isotonic) and 0/16 (B, both methods) windows; mean ECE degrades
  by +0.010 to +0.020 and mean Brier rises ~0.002-0.004. The tail is
  small (hundreds to ~13k loans) and the quarter-to-quarter bad-rate
  mix shifts enough that a freshly fit map mostly adds variance.
- **Calibration slope drifts down modestly, not monotonically.** Mean
  slope is ~1.02 (A) / ~0.96 (B); second-half means fall to 0.99 / 0.90
  vs first-half 1.06 / 1.03 — mild late-sample overconfidence, matching
  `reports/calibration_slope.md`. The drift is slow and within the
  noise band quarter to quarter.
- **Calibration does not decay faster than discrimination.** AUC
  declines ~0.06 peak-to-trough; ECE stays flat near 0.02 and slope
  wanders in [0.82, 1.27] without a clean trend.

## Decision

No separate recalibration cadence is warranted. Recalibrate only at
retraining time, on a large temporally adjacent validation window —
which is what the production bundle selection already does. The
logit recalibrator validated in `reports/calibration_slope.md`
remains a candidate bundle upgrade at the next scheduled retrain,
not a per-quarter operation.
