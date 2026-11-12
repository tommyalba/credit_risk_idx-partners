# Calibration parity across segments

Companion to `reports/adverse_impact` findings. Script:
`experiments/calibration_parity.py`; data:
`experiments/reports/calibration_parity.csv` (73 segment rows across 9 groupings,
valid + OOT). Predictions are the production Scenario-A bundle's
`pd_score` (raw probabilities; the bundle calibrator is `None`).

## Question

Adverse-impact ratios test whether the *decision* treats segments
equally. Calibration parity tests the stronger property: does a
predicted PD of, say, 0.15 mean the same observed default frequency
inside every segment?

## Method

For each segment level (share >= 1%, n >= 500 for slope estimates):
observed bad rate, mean predicted PD, calibration gap
(`bad_rate - mean_pd`, positive = under-predicts risk), and an
in-group recalibration slope (`logit(y) ~ a + b*logit(p)`). A segment
is flagged when |gap| > 0.02. Same proxy caveat as adverse impact:
no protected-class labels exist, so these are directional findings.

## Findings

- **Level**: the model slightly over-predicts on both splits
  (mean PD 0.154 vs bad 0.148 valid; 0.156 vs 0.139 OOT), consistent
  with the negative intercept in `experiments/reports/calibration_slope.md`.
- **Within-group slopes are mostly < 1** (mean 0.91 valid / 0.89 OOT):
  the overconfidence found globally also holds inside segments.
- **Worst OOT gaps**: 60-month term over-predicts by 5.7pp
  (predicted 0.245 vs observed 0.188); `small_business` purpose
  *under*-predicts by 5.1pp (0.261 vs 0.312 observed) and `other`
  purpose by 2.9pp. These are the segments where PD does not mean the
  same thing — `small_business` is simultaneously the AIR-flagged
  group, so its risk is understated while its approval rate is also
  depressed.
- **Purpose is the least calibrated grouping on OOT**: 5 of 6 levels
  flagged. `home_improvement` shows the flattest in-group slope (0.68).
- **Income and DTI bands are comparatively clean**: income_band flags
  only on level (uniform over-prediction ~2pp), not on slope; dti_band
  flags nothing on either split.

## Decision

Calibration parity broadly holds — gaps are a few pp and mostly share
the global over-prediction direction — but two exceptions warrant
monitoring: `small_business` (under-predicted risk plus adverse
impact) and 60-month term (over-predicted risk). Neither justifies a
model change on this evidence; both should be re-checked when
reject-level data or richer purpose granularity arrives. Segment-level
calibration is now part of the fairness evidence base alongside AIR.
