# Calibration slope/intercept diagnostic

Session 2 of the robustness roadmap. Script:
`experiments/calibration_slope.py`; data:
`experiments/reports/calibration_slope.csv`. Primitives added to
`src/calibration.py`: `calibration_slope_intercept`,
`fit_logit_recalibration`, `apply_logit_recalibration`,
`expected_calibration_error`.

## Method

The logistic recalibration map `logit(y) ~ a + b * logit(p)` is fit on
the temporal validation split of deployed bundle probabilities, then
interpreted two ways:

- **Own-fit diagnostic**: slope/intercept refit on each split's own
  data, showing whether calibration drifted between valid and OOT.
- **Transfer test**: the valid-fit recalibrator applied unchanged to
  OOT, measuring whether the correction generalizes or just fits
  validation noise.

ECE (expected calibration error, 10 quantile bins) is reported per
version as a scalar companion to the slope.

## Results

| Scenario | Split | Slope (own fit) | Intercept | ECE | LogLoss | Brier |
|---|---|---|---|---|---|---|
| A | valid, deployed | 0.897 | −0.217 | 0.0130 | 0.4024 | 0.1217 |
| A | valid, recal | 0.897 | −0.217 | 0.0094 | 0.4021 | 0.1216 |
| A | OOT, deployed | 0.881 | −0.327 | 0.0190 | 0.3897 | 0.1169 |
| A | OOT, recal-from-valid | 0.881 | −0.327 | 0.0138 | 0.3887 | 0.1165 |
| B | valid, deployed | 0.865 | −0.260 | 0.0119 | 0.3968 | 0.1202 |
| B | valid, recal | 0.865 | −0.260 | 0.0048 | 0.3961 | 0.1200 |
| B | OOT, deployed | 0.827 | −0.399 | 0.0184 | 0.3841 | 0.1155 |
| B | OOT, recal-from-valid | 0.827 | −0.399 | 0.0094 | 0.3827 | 0.1149 |

## Findings

- **Scores are mildly overconfident.** Slope b < 1 on every split
  (0.83–0.90): predicted probabilities are slightly too extreme — high
  PDs a touch too high, low PDs a touch too low. Typical for
  gradient-boosted trees.
- **Calibration degrades slightly OOT, consistent with drift.** A:
  0.897 → 0.881; B: 0.865 → 0.827. The slope loss is larger for B,
  matching its larger feature drift (pricing variables).
- **Intercept is negative and grows on OOT** (−0.22 → −0.33 for A,
  −0.26 → −0.40 for B): deployed PDs overestimate risk slightly more
  out-of-time, i.e. the model is a touch pessimistic on 2013 vintages.
- **ECE is small** (0.012–0.019) — calibration is already adequate;
  the logit recalibrator mostly re-centers the level.
- **The valid-fit recalibrator transfers.** Applied unchanged to OOT it
  improves ECE by ~30–50% relative and shaves ~0.001 off log-loss and
  ~0.0005 off Brier on both scenarios. The gain is real but small.

## Decision

A logit recalibrator is justified but not urgent. The improvement is
consistent across both scenarios and holds OOT, so it is a legitimate
bundle upgrade — but it requires refitting bundles (`train` writes
`calibrator` only from the sigmoid/isotonic candidates, so this would
be a small `core.py` change plus retrain). Recommended as a bundled
change at the next scheduled retraining (see Scheduled retraining in
Future Directions) rather than as a standalone artifact rebuild. Until
then the deployed raw probabilities remain acceptable: ECE ≤ 0.02.
