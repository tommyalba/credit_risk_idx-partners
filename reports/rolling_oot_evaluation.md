# Rolling-origin out-of-time evaluation

## Purpose

The fixed train/validation/OOT split evaluates each model once, on
2013Q1. A single window cannot distinguish a stable regime from a lucky
quarter, and it cannot show whether the roughly 0.06 AUC gap between the
in-sample training score and the held-out scores is a one-time regime
break or gradual drift. This study refits the production CatBoost recipe
on an expanding window and scores the following calendar quarter,
quarter by quarter, across the full eligible history.

## Method

- Population: the canonical eligible frame (117,841 loans,
  2007-08 to 2013-03 issue dates), reconstructed by concatenating the
  `train`/`valid`/`test` interim pickles. Identical rows for both
  scenarios; only the feature derivation differs.
- Origins: every calendar quarter with at least 3,000 training loans
  and 300 test loans. Scenario A yields 15 windows from 2009Q2; Scenario
  B yields 16 from 2009Q2 (its first window differs by one quarter
  because of an empty early quarter).
- Fit: CatBoost with the production hyperparameters
  (`pipeline.core.BASE_PARAMS['cat']`), an internal 85/15
  fit/early-stop split per window, `task_type="CPU"`. Probabilities are
  raw (uncalibrated); AUC, KS, PR-AUC, log-loss, and Brier are
  computed per window.
- Only CatBoost is evaluated: it won both scenarios, and XGBoost and
  LightGBM tracked within ~0.002 AUC in the fixed split.
- Script: `experiments/rolling_oot.py`. Per-window metrics:
  `reports/rolling_oot_a.csv`, `reports/rolling_oot_b.csv`.

## Results

Quarterly out-of-time ROC-AUC (full metrics in the CSVs):

| Test window | A: n | A: AUC | A: KS | B: n | B: AUC | B: KS |
|---|---|---|---|---|---|---|
| 2009Q2 | 1,499 | 0.647 | 0.227 | 1,499 | 0.684 | 0.278 |
| 2009Q3 | 1,961 | 0.627 | 0.213 | 1,961 | 0.660 | 0.234 |
| 2009Q4 | 2,565 | 0.621 | 0.182 | 2,565 | 0.653 | 0.236 |
| 2010Q1 | 3,066 | 0.666 | 0.246 | 3,066 | 0.698 | 0.291 |
| 2010Q2 | 4,195 | 0.672 | 0.264 | 4,195 | 0.711 | 0.313 |
| 2010Q3 | 4,783 | 0.690 | 0.284 | 4,783 | 0.709 | 0.318 |
| 2010Q4 | 5,149 | 0.707 | 0.320 | 5,149 | 0.720 | 0.340 |
| 2011Q1 | 5,663 | 0.694 | 0.307 | 5,663 | 0.705 | 0.309 |
| 2011Q2 | 6,948 | 0.699 | 0.305 | 6,948 | 0.709 | 0.305 |
| 2011Q3 | 7,964 | 0.695 | 0.295 | 7,964 | 0.701 | 0.289 |
| 2011Q4 | 9,181 | 0.687 | 0.269 | 9,181 | 0.700 | 0.297 |
| 2012Q1 | 11,257 | 0.674 | 0.256 | 11,257 | 0.687 | 0.279 |
| 2012Q2 | 15,020 | 0.665 | 0.243 | 15,020 | 0.682 | 0.264 |
| 2012Q3 | 22,328 | 0.652 | 0.221 | 22,328 | 0.676 | 0.257 |
| 2012Q4 | 25,418 | 0.643 | 0.213 | 25,418 | 0.662 | 0.242 |
| 2013Q1 | 22,481 | 0.646 | 0.212 | 22,481 | 0.670 | 0.257 |

## Findings

**Discrimination is not stationary; it peaks and then decays.** Both
scenarios reach their maximum around late 2010 (A: 0.707 in 2010Q4, B:
0.720 in 2010Q4) and decline roughly monotonically thereafter to
0.646/0.670 by 2013Q1. The decline is gradual, about 0.004–0.006 AUC
per quarter over 2011–2013, not a cliff. Early windows (2009) are
noisy because training sets are small (under 10k loans).

**The fixed-split numbers are consistent with the rolling curve.** The
2013Q1 rolling AUC (0.646 for A, 0.670 for B) matches the fixed-split
OOT values (0.644, 0.669) within noise. The fixed evaluation is not a
lucky quarter; it sits on the same decay trajectory.

**Valid and OOT are flat because they are adjacent, not because the
model is stable.** The rolling curve explains the earlier observation:
2012Q4 scores 0.643/0.662 and 2013Q1 scores 0.646/0.670, a difference
smaller than quarter-to-quarter noise. There is no regime break between
validation and OOT; both sit on the same declining segment of the
curve.

**The train–held-out gap has two components.** First, in-sample
optimism: the reported training AUC (0.70–0.75) is computed on data the
model fitted, which inflates it for any model class, including the
logistic scorecard. Second, and larger, temporal drift: the fixed
training window pools 2007–2012, a period whose early crisis-era
vintages behave differently from later ones. A model evaluated on any
single recent quarter scores lower than its pooled in-sample average.
That the gap appears for logistic regression and all three tree
families alike indicates the cause is in the data, not model capacity.

**Vintage composition drifts with the scores.** Test-window bad rates
move from ~0.11–0.14 in 2009–2011 to ~0.15–0.17 in 2012 before falling
to ~0.14 in 2013Q1, tracking Lending Club's book growth and risk mix.
Score discrimination and outcome prevalence drift together, which is
what PSI monitoring on the feature side is designed to detect.

## Implications

- Expect out-of-time AUC near 0.64–0.67 on future vintages absent
  retraining; the 2010 peak is not recoverable.
- Drift is gradual rather than abrupt, so a scheduled retraining
  cadence (e.g., semi-annual) is likely sufficient; sudden decay would
  argue for event-triggered retraining instead.
- The rolling harness is cheap on this hardware (full run under five
  minutes per scenario) and can be re-run whenever the tape is
  refreshed; it is the natural validation step before promoting a
  retrained bundle.
- Calibration was not applied per window; Brier/log-loss comparisons
  across windows carry a small calibration component, while AUC/KS are
  unaffected.
