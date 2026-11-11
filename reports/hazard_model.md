# Discrete-time hazard model (survival formulation)

Revisit of the v2 experiment listed under Future Directions, now with
calibration, a stronger learner option, and the canonical snapshot date.
Script: `experiments/survival_hazard.py`
(`python experiments/survival_hazard.py --learner {lr,hgb}`).
Per-split metrics: `reports/hazard_model_{lr,hgb}.csv`; test decile
calibration: `reports/hazard_model_{lr,hgb}_deciles.csv`.

## Method

The binary models answer "BAD within M = 33 months?". The hazard model
instead learns the discrete-time hazard h(t|x) = P(event in month t |
survived to t) on person-period data, then composes the survival curve:

    PD(T) = 1 - prod_{t<=T} (1 - h(t|x))

which yields a full PD(t) trajectory per loan, not only PD(33).

- Event: BAD status at `months(issue_d -> last_pymnt_d)`.
- Censoring: Fully Paid at payoff proxy; `Current` at the snapshot.
- All durations capped at M = 33, matching the binary performance window.
- Features: the same WoE-transformed feature set as the nb3 scorecard
  (12 columns) plus period terms (t, t^2, log t) for the baseline hazard.
- Calibration: isotonic regression fit on the last 15% of the training
  frame (temporally adjacent to validation), applied to PD(33). This is
  the addition the original prototype lacked.
- Fix vs the original run: censoring snapshot corrected from
  2015-09-01 to the canonical 2016-01-01.

## Results (Scenario A splits)

| Learner | Split | AUC raw | AUC iso | KS iso | LogLoss iso | Brier iso | Mean PD iso |
|---|---|---|---|---|---|---|---|
| LR | valid | 0.6327 | 0.6317 | 0.194 | 0.4055 | 0.1225 | 0.146 |
| LR | OOT | 0.6340 | 0.6331 | 0.202 | 0.3918 | 0.1169 | 0.145 |
| HGB | valid | 0.6381 | 0.6370 | 0.198 | 0.4121 | 0.1230 | 0.146 |
| HGB | OOT | 0.6373 | 0.6367 | 0.197 | 0.3934 | 0.1178 | 0.146 |
| CatBoost (binary) | valid | — | 0.6468 | 0.215 | 0.4024 | 0.1217 | — |
| CatBoost (binary) | OOT | — | 0.6442 | 0.214 | 0.3897 | 0.1169 | — |

## Findings

- **The hazard formulation still does not beat the binary model.** The
  best variant (HGB, isotonic) reaches OOT AUC 0.6367 vs CatBoost's
  0.6442, a deficit of about 0.008. The earlier conclusion holds after
  calibration and a stronger learner.
- **Calibration fixes the level, not the ranking.** Isotonic
  calibration moves mean PD from 0.175 to ~0.146, matching the realized
  bad rate (~0.14), and improves log-loss/Brier for LR. AUC and KS are
  unchanged by construction (monotone transform). For HGB the isotonic
  step slightly worsened valid log-loss (0.4121 vs 0.4069 raw), a sign
  the raw HGB hazards were already near-calibrated on this split.
- **The decile table is honest.** Empirical bad rates track predicted
  PDs monotonically across all ten deciles for both learners; the
  ranking is sound even though it is weaker than the binary model.
- **What the hazard model buys that binary cannot:** a PD(t) curve per
  loan (e.g. PD at 6/12/24/33 months), principled handling of loans
  censored before M, and a natural venue for expected-loss term
  structures. The binary model cannot express any of these.
- **Why it underperforms:** the hazard objective spends capacity
  modelling the timing of defaults rather than only their occurrence
  within the window, and the person-period expansion multiplies the
  training set ~25x (2.12M rows) while each loan contributes only one
  event. With only 12 WoE features and a single baseline-hazard shape,
  the binary discriminative signal is diluted across periods.

## Conclusion

The survival formulation remains a documented candidate, not a
production component. It is the right tool if the business ever needs
PD as a function of time (e.g. lifetime expected loss, cross-sell
windows, or early-warning triggers at specific horizons); for the
approve/reject and funded-book decisions it does not displace the
binary CatBoost artifacts. The script is cheap to re-run (~45s fit for
HGB on this hardware) and the CSVs make future comparisons one command
away.
