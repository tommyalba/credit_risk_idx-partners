# Class-imbalance sampling study — oversampling AND undersampling

**Question.** Bad rate ≈ 15%. Does resampling the fit split improve the
CatBoost champion (valid AUC 0.6468)? Prior expectation: oversampling helps
little for GBMs; undersampling hurts via data loss. **Confirmed by the numbers.**

**Setup.** `autoresearch/experiment.py` gained an optional `sampling` config
key (`{"method", "minority_frac"}`) applied to the fit split only — es/valid
keep the natural distribution. `smote` = SMOTENC (required: 4 of 12 features
are categorical); `rus` = RandomUnderSampler. `minority_frac` is the target
bad fraction after resampling. NaN `emp_length_ordinal` (~3% of fit rows) is
filled with −1 inside sampled variants only. The cost-optimal threshold is
**re-tuned on validation per variant** (`cost_tau`) — the old nb3 mistake of
reusing a fixed τ is avoided. All trials in `autoresearch/results/trials.jsonl`.

## Grid results (CatBoost, temporal valid, n=10,596)

| config | sampling | fit n | valid AUC | LogLoss | Brier | cost τ | approved bad rate |
|---|---|---|---|---|---|---|---|
| baseline_cat | none | 72,049 | **0.646804** | **0.402443** | **0.121717** | 0.2883 | 0.1350 |
| cat_smote30 | SMOTENC → 30% bad | ~96k | 0.640366 | 0.406778 | 0.123154 | 0.3013 | 0.1327 |
| cat_smote50 | SMOTENC → 50% bad | ~122k | 0.626594 | 0.419125 | 0.127771 | 0.3483 | 0.1363 |
| cat_rus30 | RUS → 30% bad | ~37k | 0.646164 | 0.462466 | 0.144623 | 0.4815 | 0.1350 |
| cat_rus20 | RUS → 20% bad | ~55k | 0.647112 | 0.411951 | 0.125026 | 0.3473 | 0.1338 |

Champion to beat: **0.6468**; keep margin δ = 0.002.

## Verdict per technique

- **SMOTE (30%)**: AUC −0.0064, all proper scores worse. Synthetic minority
  points add noise, not signal, for a GBM that already splits on the real
  bads. **Reject.**
- **SMOTE (50%)**: AUC −0.020 — monotonically worse with more synthesis.
  Directly refutes the old nb3 BorderlineSMOTE setup (which also used a fixed
  threshold on top). **Reject.**
- **RUS → 30% bad**: AUC −0.0006 (noise) but LogLoss +0.060 — the model is
  trained on a 30%-bad world, so its probabilities are inflated ~2×; the
  re-tuned τ=0.48 compensates for ranking decisions but calibration is
  destroyed. Approved-book bad rate unchanged (0.1350). **Reject.**
- **RUS → 20% bad**: AUC +0.0003 — the only nominal improvement, but far
  below the 0.002 noise margin, and LogLoss/Brier still worse than baseline.
  Mild majority-trimming is approximately AUC-neutral. **Not a keeper.**
- **NearMiss / ClusterCentroids**: not runnable — both need numeric features;
  the 12-feature set includes 4 categorical columns and encoding them would
  change the model's feature space. Skipped as not cheap.

## Threshold sensitivity

Every variant got its own cost-optimal τ (0.29–0.48 vs baseline 0.288). The
re-tuned thresholds mostly compensate for shifted score scales; approved-book
bad rate stays ~0.133–0.136 across all variants — **sampling changes
calibration and ranking marginally, but not the economics of the decision**.

## Conclusion

**No variant beats 0.6468 by >0.002. Keep `cat_raw` unchanged; leave
`use_blsmote=False`.** Oversampling hurts monotonically (SMOTE 30% −0.006,
50% −0.020); undersampling is AUC-neutral at best (RUS-20% +0.0003, within
noise) and wrecks probability calibration at 30%. The prior holds: for
CatBoost on this feature set, class-imbalance resampling is not a lever.

## Reproduce

```bash
conda activate credit-risk_VENV
python autoresearch/experiment.py --config autoresearch/configs/cat_smote30.json
# etc.; results append to autoresearch/results/trials.jsonl
```
