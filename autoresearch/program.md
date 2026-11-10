# autoresearch — credit-risk experiment protocol

This directory adapts karpathy/autoresearch's loop to this project: an agent
proposes one candidate config per experiment, `experiment.py` evaluates it on
the **temporal validation split only**, and keep/discard is decided by a
single primary metric.

## The contract

- **Mutable surface**: `configs/*.json` only. One file per experiment.
  `experiment.py` itself should stay fixed — if it needs changes, that is a
  pipeline bug, not an experiment.
- **Untouchable**: `data/interim/X_test_raw.pkl`, `y_test.pkl` (OOT) are never
  loaded by `experiment.py`. Do not evaluate on OOT until a winner is chosen
  by the normal nb4 pipeline.
- **Forbidden features**: post-origination columns (`pymnt_time_month`,
  `recoveries`, `out_prncp`, `total_pymnt`, `last_pymnt_d`, …) and anything
  not present in `data/interim/X_*_raw.pkl` at application time.
- **No refitting upstream**: do not rerun nb2 or change WOE binning,
  `features_to_selected`, the target, or M=33 inside an experiment.

## Config schema (`configs/<name>.json`)

```json
{
  "model": "cat | xgb | lgbm",
  "features": "all" | ["col1", "col2", ...],
  "params": { ...estimator kwargs, merged over the family baseline... },
  "sampling": { "method": "smote | rus | nearmiss | clustercentroids",
                "minority_frac": 0.30 }
}
```

`params` are deep-merged over the family baseline in `experiment.py`
(`BASE_PARAMS`), so a config only needs the keys it changes.
`"cat_features": "__AUTO__"` resolves to the category-dtype columns at
runtime (same as nb4).

`sampling` (optional) resamples ONLY the fit split — es/valid keep the
natural ~15% bad rate, and the cost-optimal threshold is re-tuned on
validation per variant (`cost_tau`, `approved_bad_rate` in the result).
`minority_frac` is the target bad fraction of the resampled set: oversamplers
(`smote` = SMOTENC, needed because features include categoricals) grow the
minority class to it; undersamplers (`rus`, `nearmiss`, `clustercentroids`)
shrink the majority class to it. NaN in numeric features is filled with -1
inside sampled variants only (imblearn rejects NaN).

## Run

```bash
conda activate credit-risk_VENV
python autoresearch/experiment.py --config autoresearch/configs/baseline_cat.json
```

## Scoring & keep/discard

- Primary metric: **validation ROC-AUC** (temporal valid, ~Nov–Dec 2012).
- Reported alongside: log-loss, Brier. Guardrails: valid bad-rate sanity,
  no feature outside the allowed set.
- Keep a config only if AUC improves over the current champion by a margin
  larger than noise — default δ = 0.002 (roughly the bootstrap CI width we
  measured for CatBoost deltas). Record every trial in `results/trials.jsonl`
  (automatic) and note the decision in the commit message.
- Wall-clock budget: aim ≤ 15 min per experiment on this machine
  (i3-1215U, 7.5 GB RAM). CatBoost baseline fit ≈ 1–3 min.

## Champion

Current champion (nb4, temporal split): **cat_raw**, valid AUC ≈ 0.6468,
OOT ≈ 0.6442. Beat that on validation, then promote through nb4 — never by
editing notebooks inside this loop.
