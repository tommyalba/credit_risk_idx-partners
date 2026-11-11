# Count-feature representation experiment

Session 6 of the robustness roadmap. Question: do the heavy-zero count
features carry more signal when re-expressed than under the current
`has_derogatory` collapse?

## Setup

`autoresearch/experiment.py` gained a `"counts"` config key (schema
documented in `autoresearch/program.md`). It re-expresses four raw
count columns that `add_derived_features` currently collapses into
`has_derogatory` or keeps raw:

- `delinq_2yrs` (77.5% zeros), `pub_rec` (97.3%),
  `collections_12_mths_ex_med` (99.8%), `inq_last_6mths` (46.9%)

Modes:

- `raw` — the four counts as plain numeric features (14 total).
- `twopart` — per count, a zero-indicator `<c>__nz` plus the positive
  part `<c>__pos` (NaN where zero; CatBoost handles NaN natively),
  18 features total.

Both modes drop `has_derogatory` and the surviving `inq_last_6mths`
column so the comparison isolates the representation. Model: production
CatBoost config (`cat_counts_raw.json`, `cat_counts_twopart.json`),
temporal validation split only, OOT untouched.

A WoE-binned variant was considered and skipped: CatBoost bins numeric
features internally, so an explicit WoE transform duplicates work the
tree already does; WoE is the logistic-regression path's tool.

## Results (validation, champion = 0.646804)

| config | n_features | AUC | logloss | brier | cost_tau |
|---|---|---|---|---|---|
| baseline_cat | 12 | 0.646804 | 0.402443 | 0.121717 | 0.2883 |
| cat_counts_raw | 14 | 0.648668 | 0.401988 | 0.121601 | 0.2613 |
| cat_counts_twopart | 18 | 0.648490 | 0.402047 | 0.121624 | 0.2783 |

Keep rule: beat champion by delta >= 0.002 (AUC > 0.648804).

## Findings

- Both representations improve AUC by ~+0.0017-0.0019 and improve
  log-loss/Brier slightly — a consistent direction, suggesting the
  collapsed `has_derogatory` flag does discard a little signal.
- Neither clears the keep margin, and +0.002 sits well inside the
  bootstrap CI width on this split (~±0.015), so the gain is not
  statistically distinguishable from noise.
- The two modes are statistically indistinguishable from each other
  (0.64867 vs 0.64849): trees already separate zero from positive
  values via splits, so the explicit two-part encoding adds nothing
  the raw counts don't.
- `collections_12_mths_ex_med` is 99.8% zero — effectively dead weight
  in all representations.

## Decision

Discard both configs; `cat_raw` remains champion. The count features'
contribution is captured well enough by `has_derogatory` plus
`inq_last_6mths`. The `counts` key stays in `experiment.py` — it is
the correct harness if reject-level or richer bureau count data ever
arrives.
