# Conformal PD intervals (Venn-Abers)

Session 1 of the robustness roadmap: per-loan prediction intervals for
the deployed PD scores. Implementation: `src/conformal.py`
(`venn_abers_intervals`), wired into `pipeline/core.py`
(`calibration_scores`, `add_conformal_intervals`) and exposed as
`--intervals` on `pipeline/scenario_{a,b}/score.py` and
`pipeline/infer.py`.

## Method

For each scored loan with model score s, isotonic calibration is fit
twice on the temporal-validation conformity set (n = 10,596): once with
the candidate label forced to 0, once forced to 1. The interval
`[min(p0, p1), max(p0, p1)]` brackets a calibrated probability for that
loan; the midpoint `(p0 + p1) / 2` is the Venn-Abers point predictor.
No distributional assumptions; validity is marginal over the
calibration/scored exchangeability.

## Verification on OOT (2,000-row subsample per scenario)

| Scenario | Mean width | Median width | P90 width | pd_score inside own interval |
|---|---|---|---|---|
| A | 0.0029 | 0.0016 | 0.0054 | 12.5% |
| B | 0.0028 | 0.0011 | 0.0043 | 6.0% |

Midpoint reliability on OOT is reasonable: decile empirical bad rates
track midpoint predictions monotonically for both scenarios (see the
session transcript / rerun for the full table).

## Honest interpretation — read before using

- **The intervals are narrow because the conformity set is large.**
  Venn-Abers width reflects uncertainty in the *calibration map*, which
  shrinks as n_cal grows. With 10.6k calibration loans, widths of
  ~0.003 are expected.
- **They do not include model-training variance.** The interval is
  conditional on the fitted bundle; it says nothing about how the model
  itself would change under resampling. For total predictive
  uncertainty, combine with the bootstrap metric CIs
  (`reports/metric_ci.csv`) or the τ-sensitivity analysis (Session 4).
- **pd_score can fall outside its own interval** (it does ~90% of the
  time here). That is normal: the interval brackets the imprecise
  calibrated probability, not the point score. Use `pd_mid` as the
  conformal point estimate if you want the calibrated value.
- **Decision use:** near τ = 0.288, an interval of ±0.002 means the
  calibration layer is confident; residual uncertainty lives in model
  drift (PSI) and threshold governance, not in per-loan calibration.

## Usage

```bash
python -m pipeline.scenario_a.score applications.csv -o scores.csv --intervals
python -m pipeline.scenario_b.score funded.csv -o scores.csv --intervals
python -m pipeline.infer records.csv -o lifecycle.csv --intervals
```

Adds `pd_lower`, `pd_upper`, `pd_mid`, `interval_width` (suffixes
`_application`/`_funded` under `infer`). Runtime is two isotonic fits
per row per scenario — fine for application batches, subsample large
backfills.
