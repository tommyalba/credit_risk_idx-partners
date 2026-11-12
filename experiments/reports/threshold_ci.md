# Bootstrap CI for the decision threshold τ

Session 4 of the robustness roadmap. Script: `experiments/tau_ci.py`;
data: `experiments/reports/threshold_ci.csv` (500 bootstrap resamples of the
temporal validation split, C_FN = 8.25M / C_FP = 2.7M IDR).

## Question

The deployed τ is a point estimate. Is the E[C(τ)] optimum flat (τ
robust to sampling noise) or sharp (τ needs tight governance)? And
what does the deployed bundle τ cost relative to the per-resample
optimum?

## Results

| Scenario | τ bundle | τ* on valid | τ* bootstrap 95% CI | median τ* | P(|τ*−τ|≤0.02) | regret mean | regret p90 |
|---|---|---|---|---|---|---|---|
| A | 0.2883 | 0.2883 | [0.244, 0.292] | 0.272 | 51% | ~5,100 IDR | ~13,700 IDR |
| B | 0.2823 | 0.2823 | [0.238, 0.302] | 0.280 | 74% | ~5,600 IDR | ~11,700 IDR |

Deterministic sensitivity on the full validation curve (relative
E[C] increase vs optimum):

| Δτ | A | B |
|---|---|---|
| −0.05 | +1.4% | +1.9% |
| −0.02 | +0.5% | +0.7% |
| +0.02 | +1.3% | +0.7% |
| +0.05 | +2.3% | +2.0% |

## Findings

- **The cost curve is flat near the optimum.** Moving τ by ±0.05 costs
  only 1–2% of minimum expected cost; ±0.02 costs under 1%. The exact
  τ value matters far less than getting the ballpark right.
- **Deployed τ sits inside the bootstrap noise band.** For B, 74% of
  resamples put τ* within ±0.02 of the bundle value. For A it is 51% —
  the bundle τ (0.288) sits near the upper edge of the A τ* interval
  [0.244, 0.292], consistent with it being tuned with a slight
  conservatism margin over the statistical optimum (~0.27 median).
- **Regret at the deployed τ is small.** Mean ~5k IDR per applicant
  against a minimum E[C] of ~1.1M — under 0.5%. Even the p90 regret
  (~12–14k) is ~1.2% of minimum cost.
- **Governance implication.** τ does not need tight statistical
  control; it needs *policy* control. The binding constraint is the
  reject-inference bias documented in `experiments/reports/reject_inference.md`
  (funded-tuned τ likely too loose for the true applicant pool,
  oracle ≈ 0.12–0.18 under plausible assumptions) — a far larger
  effect than the ±0.05 sampling band measured here.

## Decision

Keep the deployed τ values. Revisit τ on the scheduled retraining
cadence and whenever reject-level data arrives; the statistical
uncertainty around τ is an order of magnitude smaller than the
population-bias uncertainty.
