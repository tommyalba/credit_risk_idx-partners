# Hazard model survival summaries

Session 5 of the robustness roadmap. Extends
`experiments/survival_hazard.py`: the n x 33 monthly hazard matrix is
now computed once per split (`hazard_matrix`), and all horizon
summaries derive from it (`cum_pd`, `median_event_time`).

Outputs per learner (`lr`, `hgb`):

- `reports/hazard_pd_curves_{learner}.csv` — mean cumulative PD at
  6/12/24/33 months for each predicted-risk decile on OOT.
- `reports/hazard_el_term_{learner}.csv` — monthly expected-loss term
  structure on OOT: EAD (`loan_amnt`, USD) x LGD (0.55) x PD(t),
  cumulative and marginal, months 1-33.

## Findings (OOT, 22,481 loans)

### PD(t) curves are front-loaded but persistent

For both learners, roughly a third of the 33-month PD accrues by
month 6 and about half by month 12. The top risk decile reaches
PD(6m) ~0.05 vs ~0.01 for the bottom decile — early-warning value is
real but modest, consistent with the hazard model's weaker ranking
(OOT AUC 0.633-0.637 vs CatBoost 0.644).

### Expected-loss term structure

Cumulative EL at 33 months is ~USD 0.04B on the OOT book
(EAD = loan_amnt, LGD = 0.55). Marginal EL peaks around month 11-15
for both learners — losses are not concentrated in the first year;
roughly half of lifetime expected loss accrues after month 12. A
binary PD model cannot produce this curve; it is the concrete
artifact justifying the hazard formulation as a v2 candidate for
term-structure and reserve work.

### Time-to-event

Observed median time-to-event among OOT bads is 17 months — defaults
cluster early but the median default is well inside the 33-month
window. Predicted median event time is 34 (beyond window) for both
bads and all loans: most individual PD(t) curves never reach 50%
survival-loss within 33 months, which is expected at a ~14% base rate.
The model's value is ranking and calibration of the aggregate curve,
not per-loan median timing.

## Status

Unchanged from `reports/hazard_model.md`: the hazard model remains a
documented candidate, not a production replacement. These summaries
are the deliverables that make that candidacy concrete — PD(t)
curves, EL term structure, and censoring-aware timing — should a
future use case (term-structure pricing, vintage-level provisioning,
early-warning triggers) justify productionizing it.
