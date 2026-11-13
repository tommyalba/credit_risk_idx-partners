# Stakeholder deck outline — credit-risk project results

Deep-technical variant, ~26 slides. Every title is an action title (complete
sentence stating the takeaway). Arc: situation → what was built → what it
delivers → how trustworthy → what could bite us → what to do next.
Numbers verified against `reports/scenario_*_metrics.json`, model cards, and
`experiments/reports/*.md`.

## Act 1 — Context (slides 1–4)

1. **Title** — Credit Risk Modeling: Two Production-Ready PD Models for
   LendingClub-Scale Loan Books. Subtitle: Scenario A (application) + Scenario B
   (funded book), evaluated out-of-time, shipped as reproducible pipeline.
2. **The business problem is a decision, not a metric** — lenders must approve
   or reject at application time and monitor the funded book afterward; both
   need calibrated probabilities of default, not just rankings.
3. **Two models answer two different questions** — Scenario A scores
   applicants before funding (approve/reject); Scenario B scores funded loans
   with pricing data (book monitoring). Shared engine, separate feature
   contracts that hard-block leakage.
4. **The data: 466k loan records, 117,841 usable after maturity filtering** —
   M=33-month performance window labels loans GOOD/BAD; temporal splits
   train 84,764 / valid 10,596 / OOT 22,481; bad rate ~15%.
   *Exhibit: data funnel or split diagram.*

## Act 2 — What was built (slides 5–9)

5. **A four-stage pipeline turns raw loan tapes into scored, monitored PDs** —
   prepare → train → score → drift report, shared `pipeline/core.py` engine.
   *Exhibit: pipeline diagram (diagram-design).*
6. **Leakage is prevented by contract, not convention** — forbidden-feature
   lists are enforced at scoring time; Scenario B hard-rejects payloads
   containing repayment/collection fields.
7. **CatBoost won both scenarios; calibration and threshold are chosen on
   validation, never on the OOT test set** — raw calibration won on Brier;
   expected-cost threshold τ optimized per scenario.
8. **Scenario A delivers AUC 0.645 out-of-time at τ = 0.288** —
   valid 0.6468 / OOT 0.6442; the model reproduces the notebook baseline to
   six decimals.
   *Exhibit: metrics table or leaderboard.*
9. **Scenario B is stronger: AUC 0.669 out-of-time at τ = 0.282** —
   valid 0.6684 / OOT 0.6687; funding/pricing fields (grade, int_rate,
   installment) carry real incremental signal.
   *Exhibit: A vs B comparison.*

## Act 3 — Trustworthiness (slides 10–16)

10. **Discrimination is stable but not spectacular — and that is honest** —
    rolling-origin evaluation shows gradual AUC decay across vintages (~0.64–0.67
    on future quarters), not a one-time regime break; scheduled retraining
    suffices.
    *Exhibit: rolling OOT curve.*
11. **Predictions are calibrated, not just ranked** — raw-probability CatBoost
    already achieves ECE ≈ 0.013–0.019; per-vintage recalibration was tested and
    *hurts* — recalibrate only at retrain.
12. **The decision threshold is statistically robust but politically
    consequential** — bootstrap 95% CI [0.244, 0.292], regret <0.5%; the real
    uncertainty is population bias, not sampling noise.
    *Exhibit: cost curve / τ CI.*
13. **Trained on funded loans only, the model likely overstates applicant
    quality** — reject-inference sensitivity analysis bounds the bias: true
    applicant-pool PD plausibly 1.4–1.9× funded, oracle τ ~0.12–0.18 vs
    deployed 0.288. Absolute PDs need reject-level data to guarantee.
    *Exhibit: bias bounds diagram.*
14. **Fairness screening found three segments needing monitoring** —
    adverse-impact ratio < 0.80 on 60-month terms, small_business purpose, top
    LTI tercile; income is clean; small_business also shows ~5pp
    under-prediction.
15. **Explanations are production-grade: SHAP at loan and portfolio level** —
    global importance + per-loan waterfall + dependence plots; scorecard track
    (WoE logistic regression) preserved for interpretability.
    *Exhibit: SHAP summary figure.*
16. **Every score can carry a prediction interval** — Venn-Abers conformal
    intervals wired into scoring (`--intervals`), giving per-loan PD bounds for
    risk committees.

## Act 4 — Production readiness & risks (slides 17–21)

17. **The system monitors itself: PSI drift reports on every scoring batch** —
    per-feature + score-distribution PSI with Stable/Moderate/Significant
    verdict; the silent PSI binning bug was found and fixed in audit.
18. **Artifacts are reproducible and versioned** — DVC-tracked data/models,
    `manifest.json` env+hash record, MLflow runs, 48-test suite green.
19. **A hazard model exists as a documented v2 candidate** — discrete-time
    survival formulation produces PD(t) curves and expected-loss terms the
    binary model cannot express; it trails on discrimination (OOT AUC 0.637 vs
    0.644) so it stays analytical, not production.
    *Exhibit: PD(t) curve.*
20. **Known limitations are documented, not hidden** — single OOT quarter,
    funded-loan selection bias, modest AUC ceiling on application-time
    features, synthetic-data tests rather than live-portfolio validation.
21. **Class imbalance and alternative learners were tested and rejected
    honestly** — SMOTE hurts monotonically, RUS wrecks calibration, ensembles
    add nothing; CatBoost raw stands.

## Act 5 — Decisions & next steps (slides 22–26)

22. **Recommendation: deploy Scenario A for approve/reject, Scenario B for
    book monitoring** — both bundles reproducible via `python -m
    pipeline.scenario_{a,b}.train`; unified `pipeline/infer.py` scores a record
    through both.
23. **Three decisions are needed from stakeholders** — (a) accept funded-loan
    PDs with documented bias bounds or invest in reject-data collection;
    (b) approve τ governance policy vs statistical tightening; (c) approve
    scheduled retraining cadence (~semi-annual).
24. **Next steps: reject-level data, Streamlit dashboard, retraining
    automation** — dashboard app exists as scaffold; autoresearch loop can
    absorb new experiments cheaply (~15s/trial).
25. **Conclusions: two validated models, honest about their limits, ready for
    pilot** — the deck's claims all trace to committed reports and tests.
26. **Appendix divider** — metrics tables, experiment log, reproduction
    commands.

## Design notes

- Theme: `corporate-clean` or `academic-paper` (white bg, navy `#1F4E79` /
  mid-blue `#2E75B6` per academic-pptx defaults; maps cleanly onto the
  seaborn-deep palette used in nb1 figures).
- Figures reused as-is from `reports/figures/` and `experiments/reports/*.csv`
  rendered charts; pipeline + bias-bounds diagrams via diagram-design.
- Speaker notes (`<aside class="notes">`) optional per slide; presenter mode
  built into the runtime.
