# Reject inference — quantifying Scenario A selection bias

**Question.** The CatBoost PD model is trained only on FUNDED loans
(`loan_data_2007_2014.csv`, 466k rows). Applicants that Lending Club rejected
or that withdrew are unobserved, so PDs are optimistically biased for the true
applicant population. This report quantifies the bias as far as the data
allows — without fabricating reject labels.

**Data check.** No reject/withdrawal/decline labels or covariates exist anywhere
in `data/raw/`, the interim pickles, or notebook findings. The public Lending
Club extract contains funded loans only; `loan_status` is post-origination.
Reject inference in the strict sense (Banaski–Crook style parity/fuzzy
augmentation) is **not possible**: it requires reject covariates or reject
outcomes, neither of which is observable. The bureau-variable augmentation
variant was considered and rejected for the same reason — there are no
application-level features for rejected applicants to augment.

## Method

`scripts`: `experiments/reject_inference.py` → `reports/reject_inference_results.csv`.

1. Score the temporal validation split (10,596 funded loans, Nov–Dec 2012
   vintage) with `models/cat_raw_stage_A_winner.joblib` — the raw-probability
   Stage A winner; nb4 Stage B explicitly kept raw (calibration did not help).
   Sanity: valid AUC = 0.6468, cost-optimal τ = 0.2883 — both replicate nb4.
2. Model the applicant pool as a mixture: funded share `f` ∈ {0.10, 0.20, 0.30}
   (published P2P funding rates are roughly 10–30% of applications) plus reject
   share `1−f`.
3. Reject covariates are unobservable, so reject *model scores* are resampled
   from the funded score distribution (same-X assumption — the model sees no
   difference) and reject *true* PDs are shifted in logit space:
   `PD_true = expit(logit(p_model) + δ)`. δ captures risk visible to Lending
   Club's reject decision but not to the model (soft information, unobserved
   covariates). δ = 0 is the no-bias sanity case.
4. Reject profiles: `same_X` with δ ∈ {0.5, 1.0, 1.5}; `top_decile` (rejects
   resemble the worst funded score decile — a covariate-shift proxy);
   `worst_bound` (every reject is a certain bad — absolute upper bound).
5. Per scenario: applicant-pool realized PD, approved-book realized PD at the
   operational τ = 0.2883, and the oracle cost-optimal threshold on the mixture
   (scalar C_FN = 8.25M / C_FP = 2.7M, same as nb4 Stage C).

## Results (funded share f = 0.20; full grid in CSV)

| Reject profile | Applicant-pool PD | Approved-book PD @ τ=0.288 | Oracle τ |
|---|---|---|---|
| funded only (observed) | 14.8% bad rate | 13.5% | 0.288 |
| same_X, δ=0 (sanity) | 15.3% | 14.0% | 0.249 |
| same_X, δ=0.5 | 21.1% | 19.5% | 0.181 |
| same_X, δ=1.0 | 28.5% | 26.7% | 0.120 |
| same_X, δ=1.5 | 37.1% | 35.3% | 0.081 |
| top_decile | 28.4% | 21.5% | 0.254 |
| worst_bound | 83.0% | 82.7% | 0.000 |

Across f ∈ {0.10, 0.20, 0.30} the pattern is stable: pool PD varies by ~±10%
relative; conclusions do not hinge on the funding-rate assumption.

## Findings

- **The bias is invisible to the model.** Under same-X rejects the model's
  applicant-pool PD estimate stays at 0.154 regardless of δ — selection bias
  lives entirely in realized outcomes, not in model scores.
- **Plausible bias range.** If rejected applicants carry even a moderately
  worse risk profile (δ ≈ 0.5–1.0 logits — e.g. mean model PD 0.15 → true PD
  0.23–0.30), true applicant-pool PD is roughly **1.4–1.9× the funded estimate**
  and approved-book PD roughly **1.4–2.0×** (14.8% → 21–29%). Under the
  covariate-shift proxy (top_decile) the pool PD nearly doubles to ~28%.
- **Threshold is likely too loose.** The funded-tuned τ = 0.288 approves 93% of
  applicants; on the mixture the cost-optimal τ falls to ~0.12–0.18 for
  moderate δ. The worst bound degenerates to reject-all, which is unrealistic
  but shows the direction of the bias.
- **Self-selection partially protects the book.** Reject-like applicants
  concentrate at high model scores, so the operational threshold already
  filters much of the extra risk (top_decile approved-book PD 21.5% vs pool
  28.4%).

## Assumptions & limitations

- Reject score distribution = funded distribution (same-X). True rejects
  likely have *worse observed* covariates too; the `top_decile` profile bounds
  this crudely, and δ can be read as total (observed + unobserved) selection.
- Funded share f = 10–30% is an assumption from published P2P funding rates,
  not measured here.
- Reject true PDs are parametric (logit shift), not estimated — this is a
  sensitivity analysis, not an estimate.
- Funded "true PD" uses realized labels; rejects use model-implied shifted PDs.

## Recommendation for the model card

State explicitly: *"PDs are conditional on loan funding (Scenario A). The
model was trained and validated on funded loans only; rejected/withdrawn
applicants are unobserved. On the validation vintage the funded bad rate is
14.8% and approved-book bad rate 13.5% at τ = 0.288. Sensitivity analysis
(`reports/reject_inference.md`) indicates true applicant-pool PD is plausibly
1.4–1.9× higher and the cost-optimal threshold plausibly lower (~0.12–0.18)
once reject risk is accounted for. Absolute PD calibration for the applicant
population cannot be guaranteed without reject-level data."*

## Reproduce

```bash
conda activate credit-risk_VENV
python experiments/reject_inference.py   # writes reports/reject_inference_results.csv
```
