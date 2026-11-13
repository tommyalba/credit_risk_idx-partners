# Reject-level data specification

What the bank needs to collect to convert Scenario A's bounded
selection bias into a measured one. Companion to
`experiments/reports/reject_inference.md` (sensitivity bounds used until
this data exists) and `src/reject_inference.py` (ingestion code, already
tested).

## Why this matters

Scenario A trains on **funded loans only**. Rejected applicants are
unobserved, so the model's PDs are conditional on funding. The
sensitivity analysis bounds the distortion (applicant-pool PD plausibly
1.4–1.9× funded-book PD) but cannot measure it. Even a small reject
sample with real outcomes collapses that bound into an estimate.

## Minimum viable dataset

| Requirement | Detail |
|---|---|
| Population | Applications **declined** by the current policy (not withdrawn, not approved-then-defaulted) |
| Sample size | ~1,000+ rows preferred; ~300 gives a usable first correction. Stratify across score bands and decline reasons if possible |
| Features | The 18 application-time columns in `rejects_template.csv` — bureau attributes known **at application time** |
| Labels (high value) | A `target` column: 1 = the applicant later defaulted on a comparable obligation (bureau follow-up, e.g. PR/SLIK 12–24 months post-decision), 0 = performed. Without it the pipeline falls back to fuzzy pseudo-labels — useful for covariate correction but it cannot add outcome information |
| Forbidden | No post-origination fields (`int_rate`, `grade`, `recoveries`, `out_prncp`, …). The loader hard-rejects them |

## Column contract

Exactly the columns in `docs/rejects_template.csv` (header row is
authoritative). Types follow the raw Lending Club extract:

- numerics as numbers (`annual_inc`, `dti`, `loan_amnt`, `revol_util`, …)
- `earliest_cr_line`, `issue_d` as `YYYY-MM-DD` dates
- categoricals as strings matching training levels (`term` includes the
  leading space, e.g. `" 36 months"`; `emp_length` values like
  `"10+ years"`)

Unseen categorical levels are handled at scoring time (mapped to the
training mode), but matching levels exactly gives cleaner results.

## How to run once data arrives

```bash
# fuzzy path (no labels): covariate reweighting only
python experiments/reject_augmentation.py --rejects rejects.csv

# real-label path: direct selection-bias correction
python experiments/reject_augmentation.py --rejects rejects.csv \
    --label-col target --reject-weight 0.5
```

Outputs `experiments/reports/reject_augmentation.json` comparing
funded-validation AUC/log-loss/mean-PD against the deployed bundle.
OOT test is never touched. If the augmented model wins on validation,
promote via `python -m pipeline.retrain`-style review — do **not**
auto-promote; check calibration on the applicant pool, not just funded
validation.

## Privacy

Reject files contain bureau attributes for declined applicants. Handle
under the same data-governance rules as the funded tape (PII minimization,
access logging, no commits to git — `data/` is gitignored; keep the file
in the DVC remote or a secured share).
