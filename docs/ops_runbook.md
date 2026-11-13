# Operations runbook

How the production scoring system is deployed, monitored, retrained, and
rolled back. Audience: whoever is on call when a batch scores weird or a
vintage drifts.

## Components

| Piece | Entry point | Artifact |
|---|---|---|
| Scenario A (application PD) | `pipeline/scenario_a/{prepare,train,score}.py` | `models/scenario_a/scenario_a_bundle.joblib` |
| Scenario B (funded-book PD) | `pipeline/scenario_b/{prepare,train,score}.py` | `models/scenario_b/scenario_b_bundle.joblib` |
| Unified lifecycle scoring | `pipeline/infer.py`, `pipeline/serve.py` | both bundles |
| Drift monitoring | `score.py --psi`, `POST /v1/score/monitor` | `drift_report` verdicts |
| Prediction intervals | `--intervals` flag / `?intervals=true` | Venn-Abers bounds per loan |
| Explanations | `pipeline/explain.py`, SHAP in `src/shap_interpretability.py` | per-loan reason lists |

## Deploy

1. `conda activate credit-risk_VENV` (Python 3.13, pinned `requirements.txt`).
2. `dvc pull` for `data/`, `models/`, `artifacts/` if not present.
3. Batch scoring: `python -m pipeline.scenario_{a,b}.score batch.csv -o scores.csv --psi`
4. API scoring: `uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000`
   - `GET /health` — bundle presence check
   - `POST /v1/score/application` — approve/reject PDs
   - `POST /v1/score/lifecycle` — both scenarios on one record set
   - `POST /v1/score/monitor` — PSI drift verdict for a batch (`?scenario=a|b`)
5. Streamlit decision UI (analyst-facing, Scenario-A scorecard track):
   `streamlit run dashboard/app.py`

## Monitor

Every scoring batch should carry `--psi` (CLI) or hit `/v1/score/monitor`
(API). The report gives a worst-of verdict across feature-level PSI and
score-distribution PSI:

| Verdict | Meaning | Action |
|---|---|---|
| `Stable` | distributions within bands | none; log the report |
| `Moderate shift -- investigate` | one or more features drifting | inspect `top_features`, compare to last vintage; check data pipeline upstream |
| `Significant shift -- recalibration/redevelopment likely required` | broad drift | trigger retraining (§ below); do not keep scoring silently |

Drift is expected to be gradual (see `experiments/reports/rolling_oot_evaluation.md`);
a sudden Significant verdict is more likely a data-pipeline bug than a
market shift — check ingestion first.

Score-level monitoring: watch the batch mean PD and approval rate vs the
validation vintage (bad rate 14.8%, approved-book bad rate 13.5% at τ).
A mean-PD jump without a PSI verdict usually means the input mix changed
in a way PSI bins don't capture.

## Retrain

Cadence: semi-annual scheduled, or PSI-triggered early. The rolling-OOT
harness (`experiments/rolling_oot.py`) is the validation step before
promoting any retrained bundle.

```bash
python -m pipeline.retrain --scenario a --data data/raw/new_tape.csv --promote
python -m pipeline.retrain --scenario b --data data/raw/new_tape.csv --promote
```

`pipeline/retrain.py` archives the incumbent bundle + reports to
`models/archive/scenario_{a,b}/<timestamp>/`, trains the challenger into
`models/scenario_{a,b}/staging/`, and applies a non-inferiority gate:
challenger OOT ROC-AUC must be >= incumbent − 0.005 (tune with
`--min-auc-margin`). On gate failure the incumbent is restored
automatically. Without `--promote` the gate verdict is reported but the
live bundle stays put.

After promotion: rerun `pytest tests/ -q`, commit the new
`reports/scenario_*` artifacts, and note the change in the model card.

## Rollback

Bundles are plain joblib files with archived predecessors:

```bash
ls models/archive/scenario_a/            # timestamped snapshots
cp models/archive/scenario_a/<ts>/scenario_a_bundle.joblib \
   models/scenario_a/scenario_a_bundle.joblib
# restore matching reports from the same archive dir if needed
```

DVC tracks `models/`, so `dvc checkout` also restores a previously
committed bundle state.

## Known biases to keep in mind

- PDs are conditional on funding (Scenario A). Applicant-pool PD is
  plausibly 1.4–1.9× funded-book PD; oracle τ ~0.12–0.18 vs deployed
  0.288 (`experiments/reports/reject_inference.md`).
- Segment exceptions under monitoring: `small_business` under-predicts
  ~5pp, 60-month term over-predicts ~6pp OOT
  (`experiments/reports/calibration_parity.md`).
- τ is policy-controlled, not statistically fragile: bootstrap 95% CI
  [0.244, 0.292] for A (`experiments/reports/threshold_ci.md`).

## Verification

- `pytest tests/ -q` — 48+ tests covering pipelines, metrics, PSI,
  conformal intervals, reject inference, threshold math.
- `python -m pipeline.infer smoke.csv -o out.csv` — end-to-end scoring.
- API smoke: `curl localhost:8000/health` then POST a record to
  `/v1/score/application`.
