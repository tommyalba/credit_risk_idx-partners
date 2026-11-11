# Reproduce

Environment topology, launch commands, resume behavior, and evaluation.
The README's "Production artifacts" section has the short version; this
file is the full recipe.

## Topology

- OS: WSL2 Ubuntu on Windows. Work inside the Linux filesystem
  (`~/credit_risk_modeling`); avoid `/mnt/c` paths for anything that
  writes.
- Python: conda env `credit-risk_VENV`, Python 3.13. Pickles are
  environment-sensitive: always run inside this env.
- Hardware: developed and validated on a CPU-only laptop
  (i3-1215U, ~8 GB RAM). All models train with `task_type="CPU"` /
  `tree_method="hist"`; a GPU is not required and not used.
- Storage: raw CSV plus DVC-tracked `data/`, `models/`, `artifacts/`.
  `mlruns/` and `mlflow.db` hold notebook-experiment tracking.

## Setup

```bash
conda activate credit-risk_VENV
pip install -r requirements.txt          # or: conda env create -f environment.yml
pytest                                   # 36 tests, expect all green
```

## Launch: full pipeline

```bash
# Scenario A (application-time)
python -m pipeline.scenario_a.prepare
python -m pipeline.scenario_a.train

# Scenario B (post-origination)
python -m pipeline.scenario_b.prepare
python -m pipeline.scenario_b.train
```

Each `train` run refits all three model families, selects the winner on
validation AUC, picks calibration on validation Brier, sets τ by
expected cost, and rewrites the bundle + reports. Scenario A prepare
reproduces the canonical nb2 interim values, so notebooks, autoresearch,
and the pipeline share identical artifacts.

## Launch: scoring and explanation

```bash
python -m pipeline.scenario_a.score applications.csv -o scores_a.csv --psi
python -m pipeline.scenario_b.score funded.csv        -o scores_b.csv --psi
python -m pipeline.infer records.csv                  -o lifecycle.csv
python -m pipeline.explain --scenario a
python -m pipeline.explain --scenario b --shap --n 1000 --input records.csv
```

`--psi` compares the batch's feature distributions against the
validation decile reference embedded in the bundle.

## Launch: autoresearch loop

```bash
python autoresearch/experiment.py --config autoresearch/configs/baseline_cat.json
```

Evaluates one config on temporal validation only; the OOT pickles are
never loaded. See `autoresearch/program.md` for the config schema and
the keep/discard rule (champion = nb4 CatBoost, valid AUC ≈ 0.6468).

## Resume and versioning

- Code and docs: git. `main` is the canonical branch.
- Data/models/artifacts: DVC (`data.dvc`, `models.dvc`, `artifacts.dvc`).
  Restore with `dvc pull`; the current remote is a local directory at
  `~/dvc_remotes`.
- Notebook experiments: MLflow at `sqlite:///mlflow.db`.
- Reproducibility record: `artifacts/manifest.json` stores environment
  versions, split configuration, and SHA-256 hashes of artifacts.

## Evaluation

```bash
pytest -q
```

Covers performance-window eligibility, WoE round-trips, cost-threshold
argmin on synthetic data, leakage guards (Scenario B rejects repayment
fields), feature derivation, and scoring round-trips.

Headline metrics live in `reports/scenario_{a,b}_metrics.json`;
per-family detail in `reports/scenario_{a,b}_leaderboard.csv`; human
summaries in `reports/scenario_{a,b}_model_card.md`.
