"""Fold reject-level data into Scenario-A training and measure the effect.

Companion to experiments/reject_inference.py (which bounds the bias
without reject data). This script is the implementation path for the
'Reject-level data' future direction: given a file of rejected
applications it builds an augmented training set via
src.reject_inference.build_augmented_dataset, refits the production
CatBoost recipe, and compares funded-validation metrics against the
deployed bundle.

Honest scope: with `--label-col` (real reject outcomes, e.g. bureau
follow-up) this corrects selection bias directly. Without it, fuzzy
augmentation is self-distillation: it reweights covariates but cannot
create outcome information, so funded-valid AUC is expected to stay
roughly flat while score calibration on the applicant pool shifts.

Usage:
    python experiments/reject_augmentation.py --rejects rejects.csv
    python experiments/reject_augmentation.py --rejects rejects.parquet \
        --label-col target --reject-weight 0.5

Never loads data/interim/X_test_raw.pkl or y_test.pkl (OOT untouched).
Writes reports/reject_augmentation.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba          # noqa: E402
from src.metrics import calculate_probability_metrics      # noqa: E402
from src.reject_inference import (                         # noqa: E402
    build_augmented_dataset, load_rejects)
from src.utils import deserialize_data                     # noqa: E402

from pipeline.core import (                                # noqa: E402
    BASE_PARAMS, RANDOM_STATE, prepare_scoring_frame,
    score_records)
from pipeline.scenario_a.features import (                 # noqa: E402
    FORBIDDEN_FEATURES, SCENARIO_A_FEATURE_COLUMNS,
    add_scenario_a_features)

BUNDLE_PATH = ROOT / 'models/scenario_a/scenario_a_bundle.joblib'


def main():
    ap = argparse.ArgumentParser(
        description='Train Scenario A with reject-level augmentation')
    ap.add_argument('--rejects', required=True,
                    help='CSV/parquet of rejected applications')
    ap.add_argument('--label-col', default=None,
                    help='column with real reject outcomes (0/1)')
    ap.add_argument('--reject-weight', type=float, default=1.0,
                    help='sample-weight multiplier for reject rows')
    ap.add_argument('--bundle', default=str(BUNDLE_PATH))
    args = ap.parse_args()

    t0 = time.time()
    X_train_raw = deserialize_data('data/interim/X_train_raw.pkl')
    y_train = deserialize_data('data/interim/y_train.pkl')
    X_valid_raw = deserialize_data('data/interim/X_valid_raw.pkl')
    y_valid = deserialize_data('data/interim/y_valid.pkl')
    bundle = joblib.load(args.bundle)

    rejects_raw = load_rejects(args.rejects)
    print(f'rejects loaded: {len(rejects_raw)} rows')

    if args.label_col is None:
        scored = score_records(rejects_raw, bundle,
                               add_scenario_a_features,
                               FORBIDDEN_FEATURES,
                               SCENARIO_A_FEATURE_COLUMNS)
        reject_pd = scored['pd_score'].to_numpy()
        print(f'reject pseudo-PD: mean={reject_pd.mean():.4f} | '
              f'model would approve '
              f'{(scored["decision"] == "APPROVE").mean():.1%}')
    else:
        reject_pd = None

    X_aug, y_aug, w_aug = build_augmented_dataset(
        X_train_raw, y_train, rejects_raw,
        label_col=args.label_col, reject_pd=reject_pd,
        reject_weight=args.reject_weight)
    n_rej = len(X_aug) - len(X_train_raw)
    print(f'augmented train: {len(X_aug)} rows '
          f'(+{n_rej} reject-derived, weight sum '
          f'{w_aug[len(X_train_raw):].sum():.1f})')

    # 85/15 stratified fit/early-stop split, weights carried along.
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_aug))
    fit_idx, es_idx = train_test_split(
        idx, test_size=0.15, stratify=y_aug, random_state=RANDOM_STATE)

    cat_cols = [c for c in X_aug.columns
                if str(X_aug[c].dtype) == 'category']
    params = dict(BASE_PARAMS['cat'])
    params['cat_features'] = cat_cols
    from catboost import CatBoostClassifier
    est = CatBoostClassifier(**params)
    # _fit_cat_strategy has no sample_weight slot, so fit directly.
    est.fit(X_aug.iloc[fit_idx], y_aug.iloc[fit_idx],
            eval_set=(X_aug.iloc[es_idx], y_aug.iloc[es_idx]),
            sample_weight=w_aug[fit_idx],
            verbose=False, use_best_model=True)

    # Evaluate on funded validation (labels are funded-only by design).
    Xv = add_scenario_a_features(X_valid_raw)
    feats = bundle['features']
    missing = [c for c in feats if c not in X_aug.columns]
    if missing:
        raise ValueError(f'augmented frame missing features: {missing}')
    Xv = Xv.reindex(columns=X_aug.columns)
    for col in cat_cols:
        Xv[col] = pd.Categorical(Xv[col], categories=X_aug[col].cat.categories)

    p_new = extract_positive_proba(est, Xv)
    Xv_bundle = prepare_scoring_frame(X_valid_raw, bundle,
                                      add_scenario_a_features)
    p_old = extract_positive_proba(bundle['model'], Xv_bundle)
    m_new = calculate_probability_metrics(y_valid, p_new)
    m_old = calculate_probability_metrics(y_valid, p_old)

    report = {
        'rejects': str(args.rejects),
        'label_col': args.label_col,
        'reject_weight': args.reject_weight,
        'n_rejects': int(n_rej),
        'valid_auc_baseline': m_old['roc_auc'],
        'valid_auc_augmented': m_new['roc_auc'],
        'valid_logloss_baseline': m_old['log_loss'],
        'valid_logloss_augmented': m_new['log_loss'],
        'valid_mean_pd_baseline': float(np.mean(p_old)),
        'valid_mean_pd_augmented': float(np.mean(p_new)),
        'valid_bad_rate': float(y_valid.mean()),
        'elapsed_s': round(time.time() - t0, 1),
    }
    Path('reports/reject_augmentation.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
