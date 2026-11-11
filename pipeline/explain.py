"""Model explanation CLI — SHAP, expected-cost curve, LR coefficients.

Usage:
    python -m pipeline.explain --scenario a                 # all sections
    python -m pipeline.explain --scenario b --shap --n 1000
    python -m pipeline.explain --scenario a --cost
    python -m pipeline.explain --scenario a --lr            # nb3 coefficients
    python -m pipeline.explain --scenario b --input records.csv --shap
    python -m pipeline.explain --scenario a --bootstrap   # metric CIs
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.core import (  # noqa: E402
    prepare_scoring_frame,
    score_records,
)
from pipeline.scenario_a import score as score_a  # noqa: E402
from pipeline.scenario_a.features import (  # noqa: E402
    add_scenario_a_features,
)
from pipeline.scenario_b import score as score_b  # noqa: E402
from pipeline.scenario_b.features import (  # noqa: E402
    add_scenario_b_features,
)
from src.evaluation import extract_positive_proba  # noqa: E402
from src.metrics import bootstrap_metric_ci          # noqa: E402
from src.utils import deserialize_data  # noqa: E402

SCENARIOS = {
    'a': {
        'label': 'Scenario A (application-time)',
        'bundle': score_a.BUNDLE_PATH,
        'derive': add_scenario_a_features,
        'interim': Path('data/interim'),
        'report': 'reports/scenario_a_metrics.json',
    },
    'b': {
        'label': 'Scenario B (post-origination)',
        'bundle': score_b.BUNDLE_PATH,
        'derive': add_scenario_b_features,
        'interim': Path('data/interim/scenario_b'),
        'report': 'reports/scenario_b_metrics.json',
    },
}

LR_MODEL = Path('models/LogisticRegression1.joblib')


def _load(spec_key):
    spec = SCENARIOS[spec_key]
    bundle = joblib.load(spec['bundle'])
    return spec, bundle


def _valid_frame(spec, bundle, input_path=None, n=500):
    """Model-ready features + y_true from the temporal validation split,
    or from a user CSV (--input) with y unavailable."""
    if input_path:
        path = Path(input_path)
        df = (pd.read_parquet(path) if path.suffix == '.parquet'
              else pd.read_csv(path))
        return prepare_scoring_frame(df, bundle, spec['derive']), None
    X_raw = deserialize_data(str(spec['interim'] / 'X_valid_raw.pkl'))
    y = deserialize_data(str(spec['interim'] / 'y_valid.pkl'))
    if n and len(X_raw) > n:
        idx = np.random.RandomState(42).choice(
            len(X_raw), size=n, replace=False)
        X_raw, y = X_raw.iloc[idx], y.iloc[idx]
    return prepare_scoring_frame(X_raw, bundle, spec['derive']), y


def section_shap(spec, bundle, X, n):
    from src.shap_interpretability import (
        compute_shap_values,
        get_global_importance_table,
    )
    print(f'\n=== SHAP (TreeExplainer, n={len(X)}) ===')
    exp = compute_shap_values(bundle['model'], X)
    table = get_global_importance_table(exp, bundle['features'])
    print(table.head(15).to_string(index=False))
    return exp


def section_cost(spec, bundle, X, y):
    if y is None:
        print('\n=== Expected cost ===\n'
              'skipped: --input data has no labels; run without --input '
              'to use the validation split')
        return
    from src.expected_cost_threshold import (
        compare_threshold_criteria,
        expected_cost_curve_scalar,
    )
    p = bundle['model'].predict_proba(X)[:, 1]
    if bundle.get('calibrator') is not None:
        from pipeline.core import _apply_cal
        p = _apply_cal(bundle['calibrator'], bundle['calibrator_name'], p)

    print('\n=== Expected cost (C_FN=8.25M, C_FP=2.7M) ===')
    res = expected_cost_curve_scalar(y, p, c_fn=8_250_000, c_fp=2_700_000)
    print(f'optimal tau : {res["optimal_threshold"]:.4f}')
    print(f'min E[C]     : {res["optimal_cost"]:,.0f}')
    idx = np.argmin(np.abs(res['thresholds'] - bundle['threshold']))
    print(f'bundle tau   : {bundle["threshold"]:.4f} '
          f'(E[C] there: {res["expected_costs"][idx]:,.0f})')
    print('\nthreshold criteria comparison:')
    print(compare_threshold_criteria(y, p, c_fn=8_250_000, c_fp=2_700_000))


def _split_frame(spec, bundle, split):
    """Full temporal split (valid|test) -> (X_model_ready, y)."""
    X_raw = deserialize_data(
        str(spec['interim'] / f'X_{split}_raw.pkl'))
    y = deserialize_data(str(spec['interim'] / f'y_{split}.pkl'))
    return prepare_scoring_frame(X_raw, bundle, spec['derive']), y


def _calibrated_proba(bundle, X):
    p = extract_positive_proba(bundle['model'], X)
    from pipeline.core import _apply_cal
    return _apply_cal(bundle.get('calibrator'),
                      bundle.get('calibrator_name'), p)


def section_bootstrap(spec, bundle, n_boot):
    print(f'\n=== Bootstrap CIs ({n_boot} resamples, 95%) ===')
    rows = []
    for split in ('valid', 'test'):
        X, y = _split_frame(spec, bundle, split)
        p = _calibrated_proba(bundle, X)
        cis = bootstrap_metric_ci(
            y, p, n_boot=n_boot,
            metric_names=('roc_auc', 'log_loss', 'brier_score', 'pr_auc'))
        for metric in cis:
            r = cis[metric]
            rows.append({'split': split, 'metric': metric,
                         'point': r['point'], 'lo': r['lo'],
                         'hi': r['hi']})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def section_lr():
    if not LR_MODEL.exists():
        print(f'\n=== LR coefficients ===\nskipped: {LR_MODEL} not found')
        return
    lr = joblib.load(LR_MODEL)
    coef = getattr(lr, 'coef_', None)
    if coef is None:
        print('\n=== LR coefficients ===\nmodel has no coef_ '
              '(not a linear model)')
        return
    names = (list(lr.feature_names_in_)
             if hasattr(lr, 'feature_names_in_')
             else [f'f{i}' for i in range(coef.shape[1])])
    s = pd.Series(coef.ravel(), index=names).sort_values(
        key=np.abs, ascending=False)
    print('\n=== LogisticRegression coefficients (nb3 scorecard, '
          'WoE-binned features) ===')
    print(s.to_string())


def main():
    ap = argparse.ArgumentParser(description='Explain Scenario A/B models')
    ap.add_argument('--scenario', choices=['a', 'b'], required=True)
    ap.add_argument('--input', help='CSV/parquet to explain instead of '
                    'the validation sample')
    ap.add_argument('--n', type=int, default=500,
                    help='validation rows to sample for SHAP')
    ap.add_argument('--shap', action='store_true')
    ap.add_argument('--cost', action='store_true')
    ap.add_argument('--lr', action='store_true',
                    help='print nb3 LogisticRegression coefficients')
    ap.add_argument('--bootstrap', action='store_true',
                    help='bootstrap 95% CIs on valid + OOT metrics')
    ap.add_argument('--boot', type=int, default=1000,
                    help='bootstrap resamples (with --bootstrap)')
    args = ap.parse_args()

    spec, bundle = _load(args.scenario)
    print(f'{spec["label"]} | winner={bundle["model_family"]} | '
          f'calibration={bundle["calibrator_name"]} | '
          f'tau={bundle["threshold"]:.4f}')
    m = bundle['metrics']
    print(f'valid AUC {m["valid"]["roc_auc"]:.4f} | '
          f'OOT AUC {m["oot_test"]["roc_auc"]:.4f}')

    do_all = not (args.shap or args.cost or args.lr or args.bootstrap)
    X, y = None, None
    if do_all or args.shap or args.cost:
        X, y = _valid_frame(spec, bundle, args.input, args.n)

    if do_all or args.shap:
        section_shap(spec, bundle, X, args.n)
    if do_all or args.cost:
        section_cost(spec, bundle, X, y)
    if do_all or args.bootstrap:
        section_bootstrap(spec, bundle, args.boot)
    if do_all or args.lr:
        if args.scenario == 'a':
            section_lr()
        else:
            print('\n=== LR coefficients ===\nScenario B has no nb3 '
                  'scorecard model; use --scenario a --lr')


if __name__ == '__main__':
    main()
