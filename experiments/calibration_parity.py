"""Calibration parity across protected-proxy segments (Scenario A).

adverse_impact.py asks whether the *decision* treats segments equally
(approval-rate AIRs). This script asks the stronger question: does the
predicted PD mean the same thing inside each segment? For every
segment level we report

    n, share, observed bad rate, mean predicted PD,
    calibration gap = bad_rate - mean_pd   (>0: under-predicts risk)
    recalibration slope                    (b<1: overconfident in-group)

A segment is flagged when |gap| > GAP_TOL (absolute PD points). Like
adverse_impact.py, segments are proxies — Lending Club data carries no
protected-class labels, so this is directional evidence, not a legal
determination.

Predictions are the production Scenario-A bundle's pd_score (raw
probabilities; bundle calibrator is None), scored through
pipeline.core.prepare_scoring_frame so the frame matches production.

Run from repo root:  python experiments/calibration_parity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import joblib

from src.calibration import calibration_slope_intercept
from src.utils import deserialize_data
from pipeline.core import prepare_scoring_frame
from pipeline.scenario_a.features import (
    add_scenario_a_features, SCENARIO_A_FEATURE_COLUMNS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = f'{ROOT}/models/scenario_a/scenario_a_bundle.joblib'
GAP_TOL = 0.02          # flag |observed - predicted| beyond 2pp
MIN_N = 500             # minimum segment size for a slope estimate
MIN_SHARE = 0.01

GROUP_COLS = ['term', 'purpose', 'home_ownership', 'verification_status',
              'has_derogatory', 'dti_band', 'income_band', 'lti_band',
              'emp_band']


def segment_bands(df):
    """Add the interpretable proxy bands used by adverse_impact.py."""
    out = df.copy()
    out['dti_band'] = pd.qcut(out['dti'], 3, duplicates='drop').astype(str)
    out['income_band'] = pd.qcut(out['annual_inc'], 3,
                                 duplicates='drop').astype(str)
    out['lti_band'] = pd.qcut(out['loan_amnt'] / out['annual_inc'], 3,
                              duplicates='drop').astype(str)
    out['emp_band'] = out['emp_length'].astype(str).replace(
        {'< 1 year': 'lt1', '1 year': '1', '2 years': '2', '3 years': '3',
         '4 years': '4', '5 years': '5', '6 years': '6', '7 years': '7',
         '8 years': '8', '9 years': '9', '10+ years': '10plus'})
    return out


def segment_stats(df, group_col):
    rows = []
    for seg, g in df.groupby(group_col, observed=True):
        n = len(g)
        share = n / len(df)
        if share < MIN_SHARE:
            continue
        bad = float(g['y'].mean())
        pd_ = float(g['p_bad'].mean())
        row = {'segment': str(seg), 'n': n, 'share': share,
               'bad_rate': bad, 'mean_pd': pd_, 'gap': bad - pd_}
        if n >= MIN_N and g['y'].nunique() == 2:
            d = calibration_slope_intercept(g['y'], g['p_bad'])
            row['slope'] = d['slope']
            row['intercept'] = d['intercept']
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        t['flag'] = t['gap'].abs() > GAP_TOL
    return t


def run_split(bundle, split):
    X_raw = deserialize_data(f'{ROOT}/data/interim/X_{split}_raw.pkl')
    y = np.asarray(deserialize_data(f'{ROOT}/data/interim/y_{split}.pkl'))

    X_model = prepare_scoring_frame(X_raw, bundle,
                                   add_scenario_a_features)
    p = bundle['model'].predict_proba(X_model)[:, 1]

    df = segment_bands(X_raw)
    df['has_derogatory'] = X_model['has_derogatory'].astype(str) \
        if 'has_derogatory' in X_model.columns else 'Unknown'
    df['y'] = y
    df['p_bad'] = p

    tables = {}
    for col in GROUP_COLS:
        if col not in df.columns:
            continue
        t = segment_stats(df, col)
        t.insert(0, 'group', col)
        t.insert(0, 'split', split)
        tables[col] = t
    return tables


def main():
    bundle = joblib.load(BUNDLE)
    all_rows = []
    for split in ['valid', 'test']:
        for col, t in run_split(bundle, split).items():
            all_rows.append(t)
            flag = t[t['flag']]
            worst = t.loc[t['gap'].abs().idxmax()]
            print(f'{split:5s} {col:20s} worst gap '
                  f'{worst["gap"]:+.3f} ({worst["segment"]}) | '
                  f'flagged {len(flag)}/{len(t)}')
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(f'{ROOT}/experiments/reports/calibration_parity.csv', index=False)
    print(f'\nsaved experiments/reports/calibration_parity.csv ({len(out)} rows)')


if __name__ == '__main__':
    main()
