"""Adverse-impact / fairness check for the Scenario-A decision policy.

Uses the Stage-A winner (cat_raw) predictions on the temporal validation and
OOT test splits. For each protected-proxy segment we report:

    n, portfolio share, bad rate, mean PD, approval rate @ cost threshold,
    and Adverse Impact Ratio (AIR) = group approval rate / best group rate.

Convention: AIR < 0.80 (four-fifths rule) is flagged for review. Lending Club
data has no protected-class labels, so segments are *proxies* — the check is
directional evidence, not a legal determination.

Run from repo root:  python experiments/adverse_impact.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import joblib

from src.binning_woe import add_derived_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.288288  # nb4 Stage-C cost-optimal threshold (cat_raw, validation)
FOUR_FIFTHS = 0.80

def load_split(name):
    X = joblib.load(f'{ROOT}/data/interim/X_{name}_raw.pkl')
    y = joblib.load(f'{ROOT}/data/interim/y_{name}.pkl')
    return X, np.asarray(y)

def segment_frame(X_raw, X_derived, y, p_bad):
    """Group on raw attributes; borrow derived has_derogatory by index."""
    df = X_raw.copy()
    df['has_derogatory'] = X_derived['has_derogatory']
    df['y'] = y
    df['p_bad'] = p_bad
    df['approve'] = (df['p_bad'] < THRESHOLD).astype(int)
    # interpretable proxy segments (all application-time)
    df['dti_band'] = pd.qcut(df['dti'], 3, duplicates='drop').astype(str)
    df['income_band'] = pd.qcut(df['annual_inc'], 3, duplicates='drop').astype(str)
    df['lti_band'] = pd.qcut(df['loan_amnt'] / df['annual_inc'], 3,
                             duplicates='drop').astype(str)
    df['emp_band'] = df['emp_length'].astype(str).replace(
        {'< 1 year': 'lt1', '1 year': '1', '2 years': '2', '3 years': '3',
         '4 years': '4', '5 years': '5', '6 years': '6', '7 years': '7',
         '8 years': '8', '9 years': '9', '10+ years': '10plus'})
    return df

def impact_table(df, group_col, min_share=0.01):
    g = (df.groupby(group_col, observed=True)
           .agg(n=('y', 'size'), bad_rate=('y', 'mean'),
                mean_pd=('p_bad', 'mean'), approval=('approve', 'mean'))
           .reset_index()
           .rename(columns={group_col: 'segment'}))
    g['share'] = g['n'] / g['n'].sum()
    g = g[g['share'] >= min_share].copy()
    g['air'] = g['approval'] / g['approval'].max()
    return g.sort_values('air')

def report(df, label):
    print(f'\n================ {label} ================')
    print(f'n={len(df):,} | base bad={df.y.mean():.3f} | '
          f'approval={df.approve.mean():.3f} | thr={THRESHOLD}')
    tables = {}
    for col in ['term', 'purpose', 'home_ownership', 'has_derogatory',
                'verification_status', 'dti_band', 'income_band', 'lti_band',
                'emp_band']:
        if col not in df.columns:
            continue
        t = impact_table(df, col)
        tables[col] = t
        flagged = t[t['air'] < FOUR_FIFTHS]
        print(f'\n--- {col} ---')
        print(t.round(3).to_string(index=False))
        if len(flagged):
            print(f'  !! AIR<0.80: {flagged["segment"].tolist()}')
    return tables

def main():
    model = joblib.load(f'{ROOT}/models/cat_raw_stage_A_winner.joblib')
    all_tables = {}
    for split in ['valid', 'test']:
        X_raw, y = load_split(split)
        Xd = add_derived_features(X_raw)
        p = model.predict_proba(Xd[model.feature_names_])[:, 1]
        df = segment_frame(X_raw, Xd, y, p)
        all_tables[split] = report(df, f'{split.upper()} (OOT)' if split == 'test' else split.upper())

    out = {}
    for split, tabs in all_tables.items():
        for col, t in tabs.items():
            out[f'{split}|{col}'] = t
    pd.concat(out, names=['split', 'group']).to_csv(
        f'{ROOT}/reports/adverse_impact.csv')
    print(f'\nsaved reports/adverse_impact.csv')

if __name__ == '__main__':
    main()
