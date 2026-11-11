"""Scenario B production scoring and drift monitoring.

Loads models/scenario_b/scenario_b_bundle.joblib, validates the payload
(rejects post-origination leakage columns), derives features, scores, and
optionally reports PSI drift vs the validation reference distribution.

Usage:
    python -m pipeline.scenario_b.score input.csv -o scores.csv
    python -m pipeline.scenario_b.score input.csv --psi
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from .features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    SCENARIO_B_FEATURE_COLUMNS,
    add_scenario_b_features,
    apply_calibrator,
    clean_payload,
)

BUNDLE_PATH = Path('models/scenario_b/scenario_b_bundle.joblib')


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def validate_payload(df: pd.DataFrame) -> pd.DataFrame:
    """Reject leakage columns; require the raw columns derivation needs."""
    leaked = [c for c in FORBIDDEN_FEATURES if c in df.columns]
    if leaked:
        raise ValueError(
            f'Post-origination/leakage columns not allowed in scoring '
            f'payload: {leaked}')

    # derivation inputs that must be present as raw values
    derivation_inputs = ['issue_d', 'earliest_cr_line', 'emp_length',
                         'loan_amnt', 'total_acc', 'funded_amnt',
                         'funded_amnt_inv', 'installment', 'int_rate',
                         'annual_inc']
    missing = [c for c in derivation_inputs if c not in df.columns]
    if missing:
        raise ValueError(f'Scoring payload missing required columns: '
                         f'{missing}')
    return df


def prepare_scoring_frame(df: pd.DataFrame, bundle) -> pd.DataFrame:
    """Raw payload -> model-ready frame matching bundle['features']."""
    X = add_scenario_b_features(df)  # cleans payload internally
    feats = bundle['features']
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise ValueError(f'Cannot derive required features: {missing}')
    X = X[feats].copy()

    # align categorical levels to the training category set stored in the
    # bundle; unseen levels become NaN codes (safe degradation)
    levels = bundle.get('categories', {})
    for col in bundle.get('cat_columns', []):
        if col in X.columns:
            cats = levels.get(col)
            X[col] = (pd.Categorical(X[col], categories=cats)
                      if cats else X[col].astype('category'))
    return X


def score(df: pd.DataFrame, bundle=None, id_col='member_id'):
    """Score a batch of post-origination loan applications.

    Returns a DataFrame indexed like the input with pd_score, decision.
    """
    bundle = bundle or load_bundle()
    validate_payload(df)
    X = prepare_scoring_frame(df, bundle)

    p_raw = bundle['model'].predict_proba(X)[:, 1]
    p = apply_calibrator(bundle.get('calibrator'),
                         bundle.get('calibrator_name'), p_raw)

    out = pd.DataFrame(index=df.index)
    if id_col in df.columns:
        out[id_col] = df[id_col]
    out['pd_score'] = np.round(p, 6)
    out['decision'] = np.where(p >= bundle['threshold'], 'REJECT', 'APPROVE')
    return out


def psi_report(new_df: pd.DataFrame, bundle=None, n_bins=10):
    """PSI of scoring features: new batch vs validation reference.

    The bundle stores per-feature decile edges from the temporal validation
    split; new data is binned on those edges and compared to the reference
    distribution implied by the same edges on validation scores.
    """
    from src.population_stability_index import psi_verdict
    bundle = bundle or load_bundle()
    validate_payload(new_df)
    ref = bundle.get('reference_deciles')
    if not ref:
        return pd.DataFrame(columns=['feature', 'psi', 'verdict'])

    X = add_scenario_b_features(new_df)

    rows = []
    for feat, edges in ref.items():
        if feat not in X.columns:
            continue
        s_new = pd.cut(X[feat], bins=edges, include_lowest=True)
        # reference distribution was stored as probabilities per bin
        ref_probs = np.asarray(bundle['reference_probs'][feat])
        new_counts = s_new.value_counts(normalize=True,
                                        sort=False).reindex(
            range(len(ref_probs)), fill_value=0.0).to_numpy()
        psi = float(np.sum(
            (new_counts - ref_probs)
            * np.log((new_counts + 1e-4) / (ref_probs + 1e-4))))
        rows.append({'feature': feat, 'psi': round(psi, 4),
                     'verdict': psi_verdict(psi)})
    rep = pd.DataFrame(rows).sort_values('psi', ascending=False)
    rep.loc['TOTAL'] = ['ALL', round(rep['psi'].sum(), 4), '']
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description='Score Scenario-B loans')
    ap.add_argument('input', help='CSV/parquet of new applications')
    ap.add_argument('-o', '--output', help='write scores CSV here')
    ap.add_argument('--bundle', default=str(BUNDLE_PATH))
    ap.add_argument('--psi', action='store_true',
                    help='also report feature drift vs validation')
    args = ap.parse_args()

    path = Path(args.input)
    df = (pd.read_parquet(path) if path.suffix == '.parquet'
          else pd.read_csv(path))
    bundle = load_bundle(args.bundle)
    result = score(df, bundle)

    if args.psi:
        print(psi_report(df, bundle).to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f'wrote {args.output} ({len(result)} rows)')
    else:
        print(result.head(20).to_string(index=False))
        print(f'\nAPPROVE {(result["decision"] == "APPROVE").mean():.1%} | '
              f'mean PD {result["pd_score"].mean():.4f}')


if __name__ == '__main__':
    main()
