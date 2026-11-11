"""Scenario B production scoring and drift monitoring.

Usage:
    python -m pipeline.scenario_b.score input.csv -o scores.csv [--psi]
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import (  # noqa: E402
    prepare_scoring_frame,
    psi_report as _psi_report,
    score_records,
    validate_payload,
)
from .features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    SCENARIO_B_FEATURE_COLUMNS,
    add_scenario_b_features,
)

BUNDLE_PATH = Path('models/scenario_b/scenario_b_bundle.joblib')
# columns the derivation step needs as raw inputs
REQUIRED_RAW = [
    'issue_d', 'earliest_cr_line', 'emp_length', 'loan_amnt', 'total_acc',
    'funded_amnt', 'funded_amnt_inv', 'installment', 'int_rate',
    'annual_inc',
]


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def score(df: pd.DataFrame, bundle=None, id_col='member_id'):
    bundle = bundle or load_bundle()
    return score_records(df, bundle, add_scenario_b_features,
                         FORBIDDEN_FEATURES, REQUIRED_RAW, id_col)


def psi_report(new_df: pd.DataFrame, bundle=None):
    bundle = bundle or load_bundle()
    return _psi_report(new_df, bundle, add_scenario_b_features,
                       FORBIDDEN_FEATURES, REQUIRED_RAW)


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
    result = score_records(df, bundle, add_scenario_b_features,
                           FORBIDDEN_FEATURES, REQUIRED_RAW)

    if args.psi:
        print(_psi_report(df, bundle, add_scenario_b_features,
                          FORBIDDEN_FEATURES, REQUIRED_RAW)
              .to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f'wrote {args.output} ({len(result)} rows)')
    else:
        print(result.head(20).to_string(index=False))
        print(f'\nAPPROVE {(result["decision"] == "APPROVE").mean():.1%} | '
              f'mean PD {result["pd_score"].mean():.4f}')


if __name__ == '__main__':
    main()
