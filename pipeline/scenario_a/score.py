"""Scenario A production scoring and drift monitoring.

Scores application-time payloads (no funding/pricing fields allowed).

Usage:
    python -m pipeline.scenario_a.score input.csv -o scores.csv [--psi]
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import (  # noqa: E402
    add_conformal_intervals,
    drift_report,
    psi_report as _psi_report,
    score_records,
)
from .features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    SCENARIO_A_FEATURE_COLUMNS,
    add_scenario_a_features,
)

BUNDLE_PATH = Path('models/scenario_a/scenario_a_bundle.joblib')
INTERIM_DIR = Path('data/interim')
REQUIRED_RAW = list(SCENARIO_A_FEATURE_COLUMNS)


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def score(df: pd.DataFrame, bundle=None, id_col='member_id'):
    bundle = bundle or load_bundle()
    return score_records(df, bundle, add_scenario_a_features,
                         FORBIDDEN_FEATURES, REQUIRED_RAW, id_col)


def psi_report(new_df: pd.DataFrame, bundle=None):
    bundle = bundle or load_bundle()
    return _psi_report(new_df, bundle, add_scenario_a_features,
                       FORBIDDEN_FEATURES, REQUIRED_RAW)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Score Scenario-A loan applications')
    ap.add_argument('input', help='CSV/parquet of new applications')
    ap.add_argument('-o', '--output', help='write scores CSV here')
    ap.add_argument('--bundle', default=str(BUNDLE_PATH))
    ap.add_argument('--psi', action='store_true',
                    help='also report feature drift vs validation')
    ap.add_argument('--intervals', action='store_true',
                    help='add Venn-Abers PD intervals (slower on large '
                         'batches)')
    args = ap.parse_args()

    path = Path(args.input)
    df = (pd.read_parquet(path) if path.suffix == '.parquet'
          else pd.read_csv(path))
    bundle = load_bundle(args.bundle)
    result = score_records(df, bundle, add_scenario_a_features,
                           FORBIDDEN_FEATURES, REQUIRED_RAW)

    if args.psi:
        rep = drift_report(df, bundle, add_scenario_a_features,
                           FORBIDDEN_FEATURES, REQUIRED_RAW, INTERIM_DIR)
        print(rep['features'].to_string(index=False))
        print(f"\nscore PSI: {rep['score_psi']:.4f} "
              f"({rep['score_verdict']})")
        print(f"overall drift verdict: {rep['worst_verdict']}")

    if args.intervals:
        result = add_conformal_intervals(
            result, bundle, add_scenario_a_features, INTERIM_DIR)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f'wrote {args.output} ({len(result)} rows)')
    else:
        print(result.head(20).to_string(index=False))
        print(f'\nAPPROVE {(result["decision"] == "APPROVE").mean():.1%} | '
              f'mean PD {result["pd_score"].mean():.4f}')


if __name__ == '__main__':
    main()
