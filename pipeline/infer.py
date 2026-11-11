"""Unified Scenario-A -> Scenario-B inference workflow.

One entry point for the production lifecycle:

  1. APPLICATION  — borrower applies; only application-time fields exist.
     Scenario A scores PD(default) for the approve/reject decision.
  2. FUNDING+     — loan is funded; pricing/structure fields are now known.
     Scenario B scores PD(default) for ongoing book monitoring.

`score_lifecycle` accepts ONE dataframe containing every raw column and
routes each scenario only the fields it is allowed to see:

  - Scenario A payload = record minus funding/pricing/servicing columns
    (they legitimately exist post-funding but must not reach the model).
  - Scenario B payload = record minus servicing-outcome columns
    (recoveries, out_prncp, ...). At funding time these are legitimately
    zero/unobserved; the strict per-scenario CLIs still hard-reject them.

Usage:
    python -m pipeline.infer applications.csv -o scores.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.core import (  # noqa: E402
    add_conformal_intervals,
    score_records,
)
from pipeline.scenario_a import score as score_a  # noqa: E402
from pipeline.scenario_a.features import (  # noqa: E402
    FORBIDDEN_FEATURES as FORBIDDEN_A,
    SCENARIO_A_FEATURE_COLUMNS,
    add_scenario_a_features,
)
from pipeline.scenario_b import score as score_b  # noqa: E402
from pipeline.scenario_b.features import (  # noqa: E402
    FORBIDDEN_FEATURES as FORBIDDEN_B,
    add_scenario_b_features,
)


def score_lifecycle(df: pd.DataFrame,
                    bundle_a=None, bundle_b=None,
                    id_col='member_id',
                    intervals: bool = False) -> dict:
    """Score one batch through both scenarios.

    Returns {'scenario_a': DataFrame, 'scenario_b': DataFrame} aligned
    on the input index. Scenario B is only scored for rows/records where
    its required funding-time fields are present — callers typically run
    it on the funded subset.
    """
    bundle_a = bundle_a or score_a.load_bundle()
    bundle_b = bundle_b or score_b.load_bundle()

    # --- Scenario A: strip everything it may not see ---
    drop_a = [c for c in FORBIDDEN_A if c in df.columns]
    df_a = df.drop(columns=drop_a)
    res_a = score_records(df_a, bundle_a, add_scenario_a_features,
                          forbidden=[],          # already stripped
                          required_raw=list(SCENARIO_A_FEATURE_COLUMNS),
                          id_col=id_col)
    res_a = res_a.rename(columns={
        'pd_score': 'pd_score_application',
        'decision': 'decision_application'})

    # --- Scenario B: strip servicing outcomes, keep funding fields ---
    drop_b = [c for c in FORBIDDEN_B if c in df.columns]
    df_b = df.drop(columns=drop_b)
    res_b = score_records(df_b, bundle_b, add_scenario_b_features,
                          forbidden=[],          # already stripped
                          required_raw=score_b.REQUIRED_RAW,
                          id_col=id_col)
    res_b = res_b.rename(columns={
        'pd_score': 'pd_score_funded',
        'decision': 'decision_funded'})

    if intervals:
        # Venn-Abers intervals per scenario; renamed to avoid collision
        # in the merged frame. Two isotonic fits per row per scenario.
        res_a = add_conformal_intervals(
            res_a, bundle_a, add_scenario_a_features,
            Path('data/interim'), score_col='pd_score_application')
        res_a = res_a.rename(columns={
            'pd_lower': 'pd_lower_application',
            'pd_upper': 'pd_upper_application',
            'pd_mid': 'pd_mid_application',
            'interval_width': 'interval_width_application'})
        res_b = add_conformal_intervals(
            res_b, bundle_b, add_scenario_b_features,
            Path('data/interim/scenario_b'), score_col='pd_score_funded')
        res_b = res_b.rename(columns={
            'pd_lower': 'pd_lower_funded',
            'pd_upper': 'pd_upper_funded',
            'pd_mid': 'pd_mid_funded',
            'interval_width': 'interval_width_funded'})

    merged = res_a.join(res_b.drop(columns=[id_col], errors='ignore'))
    return {'scenario_a': res_a, 'scenario_b': res_b,
            'combined': merged}


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Unified Scenario-A + Scenario-B scoring')
    ap.add_argument('input', help='CSV/parquet with all raw columns')
    ap.add_argument('-o', '--output', help='write combined scores CSV')
    ap.add_argument('--bundle-a', default=str(score_a.BUNDLE_PATH))
    ap.add_argument('--bundle-b', default=str(score_b.BUNDLE_PATH))
    ap.add_argument('--intervals', action='store_true',
                    help='add Venn-Abers PD intervals per scenario')
    args = ap.parse_args()

    path = Path(args.input)
    df = (pd.read_parquet(path) if path.suffix == '.parquet'
          else pd.read_csv(path))

    result = score_lifecycle(
        df,
        bundle_a=score_a.load_bundle(args.bundle_a),
        bundle_b=score_b.load_bundle(args.bundle_b),
        intervals=args.intervals)

    combined = result['combined']
    print(combined.head(20).to_string(index=False))
    print(f'\nA approve {(combined["decision_application"] == "APPROVE").mean():.1%} | '
          f'B approve {(combined["decision_funded"] == "APPROVE").mean():.1%}')

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f'wrote {args.output} ({len(combined)} rows)')


if __name__ == '__main__':
    main()
