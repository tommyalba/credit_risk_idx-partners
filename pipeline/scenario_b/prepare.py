"""Scenario B preparation — .py equivalent of notebooks/2 nb2 cells 2-37.

Replicates the canonical nb2 flow (import -> clean -> performance window ->
target -> temporal split) but selects the Scenario-B feature set and writes
interim pickles under data/interim/scenario_b/.

Usage:
    python -m pipeline.scenario_b.prepare
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.performance_window import (  # noqa: E402
    apply_performance_window_filter,
    compute_time_to_bad_distribution,
    eligibility_funnel_report,
    recommend_performance_window,
    truncate_to_safe_horizon,
)
from src.utils import (  # noqa: E402
    deserialize_data,
    importData,
    serialize_data,
    split_input_output,
)

from .features import SCENARIO_B_FEATURE_COLUMNS  # noqa: E402

DATA_PATH = 'data/raw/loan_data_2007_2014.csv'
OUT_DIR = Path('data/interim/scenario_b')
SNAPSHOT_DATE = pd.Timestamp('2016-01-01')
OOT_CUTOFF = pd.Timestamp('2013-01-01')   # same Q1-2013 boundary as nb2
BAD_STATUSES = [
    'Charged Off', 'Default', 'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off',
]


def load_and_clean() -> pd.DataFrame:
    """nb2 cells 2-19: import, index, date/missingness cleaning, categoricals."""
    df = importData(DATA_PATH)
    df = df[sorted(df.columns)].set_index('member_id')

    # nb2 cell 5: parse dates (%b-%y) and fix future-year anomalies first
    date_columns = ['issue_d', 'earliest_cr_line',
                    'last_pymnt_d', 'last_credit_pull_d']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%b-%y', errors='coerce')
        future_mask = df[col].dt.year > 2016
        df.loc[future_mask, col] = df.loc[future_mask, col] - pd.DateOffset(years=100)
    df = df.dropna(subset=date_columns)

    # nb2 canonical drops (data-quality / missingness)
    df = df.drop(columns=['tot_coll_amt', 'tot_cur_bal',
                          'total_rev_hi_lim'])
    df = df.dropna(subset='collections_12_mths_ex_med')
    df = df.dropna(subset='revol_util')

    # nb2 category cleaning
    df['home_ownership'] = df['home_ownership'].replace(
        {'ANY': 'Other', 'NONE': 'Other'})
    df['purpose'] = df['purpose'].replace({'vacation': 'other'})
    df['purpose'] = df['purpose'].fillna('other')

    # parse percentage strings the same way nb2 expects downstream
    for col in ('int_rate', 'revol_util'):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce')

    # ensure required Scenario-B columns exist before use
    missing = [c for c in SCENARIO_B_FEATURE_COLUMNS
               + ['last_pymnt_d', 'loan_status'] if c not in df.columns]
    if missing:
        raise ValueError(f'raw data missing required columns: {missing}')
    return df


def build_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    """nb2 cells 21-22: performance window M + eligibility + target."""
    df_mature = df[df['issue_d'].dt.year.isin([2011, 2012])]
    tt_bad = compute_time_to_bad_distribution(
        df_mature, bad_statuses=BAD_STATUSES,
        issue_col='issue_d', last_pymnt_col='last_pymnt_d',
        status_col='loan_status')
    M = recommend_performance_window(tt_bad, coverage=0.90)
    print(f'M selected: {M} month')

    result = apply_performance_window_filter(
        df, snapshot_date=SNAPSHOT_DATE, M=M)
    print(eligibility_funnel_report(result))

    df_safe = truncate_to_safe_horizon(
        result.df, snapshot_date=SNAPSHOT_DATE, M=M)
    return df_safe


def temporal_split(X: pd.DataFrame, y: pd.Series):
    """nb2 cells 33-36: OOT = 2013Q1+, valid = latest ~11% of history."""
    is_oot = X['issue_d'] >= OOT_CUTOFF
    X_test, y_test = X[is_oot].copy(), y[is_oot].copy()
    X_hist, y_hist = X[~is_oot].copy(), y[~is_oot].copy()

    X_hist = X_hist.sort_values('issue_d')
    n_valid = int(round(len(X_hist) * (10 / 90)))
    X_valid = X_hist.iloc[-n_valid:]
    X_train = X_hist.iloc[:-n_valid]
    # label-based alignment, same as nb2 (positional iloc on y_hist would
    # misalign targets because y_hist is not sorted by issue_d)
    y_valid = y_hist.loc[X_valid.index]
    y_train = y_hist.loc[X_train.index]

    for name, frame in (('X_train', X_train), ('X_valid', X_valid),
                        ('X_test', X_test)):
        print(f'{name}: {len(frame)} rows | '
              f'{frame["issue_d"].min().date()} -> '
              f'{frame["issue_d"].max().date()}')
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main() -> None:
    df = load_and_clean()
    df_safe = build_target_frame(df)

    cols = SCENARIO_B_FEATURE_COLUMNS + ['target']
    X, y = split_input_output(data=df_safe, target_col='target',
                              columns_to_keep=cols)
    # union categoricals so all splits share the same category set
    for col in ('home_ownership', 'purpose', 'term', 'grade',
                'initial_list_status'):
        if col in X.columns:
            X[col] = X[col].astype('category')

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        temporal_split(X, y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # union categorical levels across splits so every split shares one
    # category set (avoids CatBoost/XGB category-index mismatches)
    cat_cols = ['home_ownership', 'purpose', 'term', 'grade',
                'initial_list_status']
    splits = {'train': X_train, 'valid': X_valid, 'test': X_test}
    for col in cat_cols:
        if col not in X_train.columns:
            continue
        union = pd.Categorical(pd.concat(
            [s[col] for s in splits.values()])).categories
        for s in splits.values():
            s[col] = pd.Categorical(s[col], categories=union)

    for name, frame in (('X_train_raw', X_train), ('X_valid_raw', X_valid),
                        ('X_test_raw', X_test)):
        frame = frame[sorted(frame.columns)]
        serialize_data(frame, str(OUT_DIR / f'{name}.pkl'))
    for name, series in (('y_train', y_train), ('y_valid', y_valid),
                         ('y_test', y_test)):
        serialize_data(series, str(OUT_DIR / f'{name}.pkl'))

    print(f'Wrote interim pickles to {OUT_DIR}/')
    print(f'y bad rates -> train {y_train.mean():.4f} | '
          f'valid {y_valid.mean():.4f} | test {y_test.mean():.4f}')


if __name__ == '__main__':
    main()
