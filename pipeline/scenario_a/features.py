"""Scenario A (application-time) feature specification.

Feature eligibility follows notebooks/1_EDA.ipynb section 3.1
"Scenario A — Application-Time Model": only information available before
loan approval/funding. Derivation reuses the canonical
src.binning_woe.add_derived_features used by nb2/nb4.
"""
from typing import List

import pandas as pd

from src.binning_woe import add_derived_features

# nb2 `features_to_selected` minus 'target' — the canonical 18 raw columns
SCENARIO_A_FEATURE_COLUMNS: List[str] = [
    'annual_inc',
    'collections_12_mths_ex_med',
    'delinq_2yrs',
    'dti',
    'earliest_cr_line',
    'emp_length',
    'home_ownership',
    'inq_last_6mths',
    'issue_d',
    'loan_amnt',
    'open_acc',
    'pub_rec',
    'purpose',
    'revol_bal',
    'revol_util',
    'term',
    'total_acc',
    'verification_status',
]

# Everything that only exists after approval/funding or during servicing.
# Superset of the nb1 Scenario-A removal list plus repayment outcomes.
FORBIDDEN_FEATURES: List[str] = [
    # funding / pricing (Scenario-B territory)
    'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'int_rate',
    'installment', 'initial_list_status',
    # post-origination payment / collection outcomes
    'last_pymnt_amnt', 'last_pymnt_d', 'last_credit_pull_d',
    'recoveries', 'collection_recovery_fee', 'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'pymnt_plan',
    # the label itself must never appear in a scoring payload
    'loan_status', 'target',
]


def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """nb2 upstream cleaning needed before add_derived_features."""
    df = df.copy()
    for col in ('issue_d', 'earliest_cr_line'):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], format='%b-%y',
                                     errors='coerce')
    # nb2 cells 18-19 category cleaning
    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].replace(
            {'ANY': 'Other', 'NONE': 'Other'})
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].replace({'vacation': 'other'})
        df['purpose'] = df['purpose'].fillna('other')
    return df


def add_scenario_a_features(X: pd.DataFrame) -> pd.DataFrame:
    """Canonical nb2/nb4 derivation, tolerant of raw scoring payloads."""
    return add_derived_features(_clean_dates(X))
