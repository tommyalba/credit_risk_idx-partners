"""Scenario B (post-origination) feature specification and derivation.

Feature eligibility follows notebooks/1_EDA.ipynb section 3.1
"Scenario B — Post-Origination Model": funding/pricing features are known
at prediction time; repayment/collection outcomes remain excluded.

`loan_amount_cut` / `loan_cut_ratio` from the nb1 list are intentionally
absent: they were never implemented in code (documentation-only names).
"""
from typing import List

import numpy as np
import pandas as pd

from src.binning_woe import extract_emp_length_years

# ---------------------------------------------------------------------------
# Columns carried through preparation (nb2 keeps these plus Scenario-B
# funding/pricing fields). `last_pymnt_d` is needed upstream for the
# performance-window maturity calc but is NOT a model feature.
# ---------------------------------------------------------------------------
SCENARIO_B_FEATURE_COLUMNS: List[str] = [
    # application-time bureau / demographic (shared with Scenario A)
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
    # post-origination / funding-time fields (Scenario B only)
    'funded_amnt',
    'funded_amnt_inv',   # consumed to build investor_funding_ratio, then dropped
    'grade',
    'initial_list_status',
    'installment',
    'int_rate',
    'mths_since_last_delinq',
    'mths_since_last_major_derog',
    'mths_since_last_record',
]

# nb1 Scenario-B list minus the two never-implemented cut columns and the
# nb1-era aliases, mapped to canonical/derived names:
#   credit_history_length -> credit_age_years
#   emp_length_num        -> emp_length_ordinal
#   tot_coll_amt/tot_cur_bal/total_rev_hi_lim stay dropped (nb2 canonical
#   data-quality decision applies to both scenarios)
SCENARIO_B_MODEL_FEATURES: List[str] = [
    'annual_inc',
    'collections_12_mths_ex_med',
    'credit_age_years',
    'delinq_2yrs',
    'dti',
    'emp_length_ordinal',
    'funded_amnt',
    'grade',
    'has_derogatory',
    'home_ownership',
    'initial_list_status',
    'inq_last_6mths',
    'installment',
    'int_rate',
    'investor_funding_ratio',
    'loan_to_income',
    'mths_since_last_delinq',
    'mths_since_last_major_derog',
    'mths_since_last_record',
    'open_acc',
    'open_acc_ratio',
    'payment_to_income',
    'pub_rec',
    'purpose',
    'revol_bal',
    'revol_util',
    'term',
    'verification_status',
]

SCENARIO_B_CATEGORICALS: List[str] = [
    'grade',
    'has_derogatory',
    'home_ownership',
    'initial_list_status',
    'purpose',
    'term',
    'verification_status',
]

# Post-origination / repayment-outcome fields that must NEVER appear in a
# scoring payload. Enforced by score.py at inference time.
FORBIDDEN_FEATURES: List[str] = [
    'total_pymnt',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'last_pymnt_amnt',
    'recoveries',
    'collection_recovery_fee',
    'out_prncp',
    'out_prncp_inv',
    'total_pymnt_inv',
    'last_pymnt_d',
    'last_credit_pull_d',
    'pymnt_plan',
    # the label itself must never appear in a scoring payload
    'loan_status', 'target',
]


def clean_payload(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the fields prepare.py parses upstream (dates, %, categories)."""
    df = df.copy()
    for col in ('issue_d', 'earliest_cr_line'):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], format='%b-%y',
                                     errors='coerce')
    for col in ('int_rate', 'revol_util'):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce')
    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].replace(
            {'ANY': 'Other', 'NONE': 'Other'})
    if 'purpose' in df.columns:
        df['purpose'] = df['purpose'].replace({'vacation': 'other'})
        df['purpose'] = df['purpose'].fillna('other')
    return df


def add_scenario_b_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive Scenario-B model features from raw/prep columns.

    Accepts either raw payloads (dates/percent strings) or the pickled
    prep frames; `clean_payload` normalizes the former.
    """
    out = clean_payload(X)

    credit_history_age_month = (
        (out['issue_d'].dt.year - out['earliest_cr_line'].dt.year) * 12
        + (out['issue_d'].dt.month - out['earliest_cr_line'].dt.month)
    ).fillna(0).astype(int)
    out['credit_age_years'] = credit_history_age_month / 12
    out['emp_length_ordinal'] = out['emp_length'].apply(extract_emp_length_years)

    condition = (
        (out['collections_12_mths_ex_med'] > 0)
        | (out['delinq_2yrs'] > 0)
        | (out['pub_rec'] > 0)
    )
    out['has_derogatory'] = np.where(condition, 1, 0)
    out['has_derogatory'] = out['has_derogatory'].apply(
        lambda v: 'Unknown' if pd.isna(v)
        else ('No_Derogatory' if v == 0 else 'Has_Derogatory')
    ).astype('category')

    out['loan_to_income'] = out['loan_amnt'] / out['annual_inc']
    out['open_acc_ratio'] = out['open_acc'] / out['total_acc'].replace(0, np.nan)

    # nb1-derived Scenario-B ratios
    out['investor_funding_ratio'] = np.where(
        out['funded_amnt'] > 0,
        out['funded_amnt_inv'] / out['funded_amnt'],
        0.0,
    )
    out['payment_to_income'] = np.where(
        out['annual_inc'] > 0,
        (out['installment'] * 12) / out['annual_inc'] * 100,
        0.0,
    )

    # int_rate may arrive as e.g. '12.34%' — parse to float
    if not pd.api.types.is_numeric_dtype(out['int_rate']):
        out['int_rate'] = pd.to_numeric(
            out['int_rate'].astype(str).str.replace('%', '', regex=False),
            errors='coerce')

    # mths_since_last_* use NaN for "never"; sentinel 999 matches the
    # convention tested in the nb4 mths_since_last_* experiment.
    for col in ('mths_since_last_delinq', 'mths_since_last_major_derog',
                'mths_since_last_record'):
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(999)

    # categorical coercion (consistent categories across splits happen via
    # union_categoricals in prepare.py; here we only fix dtypes)
    for col in ('term', 'home_ownership', 'purpose', 'grade',
                'initial_list_status', 'verification_status'):
        out[col] = out[col].astype('category')

    # drop raw columns consumed by derivation or excluded by design
    drop_cols = [
        'earliest_cr_line', 'emp_length', 'issue_d', 'loan_amnt',
        'funded_amnt_inv', 'total_acc',
        'last_pymnt_d', 'last_credit_pull_d',
        'sub_grade', 'funded_amnt_funded',
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


def apply_calibrator(calibrator, name: str, p_raw: np.ndarray) -> np.ndarray:
    """Apply a stored calibrator to raw positive-class probabilities."""
    if calibrator is None or name is None:
        return np.asarray(p_raw)
    if name == 'sigmoid':
        return calibrator.predict_proba(np.asarray(p_raw).reshape(-1, 1))[:, 1]
    if name == 'isotonic':
        return calibrator.predict(np.asarray(p_raw))
    raise ValueError(f'unknown calibrator {name!r}')
