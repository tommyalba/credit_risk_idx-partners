"""Scenario-A pipeline tests: leakage guard, derivation, scoring."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scenario_a.features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    SCENARIO_A_FEATURE_COLUMNS,
    add_scenario_a_features,
)
from pipeline.scenario_a.score import REQUIRED_RAW, score  # noqa: E402
from pipeline.core import validate_payload  # noqa: E402


def _raw_row(**over):
    row = {
        'member_id': 'M001',
        'annual_inc': 60000.0,
        'collections_12_mths_ex_med': 0.0,
        'delinq_2yrs': 0.0,
        'dti': 15.0,
        'earliest_cr_line': 'Jan-05',
        'emp_length': '5 years',
        'home_ownership': 'RENT',
        'inq_last_6mths': 1.0,
        'issue_d': 'Jan-14',
        'loan_amnt': 10000.0,
        'open_acc': 10.0,
        'pub_rec': 0.0,
        'purpose': 'debt_consolidation',
        'revol_bal': 5000.0,
        'revol_util': 30.0,
        'term': ' 36 months',
        'total_acc': 15.0,
        'verification_status': 'Verified',
    }
    row.update(over)
    return row


def test_feature_columns_match_nb2_canonical():
    assert SCENARIO_A_FEATURE_COLUMNS == [
        'annual_inc', 'collections_12_mths_ex_med', 'delinq_2yrs', 'dti',
        'earliest_cr_line', 'emp_length', 'home_ownership',
        'inq_last_6mths', 'issue_d', 'loan_amnt', 'open_acc', 'pub_rec',
        'purpose', 'revol_bal', 'revol_util', 'term', 'total_acc',
        'verification_status',
    ]


def test_derived_features_canonical():
    out = add_scenario_a_features(pd.DataFrame([_raw_row()]))
    for col in ('credit_age_years', 'emp_length_ordinal', 'has_derogatory',
                'loan_to_income', 'open_acc_ratio'):
        assert col in out.columns
    # canonical drops
    for col in ('issue_d', 'earliest_cr_line', 'emp_length', 'loan_amnt',
                'revol_bal', 'total_acc', 'verification_status'):
        assert col not in out.columns


def test_funding_fields_rejected_for_scenario_a():
    df = pd.DataFrame([_raw_row()])
    df['int_rate'] = 12.5
    with pytest.raises(ValueError, match='leakage'):
        validate_payload(df, FORBIDDEN_FEATURES, REQUIRED_RAW)

    df2 = pd.DataFrame([_raw_row()])
    df2['grade'] = 'B'
    with pytest.raises(ValueError):
        validate_payload(df2, FORBIDDEN_FEATURES, REQUIRED_RAW)

    df3 = pd.DataFrame([_raw_row()])
    df3['recoveries'] = 0.0
    with pytest.raises(ValueError):
        validate_payload(df3, FORBIDDEN_FEATURES, REQUIRED_RAW)


def test_missing_application_field_rejected():
    df = pd.DataFrame([_raw_row()]).drop(columns=['dti'])
    with pytest.raises(ValueError, match='missing required'):
        validate_payload(df, FORBIDDEN_FEATURES, REQUIRED_RAW)


def test_score_roundtrip_with_toy_bundle():
    # build a toy bundle whose features exist after derivation
    derived = add_scenario_a_features(pd.DataFrame([_raw_row()]))
    feats = sorted(derived.columns)[:3]
    X = pd.DataFrame({
        'f_a': [0.0, 1.0], 'f_b': [0.2, 0.9], 'f_c': [1, 2]})
    model = DummyClassifier(strategy='prior').fit(X, [0, 1])
    bundle = {
        'model': model, 'calibrator': None, 'calibrator_name': None,
        'features': ['f_a', 'f_b', 'f_c'], 'cat_columns': [],
        'categories': {}, 'threshold': 0.5,
    }
    df = pd.DataFrame([_raw_row(), _raw_row(dti=20.0)])
    # inject toy features; they pass through derivation untouched
    df['f_a'] = [0.1, 0.8]
    df['f_b'] = [0.3, 0.7]
    df['f_c'] = [1, 2]
    out = score(df, bundle)
    assert {'pd_score', 'decision'} <= set(out.columns)
    assert len(out) == 2
