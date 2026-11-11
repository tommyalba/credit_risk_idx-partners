"""Scenario-B pipeline tests: leakage guard, feature derivation, scoring."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scenario_b.features import (  # noqa: E402
    FORBIDDEN_FEATURES,
    add_scenario_b_features,
    apply_calibrator,
)
from pipeline.scenario_b.score import (  # noqa: E402
    prepare_scoring_frame,
    score,
    validate_payload,
)


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
        'funded_amnt': 10000.0,
        'funded_amnt_inv': 10000.0,
        'grade': 'B',
        'initial_list_status': 'f',
        'installment': 300.0,
        'int_rate': '12.5%',
        'mths_since_last_delinq': np.nan,
        'mths_since_last_major_derog': np.nan,
        'mths_since_last_record': 24.0,
    }
    row.update(over)
    return row


def test_derived_features_math():
    df = pd.DataFrame([_raw_row(), _raw_row(
        funded_amnt=8000.0, funded_amnt_inv=6000.0,
        installment=200.0, annual_inc=50000.0)])
    out = add_scenario_b_features(df)

    assert out.loc[0, 'investor_funding_ratio'] == pytest.approx(1.0)
    assert out.loc[1, 'investor_funding_ratio'] == pytest.approx(0.75)
    # payment_to_income = installment*12/annual_inc*100
    assert out.loc[0, 'payment_to_income'] == pytest.approx(6.0)
    assert out.loc[1, 'payment_to_income'] == pytest.approx(4.8)
    assert out.loc[0, 'int_rate'] == pytest.approx(12.5)
    assert out.loc[0, 'mths_since_last_delinq'] == 999  # NaN sentinel
    assert 'credit_age_years' in out.columns
    assert 'emp_length_ordinal' in out.columns
    # consumed columns are dropped
    for col in ('funded_amnt_inv', 'issue_d', 'earliest_cr_line'):
        assert col not in out.columns


def test_leakage_columns_rejected():
    df = pd.DataFrame([_raw_row()])
    df['recoveries'] = 0.0
    with pytest.raises(ValueError, match='leakage'):
        validate_payload(df)

    df2 = pd.DataFrame([_raw_row()])
    df2['out_prncp'] = 5000.0
    with pytest.raises(ValueError):
        validate_payload(df2)


def test_missing_derivation_input_rejected():
    df = pd.DataFrame([_raw_row()]).drop(columns=['installment'])
    with pytest.raises(ValueError, match='missing required'):
        validate_payload(df)


def _toy_bundle(n_features=3):
    X = pd.DataFrame({
        'f1': [0.0, 1.0, 0.2, 0.8],
        'f2': ['a', 'b', 'a', 'b'],
        'f3': [1, 2, 3, 4],
    })
    y = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy='prior').fit(X, y)
    return {
        'model': model,
        'calibrator': None,
        'calibrator_name': None,
        'features': ['f1', 'f2', 'f3'],
        'cat_columns': ['f2'],
        'categories': {'f2': ['a', 'b']},
        'threshold': 0.5,
    }


def test_score_returns_decision_frame():
    bundle = _toy_bundle()
    df = pd.DataFrame([_raw_row(), _raw_row(grade='A')])
    # inject the toy feature columns so prepare_scoring_frame can select them
    df['f1'] = [0.1, 0.9]
    df['f2'] = ['a', 'b']
    df['f3'] = [2, 3]

    # prepare_scoring_frame derives features; toy cols pass through only if
    # add_scenario_b_features keeps them — it drops unknown cols? No: it
    # keeps all input cols except drop list, so f1/f2/f3 survive.
    X = prepare_scoring_frame(df, bundle)
    assert list(X.columns) == ['f1', 'f2', 'f3']
    assert str(X['f2'].dtype) == 'category'

    out = score(df.assign(f1=df['f1'], f2=df['f2'], f3=df['f3']), bundle)
    assert {'pd_score', 'decision'} <= set(out.columns)
    assert set(out['decision'].unique()) <= {'APPROVE', 'REJECT'}
    assert len(out) == 2


def test_apply_calibrator_passthrough_and_isotonic():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_calibrator(None, None, p), p)
    assert np.allclose(apply_calibrator(None, 'raw', p), p)

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip').fit(
        [0.0, 0.5, 1.0], [0, 0, 1])
    out = apply_calibrator(iso, 'isotonic', p)
    assert out.shape == (3,)
    assert ((out >= 0) & (out <= 1)).all()


def test_forbidden_list_covers_nb1_exclusions():
    for col in ('recoveries', 'out_prncp', 'total_pymnt',
                'last_pymnt_amnt', 'collection_recovery_fee'):
        assert col in FORBIDDEN_FEATURES
