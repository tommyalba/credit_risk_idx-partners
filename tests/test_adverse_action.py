"""Tests for src/adverse_action.py reason-code mapping."""
import pandas as pd

from src.adverse_action import adverse_action_reasons, REASON_MAP


def _shap_table(rows):
    """rows: list of (feature, value, shap_value)."""
    return pd.DataFrame([
        {'feature': f, 'feature_value': v, 'shap_value': s,
         'abs_shap_value': abs(s),
         'direction': 'raises risk (pushes toward BAD)' if s > 0
                      else 'lowers risk (pushes toward GOOD)'}
        for f, v, s in rows
    ])


def test_adverse_reasons_only_positive_shap():
    tbl = _shap_table([('dti', 30.0, 0.5), ('revol_util', 80.0, -0.2),
                       ('term', ' 60 months', 0.1)])
    reasons = adverse_action_reasons(tbl)
    assert reasons == [
        'High debt-to-income ratio',
        'Loan term longer than standard ( 60 months)',
    ]


def test_adverse_reasons_respect_top_n_and_order():
    tbl = _shap_table([('dti', 30.0, 0.9), ('revol_util', 80.0, 0.4),
                       ('inq_last_6mths', 5.0, 0.3)])
    reasons = adverse_action_reasons(tbl, top_n=2)
    assert reasons[0] == 'High debt-to-income ratio'
    assert len(reasons) == 2


def test_value_substitution_for_categoricals():
    tbl = _shap_table([('purpose', 'small_business', 0.4)])
    reasons = adverse_action_reasons(tbl)
    assert 'small_business' in reasons[0]


def test_unknown_feature_falls_back():
    tbl = _shap_table([('mystery_feat', 1.0, 0.3)])
    reasons = adverse_action_reasons(tbl)
    assert 'mystery_feat' in reasons[0]


def test_no_adverse_factors_returns_empty():
    tbl = _shap_table([('dti', 30.0, -0.5)])
    assert adverse_action_reasons(tbl) == []


def test_reason_map_covers_scenario_a_bundle_features():
    import joblib
    from pathlib import Path
    bundle_path = Path('models/scenario_a/scenario_a_bundle.joblib')
    if not bundle_path.exists():
        import pytest
        pytest.skip('bundle not present (DVC data unavailable)')
    bundle = joblib.load(bundle_path)
    missing = [f for f in bundle['features'] if f not in REASON_MAP]
    assert not missing, f'features without reason codes: {missing}'
