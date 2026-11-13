"""Adverse-action reason codes derived from SHAP explanations.

Maps model features to standardized, plain-language reason phrases of
the kind used in credit adverse-action notices (ECOA-style reason
factors; validate final wording with compliance before use — see
README §Dashboard follow-ups).

`adverse_action_reasons` takes the per-record SHAP contributor table
(from src.shap_interpretability.explain_individual_prediction) and
returns reason strings for the features that pushed the prediction
toward default, strongest first.
"""
from typing import List

import pandas as pd

# feature -> (adverse phrase when SHAP > 0, mitigating phrase when < 0)
# {value} is replaced with the observed feature value where useful.
REASON_MAP = {
    'annual_inc': (
        'Annual income is low relative to the requested loan',
        'Annual income is strong relative to the requested loan',
    ),
    'credit_age_years': (
        'Limited length of credit history',
        'Long, established credit history',
    ),
    'dti': (
        'High debt-to-income ratio',
        'Manageable debt-to-income ratio',
    ),
    'emp_length_ordinal': (
        'Short employment history',
        'Stable, long-term employment',
    ),
    'has_derogatory': (
        'History of late or delinquent payments',
        'No history of late or delinquent payments',
    ),
    'home_ownership': (
        'Housing status: {value} (higher-risk category)',
        'Housing status: {value} (lower-risk category)',
    ),
    'inq_last_6mths': (
        'Too many recent credit inquiries',
        'Few recent credit inquiries',
    ),
    'loan_to_income': (
        'Requested loan amount is high relative to income',
        'Requested loan amount is modest relative to income',
    ),
    'open_acc_ratio': (
        'Many open accounts relative to total accounts',
        'Balanced open-account ratio',
    ),
    'purpose': (
        'Loan purpose "{value}" is associated with higher risk',
        'Loan purpose "{value}" is associated with lower risk',
    ),
    'revol_util': (
        'High revolving credit utilization',
        'Low revolving credit utilization',
    ),
    'term': (
        'Loan term longer than standard ({value})',
        'Standard loan term ({value})',
    ),
}

FALLBACK_ADVERSE = 'Elevated risk on factor: {feature}'


def _phrase(feature: str, value, adverse: bool) -> str:
    adverse_txt, mitigating_txt = REASON_MAP.get(
        feature, (FALLBACK_ADVERSE, 'Favorable factor: {feature}'))
    tpl = adverse_txt if adverse else mitigating_txt
    try:
        return tpl.format(feature=feature, value=value)
    except (KeyError, IndexError):
        return tpl


def adverse_action_reasons(shap_table: pd.DataFrame,
                           top_n: int = 4) -> List[str]:
    """Reason phrases for adverse (risk-raising) SHAP contributors.

    Parameters
    ----------
    shap_table : DataFrame from explain_individual_prediction['table']
        Columns: feature, feature_value, shap_value, direction, ...
    top_n : max reasons returned, strongest |SHAP| first.

    Returns
    -------
    List[str] of standardized reason phrases, e.g.
    'High debt-to-income ratio'. Empty list if nothing raised risk.
    """
    adverse = shap_table[shap_table['shap_value'] > 0]
    reasons = [
        _phrase(r.feature, r.feature_value, adverse=True)
        for r in adverse.head(top_n).itertuples()
    ]
    return reasons
