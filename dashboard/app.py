"""Scenario-A credit-score dashboard (Streamlit).

Scores a single new borrower application end-to-end with the exact artifacts
produced by notebooks 2-3:

    raw inputs -> add_derived_features -> transform_binning (WoE)
               -> LogisticRegression PD -> scorecard points -> decision

Run from the repo root:
    streamlit run dashboard/app.py
"""
import os
import sys
import json
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.binning_woe import add_derived_features, transform_binning, transform_bin_labels
from src.scorecard import ScorecardConfig, build_scorecard, score_dataframe, explain_score
from src.thresholding import find_best_cost_threshold

st.set_page_config(page_title='Credit Risk — Scenario A', layout='wide')

RAW_FEATURES = [
    'annual_inc', 'collections_12_mths_ex_med', 'delinq_2yrs', 'dti',
    'earliest_cr_line', 'emp_length', 'home_ownership', 'inq_last_6mths',
    'issue_d', 'loan_amnt', 'open_acc', 'pub_rec', 'purpose', 'revol_bal',
    'revol_util', 'term', 'total_acc', 'verification_status',
]

REVIEW_BAND = 0.02  # |PD - threshold| <= band -> manual review


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(ROOT, 'models/LogisticRegression1.joblib'))
    binning = joblib.load(os.path.join(ROOT, 'artifacts/binning_artifacts.pkl'))
    X_train_lr = joblib.load(os.path.join(ROOT, 'data/processed/logreg/X_train_lr.pkl'))
    feature_names = X_train_lr.columns.tolist()
    scorecard = build_scorecard(
        model=model, binning_artifacts=binning,
        feature_names=feature_names,
        config=ScorecardConfig(target_score=600, target_odds=50, pdo=20),
    )
    # Cost-optimal threshold recomputed on the validation split (as nb3 does)
    X_valid_lr = joblib.load(os.path.join(ROOT, 'data/processed/logreg/X_valid_lr.pkl'))
    y_valid = joblib.load(os.path.join(ROOT, 'data/interim/y_valid.pkl'))
    p_valid = model.predict_proba(X_valid_lr)[:, 1]
    threshold, _ = find_best_cost_threshold(np.asarray(y_valid), p_valid)
    return model, binning, scorecard, float(threshold)


@st.cache_data
def load_categories():
    Xtr = joblib.load(os.path.join(ROOT, 'data/interim/X_train_raw.pkl'))
    return {c: sorted(Xtr[c].dropna().unique().tolist())
            for c in ['home_ownership', 'purpose', 'verification_status', 'emp_length', 'term']}


model, binning, scorecard, THRESHOLD = load_artifacts()
cats = load_categories()
manifest = json.load(open(os.path.join(ROOT, 'artifacts/manifest.json')))

st.title('Credit Risk — New Borrower Scoring (Scenario A)')
st.caption('Application-time PD scorecard · WoE logistic model · '
           'cost-optimal threshold on validation split')

with st.sidebar:
    st.header('Applicant inputs')
    st.caption('Leave optional bureau fields blank if unknown.')

    f = {}
    with st.sidebar:
        c1, c2 = st.columns(2)
        f['annual_inc'] = c1.number_input('Annual income', min_value=0.0, value=75000.0, step=1000.0)
        f['loan_amnt'] = c2.number_input('Loan amount', min_value=0.0, value=15000.0, step=500.0)
        f['dti'] = c1.number_input('DTI (%)', min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        f['revol_util'] = c2.number_input('Revolving utilization (%)', min_value=0.0, max_value=100.0,
                                          value=30.0, step=1.0)
        f['revol_bal'] = c1.number_input('Revolving balance', min_value=0.0, value=10000.0, step=500.0)
        f['inq_last_6mths'] = c2.number_input('Inquiries (6m)', min_value=0, max_value=20, value=1)
        f['delinq_2yrs'] = c1.number_input('Delinquencies (2y)', min_value=0, max_value=20, value=0)
        f['pub_rec'] = c2.number_input('Public records', min_value=0, max_value=20, value=0)
        f['collections_12_mths_ex_med'] = c1.number_input('Collections (12m)', min_value=0, max_value=20, value=0)
        f['open_acc'] = c2.number_input('Open accounts', min_value=0, max_value=100, value=10)
        f['total_acc'] = c1.number_input('Total accounts', min_value=0, max_value=200, value=20)
        f['term'] = c2.selectbox('Term', cats['term'])
        f['purpose'] = st.selectbox('Purpose', cats['purpose'])
        f['home_ownership'] = st.selectbox('Home ownership', cats['home_ownership'])
        f['verification_status'] = st.selectbox('Verification', cats['verification_status'])
        f['emp_length'] = st.selectbox('Employment length', cats['emp_length'])
        f['earliest_cr_line'] = st.date_input('Earliest credit line',
                                              value=date(2010, 1, 1),
                                              min_value=date(1970, 1, 1), max_value=date.today())
        f['issue_d'] = st.date_input('Application/issue date', value=date.today(),
                                     min_value=date(2007, 1, 1), max_value=date.today())


def score_applicant(inputs: dict):
    """Raw inputs -> WoE features -> PD + scorecard score + point breakdown."""
    row = {c: pd.NA for c in RAW_FEATURES}
    row.update(inputs)
    X = pd.DataFrame([row])
    X['issue_d'] = pd.to_datetime(X['issue_d'])
    X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'])

    Xw = transform_binning(X, binning, metric='woe')
    feat_cols = scorecard.feature_names
    p_bad = float(model.predict_proba(Xw[feat_cols])[:, 1][0])
    score = int(score_dataframe(scorecard, X).iloc[0])
    breakdown = explain_score(scorecard, X)
    return p_bad, score, breakdown


p_bad, score, breakdown = score_applicant(f)

# ---------- decision ----------
if abs(p_bad - THRESHOLD) <= REVIEW_BAND:
    decision, color = 'MANUAL REVIEW', 'orange'
elif p_bad >= THRESHOLD:
    decision, color = 'REJECT', 'red'
else:
    decision, color = 'APPROVE', 'green'

m1, m2, m3 = st.columns(3)
m1.metric('Scorecard score', score)
m2.metric('PD (BAD)', f'{p_bad:.2%}')
m3.metric('Decision', decision)

st.markdown(
    f"<div style='border-left:6px solid {color};padding:8px 16px;font-size:1.4rem;'>"
    f"<b style='color:{color}'>{decision}</b> — cost-optimal PD threshold "
    f"<b>{THRESHOLD:.3f}</b> (review band ±{REVIEW_BAND:.0%})</div>",
    unsafe_allow_html=True)

# ---------- scorecard breakdown / adverse action ----------
st.subheader('Scorecard breakdown')
st.dataframe(breakdown, use_container_width=True, hide_index=True)

worst = breakdown[breakdown['characteristic'] != 'Base Score'].nsmallest(3, 'point')
st.markdown('**Adverse-action drivers** (largest point deductions): '
            + ', '.join(f"{r.characteristic} ({r.point} pts)" for r in worst.itertuples()))

# ---------- model card ----------
with st.expander('Model card & limitations'):
    st.json({
        'model': 'WoE LogisticRegression (elasticnet, saga)',
        'scorecard': 'PDO=20, target_score=600 @ 50:1 odds',
        'validation': {'roc_auc': 0.669, 'ks': 0.252},
        'oot_2013q1': {'roc_auc': 0.641, 'ks': 0.217},
        'decision_threshold_pd': THRESHOLD,
        'manifest_git_commit': manifest.get('git_commit'),
        'limitations': [
            'Scenario A (application-time) scope — post-origination fields excluded by design',
            'Trained on funded loans only; reject-inference bias not corrected',
            'OOT covers 2013Q1 only; monitor PSI on live applicants',
            'PD is uncalibrated raw model output; see nb3 Stage 3.6 for calibration variants',
        ],
    })
