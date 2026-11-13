"""Credit-risk decision support dashboard (Streamlit).

Two tracks in one app:

  1. Single applicant — WoE logistic scorecard (nb3 artifacts) with
     point-level adverse-action breakdown.
  2. Batch scoring — production CatBoost bundles via pipeline.infer:
     scenario or lifecycle scoring, PSI drift vs the validation
     reference, and segment views (mean PD vs realized bad rate).

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
from pipeline.core import drift_report, score_records
from pipeline.infer import score_lifecycle
from pipeline.scenario_a import score as score_a
from pipeline.scenario_a import features as feat_a
from pipeline.scenario_b import score as score_b
from pipeline.scenario_b import features as feat_b

st.set_page_config(page_title='Credit Risk Dashboard', layout='wide')

APPROVE_COLOR, REJECT_COLOR, REVIEW_COLOR = '#1a7f37', '#cf222e', '#bf8700'

RAW_FEATURES = [
    'annual_inc', 'collections_12_mths_ex_med', 'delinq_2yrs', 'dti',
    'earliest_cr_line', 'emp_length', 'home_ownership', 'inq_last_6mths',
    'issue_d', 'loan_amnt', 'open_acc', 'pub_rec', 'purpose', 'revol_bal',
    'revol_util', 'term', 'total_acc', 'verification_status',
]

REVIEW_BAND = 0.02  # |PD - threshold| <= band -> manual review

SCENARIOS = {
    'a': dict(label='Scenario A — new applications',
              interim='data/interim', derive=feat_a.add_scenario_a_features,
              forbidden=feat_a.FORBIDDEN_FEATURES,
              required=score_a.REQUIRED_RAW,
              loader=score_a.load_bundle),
    'b': dict(label='Scenario B — funded loans',
              interim='data/interim/scenario_b',
              derive=feat_b.add_scenario_b_features,
              forbidden=feat_b.FORBIDDEN_FEATURES,
              required=score_b.REQUIRED_RAW,
              loader=score_b.load_bundle),
}


# ---------------------------------------------------------------- single applicant
@st.cache_resource
def load_scorecard_artifacts():
    model = joblib.load(os.path.join(ROOT, 'models/LogisticRegression1.joblib'))
    binning = joblib.load(os.path.join(ROOT, 'artifacts/binning_artifacts.pkl'))
    X_train_lr = joblib.load(os.path.join(ROOT, 'data/processed/logreg/X_train_lr.pkl'))
    feature_names = X_train_lr.columns.tolist()
    scorecard = build_scorecard(
        model=model, binning_artifacts=binning,
        feature_names=feature_names,
        config=ScorecardConfig(target_score=600, target_odds=50, pdo=20),
    )
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


def render_applicant():
    st.subheader('Score a single application')
    st.caption('Point-based scorecard (WoE logistic model) with an '
               'adverse-action breakdown for declined applicants.')

    model, binning, scorecard, threshold = load_scorecard_artifacts()
    cats = load_categories()

    with st.sidebar:
        st.header('Applicant details')
        st.caption('Leave optional bureau fields blank if unknown.')
        f = {}
        c1, c2 = st.columns(2)
        f['annual_inc'] = c1.number_input('Annual income', min_value=0.0, value=75000.0, step=1000.0)
        f['loan_amnt'] = c2.number_input('Requested loan amount', min_value=0.0, value=15000.0, step=500.0)
        f['dti'] = c1.number_input('Debt-to-income (%)', min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        f['revol_util'] = c2.number_input('Revolving utilization (%)', min_value=0.0, max_value=100.0,
                                          value=30.0, step=1.0)
        f['revol_bal'] = c1.number_input('Revolving balance', min_value=0.0, value=10000.0, step=500.0)
        f['inq_last_6mths'] = c2.number_input('Credit inquiries (6m)', min_value=0, max_value=20, value=1)
        f['delinq_2yrs'] = c1.number_input('Delinquencies (2y)', min_value=0, max_value=20, value=0)
        f['pub_rec'] = c2.number_input('Public records', min_value=0, max_value=20, value=0)
        f['collections_12_mths_ex_med'] = c1.number_input('Collections (12m)', min_value=0, max_value=20, value=0)
        f['open_acc'] = c2.number_input('Open accounts', min_value=0, max_value=100, value=10)
        f['total_acc'] = c1.number_input('Total accounts', min_value=0, max_value=200, value=20)
        f['term'] = c2.selectbox('Loan term', cats['term'])
        f['purpose'] = st.selectbox('Loan purpose', cats['purpose'])
        f['home_ownership'] = st.selectbox('Home ownership', cats['home_ownership'])
        f['verification_status'] = st.selectbox('Income verification', cats['verification_status'])
        f['emp_length'] = st.selectbox('Employment length', cats['emp_length'])
        f['earliest_cr_line'] = st.date_input('Earliest credit line',
                                              value=date(2010, 1, 1),
                                              min_value=date(1970, 1, 1), max_value=date.today())
        f['issue_d'] = st.date_input('Application date', value=date.today(),
                                     min_value=date(2007, 1, 1), max_value=date.today())

    row = {c: pd.NA for c in RAW_FEATURES}
    row.update(f)
    X = pd.DataFrame([row])
    X['issue_d'] = pd.to_datetime(X['issue_d'])
    X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'])

    Xw = transform_binning(X, binning, metric='woe')
    feat_cols = scorecard.feature_names
    p_bad = float(model.predict_proba(Xw[feat_cols])[:, 1][0])
    score = int(score_dataframe(scorecard, X).iloc[0])
    breakdown = explain_score(scorecard, X)

    if abs(p_bad - threshold) <= REVIEW_BAND:
        decision, color = 'MANUAL REVIEW', REVIEW_COLOR
    elif p_bad >= threshold:
        decision, color = 'DECLINE', REJECT_COLOR
    else:
        decision, color = 'APPROVE', APPROVE_COLOR

    m1, m2, m3 = st.columns(3)
    m1.metric('Credit score', score)
    m2.metric('Probability of default', f'{p_bad:.2%}')
    m3.metric('Recommended decision', decision)

    st.markdown(
        f"<div style='border-left:6px solid {color};padding:8px 16px;font-size:1.4rem;'>"
        f"<b style='color:{color}'>{decision}</b> — PD threshold "
        f"<b>{threshold:.3f}</b> (manual-review band ±{REVIEW_BAND:.0%})</div>",
        unsafe_allow_html=True)

    st.subheader('Why this decision')
    st.dataframe(breakdown, width="stretch", hide_index=True)

    worst = breakdown[breakdown['characteristic'] != 'Base Score'].nsmallest(3, 'point')
    st.markdown('**Main risk drivers** (largest point deductions): '
                + ', '.join(f"{r.characteristic} ({r.point} pts)" for r in worst.itertuples()))

    with st.expander('About this model'):
        manifest = json.load(open(os.path.join(ROOT, 'artifacts/manifest.json')))
        st.json({
            'model': 'WoE logistic regression (scorecard track)',
            'scorecard': 'PDO=20, target score 600 at 50:1 odds',
            'validation': {'roc_auc': 0.669, 'ks': 0.252},
            'oot_2013q1': {'roc_auc': 0.641, 'ks': 0.217},
            'decision_threshold_pd': threshold,
            'manifest_git_commit': manifest.get('git_commit'),
            'limitations': [
                'Application-time scope — post-origination fields excluded by design',
                'Trained on funded loans only; reject-inference bias not corrected',
                'For batch scoring and drift monitoring use the "Batch scoring" tab',
            ],
        })


# ---------------------------------------------------------------- batch scoring
SCORE_COL = {'New applications': 'pd_score_application',
             'Funded loans': 'pd_score_funded'}


def _score_batch(df, mode, scenario):
    """Returns (scored dict, psi scenario keys)."""
    if mode == 'lifecycle':
        try:
            out = score_lifecycle(df)
            return ({'New applications': out['scenario_a'],
                     'Funded loans': out['scenario_b']}, ['a', 'b'])
        except ValueError as e:
            st.warning(f'Lifecycle scoring incomplete ({e}); '
                       'showing new-application scores only.')
            cfg = SCENARIOS['a']
            return ({'New applications': score_records(
                df, cfg['loader'](), cfg['derive'], cfg['forbidden'],
                cfg['required'])}, ['a'])
    cfg = SCENARIOS[scenario]
    res = score_records(df, cfg['loader'](), cfg['derive'],
                        cfg['forbidden'], cfg['required'])
    return ({cfg['label']: res}, [scenario])


def render_batch():
    st.subheader('Score a batch of loans')
    st.caption('Upload a CSV of raw loan records. Columns are checked '
               'against the model contract — repayment and collection '
               'fields are never used as inputs.')

    uploaded = st.file_uploader('Loan data CSV', type='csv', key='batch_csv')
    if not uploaded:
        st.info('Upload a CSV to get started. One row per loan or '
                'application; column names must match the raw extract '
                '(e.g. `annual_inc`, `loan_amnt`, `dti`).')
        return

    mode = st.radio('What are you scoring?',
                    ['applications', 'funded', 'lifecycle'],
                    format_func=lambda m: {
                        'applications': 'New applications only',
                        'funded': 'Funded loans only',
                        'lifecycle': 'Both (one file, both views)'}[m],
                    key='batch_mode')
    scenario = {'applications': 'a', 'funded': 'b',
                'lifecycle': 'a'}[mode]
    if mode != 'lifecycle':
        scenario = st.selectbox('Model', ['a', 'b'],
                                format_func=lambda s: SCENARIOS[s]['label'],
                                key='batch_scenario')

    df = pd.read_csv(uploaded)
    st.write(f'Loaded **{len(df):,}** records × {df.shape[1]} columns')

    try:
        scored, psi_keys = _score_batch(df, mode, scenario)
    except ValueError as e:
        st.error(f'This file cannot be scored: {e}')
        return

    for name, res in scored.items():
        score_col = SCORE_COL.get(name, 'pd_score')
        dec_col = score_col.replace('pd_score', 'decision')
        st.markdown(f'#### {name}')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Loans scored', f'{len(res):,}')
        c2.metric('Average default probability', f"{res[score_col].mean():.2%}")
        c3.metric('Decline rate', f"{(res[dec_col] == 'REJECT').mean():.1%}")
        c4.metric('Highest-risk PD', f"{res[score_col].max():.1%}")
        st.dataframe(res.head(200), width="stretch", hide_index=True)

    # ---- drift panel ----
    st.markdown('#### Population stability (drift check)')
    st.caption('Compares this batch against the model’s validation '
               'reference. “Significant shift” means the applicant mix '
               'has changed enough that the model should be reviewed.')
    for s in psi_keys:
        cfg = SCENARIOS[s]
        try:
            rep = drift_report(df, cfg['loader'](), cfg['derive'],
                               cfg['forbidden'], cfg['required'],
                               cfg['interim'])
        except Exception as e:
            st.warning(f"Drift check unavailable ({cfg['label']}): {e}")
            continue
        icon = {'Stable': '🟢', 'Moderate shift -- investigate': '🟡',
                'Significant shift -- recalibration/redevelopment likely required': '🔴'}.get(
            rep['worst_verdict'], '⚪')
        st.markdown(f"**{cfg['label']}** {icon} — score PSI "
                    f"`{rep['score_psi']:.4f}` · **{rep['worst_verdict']}**")
        feats = rep['features']
        if len(feats):
            with st.expander('Top drifting features'):
                st.dataframe(feats.head(15), width="stretch",
                             hide_index=True)

    # ---- segment views ----
    st.markdown('#### Segment breakdown')
    seg_col = st.selectbox('Group by',
                           [c for c in df.columns if df[c].nunique() <= 20],
                           key='seg_col')
    label_col = next((c for c in ('target', 'loan_status')
                      if c in df.columns), None)
    for name, res in scored.items():
        score_col = SCORE_COL.get(name, 'pd_score')
        seg = df[[seg_col]].copy().join(
            res[[score_col]].reset_index(drop=True))
        if label_col:
            seg['bad'] = df[label_col].isin(
                [1, '1', 'Charged Off', 'Default', 'Charged_off']).astype(int)
            grp = seg.groupby(seg_col, observed=True).agg(
                n=('bad', 'size'), mean_pd=(score_col, 'mean'),
                bad_rate=('bad', 'mean')).reset_index()
            grp['gap'] = grp['mean_pd'] - grp['bad_rate']
            st.markdown(f'**{name}** — predicted vs realized default rate '
                        '(positive gap = model predicts higher risk than '
                        'observed)')
            st.dataframe(grp.sort_values('n', ascending=False),
                         width="stretch", hide_index=True)
        else:
            grp = seg.groupby(seg_col, observed=True)[score_col].agg(
                ['count', 'mean', 'min', 'max']).reset_index()
            st.dataframe(grp, width="stretch", hide_index=True)


# ---------------------------------------------------------------- shell
st.title('Credit Risk Dashboard')
st.caption('Probability-of-default models for lending decisions · '
           'application-time and funded-book views')

tab_app, tab_batch = st.tabs(['Single applicant', 'Batch scoring'])
with tab_app:
    render_applicant()
with tab_batch:
    render_batch()
