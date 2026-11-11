"""Shared engine for the Scenario-A / Scenario-B pipelines.

Both scenarios run the same nb2->nb4 shape; they differ only in feature
eligibility (spec modules under pipeline/scenario_*/) and artifact paths.
This module holds the generic prepare / train / score / PSI logic.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba  # noqa: E402
from src.metrics import calculate_probability_metrics  # noqa: E402
from src.performance_window import (  # noqa: E402
    apply_performance_window_filter,
    compute_time_to_bad_distribution,
    eligibility_funnel_report,
    recommend_performance_window,
    truncate_to_safe_horizon,
)
from src.thresholding import find_best_cost_threshold  # noqa: E402
from src.training import FIT_STRATEGIES, build_estimator  # noqa: E402
from src.utils import (  # noqa: E402
    importData,
    serialize_data,
    split_input_output,
    split_train_test,
)

DATA_PATH = 'data/raw/loan_data_2007_2014.csv'
SNAPSHOT_DATE = pd.Timestamp('2016-01-01')
OOT_CUTOFF = pd.Timestamp('2013-01-01')   # same Q1-2013 boundary as nb2
BAD_STATUSES = [
    'Charged Off', 'Default', 'Late (31-120 days)',
    'Does not meet the credit policy. Status:Charged Off',
]

RANDOM_STATE = 42
TRAIN_TEMP_SIZE = 0.15
ES_CAL_SHARE_WITHIN_TEMP = 0.50

# nb4 "best consistent" hyperparameters (same as autoresearch BASE_PARAMS)
BASE_PARAMS = {
    "cat": {
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "iterations": 5000, "early_stopping_rounds": 50,
        "bootstrap_type": "Bernoulli", "cat_features": "__AUTO__",
        "task_type": "CPU", "thread_count": -1, "border_count": 64,
        "learning_rate": 0.034042376046732846, "depth": 6,
        "grow_policy": "SymmetricTree", "l2_leaf_reg": 100.0,
        "subsample": 0.65, "random_strength": 1e-09,
        "min_data_in_leaf": 45, "rsm": 0.6,
        "random_seed": RANDOM_STATE, "verbose": False,
        "allow_writing_files": False,
    },
    "xgb": {
        "objective": "binary:logistic", "booster": "gbtree",
        "eval_metric": "logloss", "tree_method": "hist",
        "enable_categorical": True, "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "learning_rate": 0.01783339959316753, "max_depth": 8,
        "min_child_weight": 20.0, "gamma": 5.0, "subsample": 0.65,
        "colsample_bytree": 0.6286729841272466, "reg_alpha": 1e-08,
        "reg_lambda": 1e-08, "max_delta_step": 0, "max_bin": 507,
        "grow_policy": "depthwise", "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "lgbm": {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "n_estimators": 5000,
        "learning_rate": 0.014752303267203249,
        "early_stopping_rounds": 50, "feature_fraction": 0.6,
        "max_depth": 6, "num_leaves": 16, "min_child_samples": 72,
        "bagging_fraction": 0.65, "bagging_freq": 1,
        "reg_alpha": 0.46210107890864205, "reg_lambda": 1.087620542758583e-08,
        "min_split_gain": 5.524657351707396e-06, "max_bin": 512,
        "verbosity": -1, "n_jobs": -1, "random_state": RANDOM_STATE,
    },
}

ESTIMATOR_CLASSES = {
    "cat": "catboost.CatBoostClassifier",
    "xgb": "xgboost.XGBClassifier",
    "lgbm": "lightgbm.LGBMClassifier",
}


# ============================================================================
# PREPARE (nb2-equivalent)
# ============================================================================
def run_prepare(feature_columns, out_dir, data_path=DATA_PATH):
    """Raw CSV -> cleaned, performance-windowed, temporally split pickles.

    feature_columns: raw columns kept into X (target appended automatically).
    """
    df = importData(data_path)
    df = df[sorted(df.columns)].set_index('member_id')

    date_columns = ['issue_d', 'earliest_cr_line',
                    'last_pymnt_d', 'last_credit_pull_d']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%b-%y', errors='coerce')
        future_mask = df[col].dt.year > 2016
        df.loc[future_mask, col] = df.loc[future_mask, col] - pd.DateOffset(years=100)
    df = df.dropna(subset=date_columns)

    df = df.drop(columns=['tot_coll_amt', 'tot_cur_bal',
                          'total_rev_hi_lim'])
    df = df.dropna(subset='collections_12_mths_ex_med')
    df = df.dropna(subset='revol_util')

    df['home_ownership'] = df['home_ownership'].replace(
        {'ANY': 'Other', 'NONE': 'Other'})
    df['purpose'] = df['purpose'].replace({'vacation': 'other'})
    df['purpose'] = df['purpose'].fillna('other')

    for col in ('int_rate', 'revol_util'):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce')

    missing = [c for c in feature_columns + ['last_pymnt_d', 'loan_status']
               if c not in df.columns]
    if missing:
        raise ValueError(f'raw data missing required columns: {missing}')

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

    cols = list(feature_columns) + ['target']
    X, y = split_input_output(data=df_safe, target_col='target',
                              columns_to_keep=cols)

    is_oot = X['issue_d'] >= OOT_CUTOFF
    X_test, y_test = X[is_oot].copy(), y[is_oot].copy()
    X_hist = X[~is_oot].sort_values('issue_d')
    y_hist = y[~is_oot]
    n_valid = int(round(len(X_hist) * (10 / 90)))
    X_valid = X_hist.iloc[-n_valid:]
    X_train = X_hist.iloc[:-n_valid]
    y_valid = y_hist.loc[X_valid.index]   # label alignment (nb2 semantics)
    y_train = y_hist.loc[X_train.index]

    for name, frame in (('X_train', X_train), ('X_valid', X_valid),
                        ('X_test', X_test)):
        print(f'{name}: {len(frame)} rows | '
              f'{frame["issue_d"].min().date()} -> '
              f'{frame["issue_d"].max().date()}')

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, frame in (('X_train_raw', X_train), ('X_valid_raw', X_valid),
                        ('X_test_raw', X_test)):
        serialize_data(frame[sorted(frame.columns)],
                       str(out / f'{name}.pkl'))
    for name, series in (('y_train', y_train), ('y_valid', y_valid),
                         ('y_test', y_test)):
        serialize_data(series, str(out / f'{name}.pkl'))

    print(f'Wrote interim pickles to {out}/')
    print(f'y bad rates -> train {y_train.mean():.4f} | '
          f'valid {y_valid.mean():.4f} | test {y_test.mean():.4f}')
    return out


# ============================================================================
# TRAIN (nb4-equivalent)
# ============================================================================
def _union_all_categoricals(frames):
    """Union category levels of every category column across splits."""
    cat_cols = [c for c in frames['train'][0].columns
                if str(frames['train'][0][c].dtype) == 'category']
    for col in cat_cols:
        union = pd.Categorical(pd.concat(
            [frames[s][0][col] for s in ('train', 'valid', 'test')])
        ).categories
        for s in ('train', 'valid', 'test'):
            X = frames[s][0]
            X[col] = pd.Categorical(X[col], categories=union)
            frames[s] = (X, frames[s][1])
    return frames, cat_cols


def run_training(interim_dir, derive_fn, out_dir, report_stem,
                 scenario_label, report_dir='reports'):
    """Fit xgb/cat/lgbm, pick valid-AUC winner, calibrate, threshold, bundle.

    derive_fn: raw pickled frame -> model-feature frame.
    Returns the bundle dict.
    """
    from src.utils import deserialize_data

    t0 = time.time()
    interim = Path(interim_dir)
    frames = {}
    for split in ('train', 'valid', 'test'):
        X = deserialize_data(str(interim / f'X_{split}_raw.pkl'))
        y = deserialize_data(str(interim / f'y_{split}.pkl'))
        frames[split] = (derive_fn(X), y)

    frames, _ = _union_all_categoricals(frames)

    cols = sorted(frames['train'][0].columns)
    for split in frames:
        missing = set(cols) - set(frames[split][0].columns)
        if missing:
            raise ValueError(f'{split} missing columns: {missing}')
        frames[split] = (frames[split][0][cols], frames[split][1])
    feature_cols = cols

    cat_cols = [c for c in feature_cols
                if str(frames['train'][0][c].dtype) == 'category']
    print(f'{scenario_label}: {len(feature_cols)} features | '
          f'categorical: {cat_cols}')

    X_train, y_train = frames['train']
    X_fit, X_temp, y_fit, y_temp = split_train_test(
        X_train, y_train, test_size=TRAIN_TEMP_SIZE,
        stratify=y_train, seed=RANDOM_STATE)
    X_es, X_cal, y_es, y_cal = split_train_test(
        X_temp, y_temp, test_size=ES_CAL_SHARE_WITHIN_TEMP,
        stratify=y_temp, seed=RANDOM_STATE)

    import importlib
    fitted, leaderboard = {}, []
    for family, params in BASE_PARAMS.items():
        p = dict(params)
        if p.get('cat_features') == '__AUTO__':
            p['cat_features'] = cat_cols
        mod_name, cls_name = ESTIMATOR_CLASSES[family].split('.')
        est = getattr(importlib.import_module(mod_name), cls_name)(**p)
        est = FIT_STRATEGIES[family](est, X_fit, y_fit, X_es, y_es)
        fitted[family] = est
        for split_name, (X, y) in frames.items():
            proba = extract_positive_proba(est, X)
            leaderboard.append({'model': f'{family}_raw',
                                'split': split_name,
                                **calculate_probability_metrics(y, proba)})
        valid_auc = next(r['roc_auc'] for r in leaderboard
                         if r['model'] == f'{family}_raw'
                         and r['split'] == 'valid')
        print(f'{family}: valid AUC = {valid_auc:.4f}')

    lb = pd.DataFrame(leaderboard)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lb.to_csv(Path(report_dir) / f'{report_stem}_leaderboard.csv', index=False)

    valid_lb = lb[lb['split'] == 'valid'].sort_values(
        ['roc_auc', 'log_loss'], ascending=[False, True])
    winner = valid_lb.iloc[0]['model'].replace('_raw', '')
    print(f'Stage A winner: {winner} (valid AUC '
          f'{valid_lb.iloc[0]["roc_auc"]:.4f})')

    model = fitted[winner]
    p_cal_raw = extract_positive_proba(model, X_cal)
    p_valid_raw = extract_positive_proba(model, frames['valid'][0])
    y_valid = frames['valid'][1]

    candidates = {'raw': (None, p_valid_raw)}
    for kind in ('sigmoid', 'isotonic'):
        cal = _fit_calibrator(p_cal_raw, y_cal, kind)
        candidates[kind] = (cal, _apply_cal(cal, kind, p_valid_raw))
    cal_df = pd.DataFrame([
        {'version': k, **calculate_probability_metrics(y_valid, p)}
        for k, (_, p) in candidates.items()
    ]).sort_values('brier_score')
    cal_df.to_csv(Path(report_dir) / f'{report_stem}_calibration.csv',
                  index=False)
    best_version = cal_df.iloc[0]['version']
    calibrator, p_valid = candidates[best_version]
    print(f'calibration: {best_version} '
          f'(valid Brier {cal_df.iloc[0]["brier_score"]:.4f})')

    tau, cost = find_best_cost_threshold(y_valid, p_valid)
    approved = p_valid < tau
    appr_bad = float(np.mean(y_valid[approved])) if approved.any() else None

    X_valid_derived = frames['valid'][0]
    ref_edges, ref_probs = {}, {}
    for feat in feature_cols:
        s = X_valid_derived[feat]
        if str(s.dtype) == 'category':
            continue
        edges = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        binned = pd.cut(s, bins=edges, include_lowest=True)
        ref_edges[feat] = edges.tolist()
        ref_probs[feat] = (binned.value_counts(normalize=True, sort=False)
                           .reindex(range(len(edges) - 1),
                                    fill_value=0.0).tolist())

    bundle = {
        'scenario': scenario_label,
        'model_family': winner,
        'model': model,
        'calibrator_name': best_version if calibrator is not None else None,
        'calibrator': calibrator,
        'features': feature_cols,
        'cat_columns': cat_cols,
        'categories': {c: list(frames['train'][0][c].cat.categories)
                       for c in cat_cols},
        # CatBoost rejects NaN inside categorical columns; fill with the
        # training mode so unseen/missing levels degrade predictably
        'cat_fill': {c: (frames['train'][0][c].mode().iloc[0]
                         if len(frames['train'][0][c].mode()) else
                         frames['train'][0][c].cat.categories[0])
                     for c in cat_cols},
        'threshold': float(tau),
        'metrics': {
            'valid': calculate_probability_metrics(y_valid, p_valid),
            'oot_test': calculate_probability_metrics(
                frames['test'][1],
                _apply_cal(calibrator, best_version,
                           extract_positive_proba(model, frames['test'][0]))),
        },
        'approved_book_bad_rate_valid': appr_bad,
        'expected_cost_at_tau': float(cost),
        'reference_deciles': ref_edges,
        'reference_probs': ref_probs,
        'splits': {
            'oot_cutoff': '2013-01-01',
            'valid': 'latest ~11% of 2007-2012 history',
            'snapshot_date': '2016-01-01',
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    joblib.dump(bundle, out / f'{report_stem}_bundle.joblib')

    report = {
        'winner': winner, 'calibration': best_version,
        'threshold': float(tau), 'metrics': bundle['metrics'],
        'leaderboard_csv': str(Path(report_dir) / f'{report_stem}_leaderboard.csv'),
        'elapsed_s': bundle['elapsed_s'],
    }
    (Path(report_dir) / f'{report_stem}_metrics.json').write_text(
        json.dumps(report, indent=2))
    print(f'Saved {out}/{report_stem}_bundle.joblib')
    print(f'valid AUC {bundle["metrics"]["valid"]["roc_auc"]:.4f} | '
          f'OOT AUC {bundle["metrics"]["oot_test"]["roc_auc"]:.4f} | '
          f'tau {tau:.4f}')
    return bundle


def _fit_calibrator(p_cal, y_cal, kind):
    if kind == 'sigmoid':
        cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000,
                                 random_state=RANDOM_STATE)
        cal.fit(p_cal.reshape(-1, 1), y_cal)
        return cal
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(p_cal, y_cal)
    return cal


def _apply_cal(calibrator, name, p_raw):
    if calibrator is None or name in (None, 'raw'):
        return np.asarray(p_raw)
    if name == 'sigmoid':
        return calibrator.predict_proba(np.asarray(p_raw).reshape(-1, 1))[:, 1]
    if name == 'isotonic':
        return calibrator.predict(np.asarray(p_raw))
    raise ValueError(f'unknown calibrator {name!r}')


# ============================================================================
# SCORE / PSI (production inference)
# ============================================================================
def validate_payload(df, forbidden, required_raw):
    leaked = [c for c in forbidden if c in df.columns]
    if leaked:
        raise ValueError(
            f'Post-origination/leakage columns not allowed in scoring '
            f'payload: {leaked}')
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f'Scoring payload missing required columns: '
                         f'{missing}')
    return df


def prepare_scoring_frame(df, bundle, derive_fn):
    """Raw payload -> model-ready frame matching bundle['features']."""
    X = derive_fn(df)
    feats = bundle['features']
    missing = [c for c in feats if c not in X.columns]
    if missing:
        raise ValueError(f'Cannot derive required features: {missing}')
    X = X[feats].copy()
    levels = bundle.get('categories', {})
    fills = bundle.get('cat_fill', {})
    for col in bundle.get('cat_columns', []):
        if col in X.columns:
            s = X[col]
            if str(s.dtype) == 'category':
                s = s.astype(object)
            fill = fills.get(col)
            if fill is not None:
                s = s.where(s.notna(), fill)
            cats = levels.get(col)
            if cats:
                X[col] = pd.Categorical(s, categories=cats)
                # unseen levels became NaN codes; refill with train mode
                if fill is not None:
                    X[col] = X[col].fillna(fill)
            else:
                X[col] = s.astype('category')
    return X


def score_records(df, bundle, derive_fn, forbidden, required_raw,
                  id_col='member_id'):
    """Score a batch. Returns DataFrame with pd_score + decision."""
    validate_payload(df, forbidden, required_raw)
    X = prepare_scoring_frame(df, bundle, derive_fn)
    p_raw = bundle['model'].predict_proba(X)[:, 1]
    p = _apply_cal(bundle.get('calibrator'),
                   bundle.get('calibrator_name'), p_raw)
    out = pd.DataFrame(index=df.index)
    if id_col in df.columns:
        out[id_col] = df[id_col]
    out['pd_score'] = np.round(p, 6)
    out['decision'] = np.where(p >= bundle['threshold'],
                               'REJECT', 'APPROVE')
    return out


def psi_report(df, bundle, derive_fn, forbidden, required_raw, n_bins=10):
    """Per-feature PSI of a new batch vs the stored validation reference."""
    from src.population_stability_index import psi_verdict
    validate_payload(df, forbidden, required_raw)
    ref = bundle.get('reference_deciles')
    if not ref:
        return pd.DataFrame(columns=['feature', 'psi', 'verdict'])
    X = derive_fn(df)
    rows = []
    for feat, edges in ref.items():
        if feat not in X.columns:
            continue
        edges = np.asarray(edges, dtype=float)
        s_new = pd.cut(pd.to_numeric(X[feat], errors='coerce'),
                       bins=edges, include_lowest=True)
        ref_probs = np.asarray(bundle['reference_probs'][feat])
        new_counts = s_new.value_counts(normalize=True, sort=False).reindex(
            range(len(ref_probs)), fill_value=0.0).to_numpy()
        psi = float(np.sum(
            (new_counts - ref_probs)
            * np.log((new_counts + 1e-4) / (ref_probs + 1e-4))))
        rows.append({'feature': feat, 'psi': round(psi, 4),
                     'verdict': psi_verdict(psi)})
    rep = pd.DataFrame(rows).sort_values('psi', ascending=False)
    rep.loc['TOTAL'] = ['ALL', round(rep['psi'].sum(), 4), '']
    return rep
