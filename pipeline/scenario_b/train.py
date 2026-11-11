"""Scenario B model training — .py equivalent of nb4 (tree-based) stage.

Fits xgb/cat/lgbm raw candidates on the Scenario-B interim pickles,
evaluates on train/valid/OOT-test once (canonical run, not a search loop),
selects the Stage-A winner on temporal validation AUC, applies the better
of sigmoid/isotonic calibration (chosen on validation Brier), tunes the
expected-cost threshold on validation, and saves a production bundle.

Usage:
    python -m pipeline.scenario_b.train
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba  # noqa: E402
from src.metrics import calculate_probability_metrics  # noqa: E402
from src.thresholding import find_best_cost_threshold  # noqa: E402
from src.training import FIT_STRATEGIES, build_estimator  # noqa: E402
from src.utils import deserialize_data  # noqa: E402

from .features import add_scenario_b_features, apply_calibrator  # noqa: E402

INTERIM = Path('data/interim/scenario_b')
OUT_DIR = Path('models/scenario_b')
REPORT = Path('reports/scenario_b_metrics.json')

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


def load_frames():
    frames = {}
    for split in ('train', 'valid', 'test'):
        X = deserialize_data(str(INTERIM / f'X_{split}_raw.pkl'))
        y = deserialize_data(str(INTERIM / f'y_{split}.pkl'))
        frames[split] = (add_scenario_b_features(X), y)

    # union categorical levels across ALL splits — derived categoricals
    # (has_derogatory etc.) get per-frame levels otherwise, scrambling
    # category codes on valid/test
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
    return frames


def align_columns(frames):
    """All splits must expose identical feature columns in identical order."""
    cols = sorted(frames['train'][0].columns)
    for split, (X, _) in frames.items():
        missing = set(cols) - set(X.columns)
        if missing:
            raise ValueError(f'{split} missing columns: {missing}')
        frames[split] = (X[cols], frames[split][1])
    return frames, cols


def lb_valid_auc(rows, family):
    return next(r['roc_auc'] for r in rows
                if r['model'] == f'{family}_raw' and r['split'] == 'valid')


def fit_es_cal_split(X_train, y_train):
    """nb4 Stage A: fit / early-stop / calibration thirds of X_train."""
    from src.utils import split_train_test
    X_fit, X_temp, y_fit, y_temp = split_train_test(
        X_train, y_train, test_size=TRAIN_TEMP_SIZE,
        stratify=y_train, seed=RANDOM_STATE)
    X_es, X_cal, y_es, y_cal = split_train_test(
        X_temp, y_temp, test_size=ES_CAL_SHARE_WITHIN_TEMP,
        stratify=y_temp, seed=RANDOM_STATE)
    return X_fit, y_fit, X_es, y_es, X_cal, y_cal


def fit_calibrator(p_cal: np.ndarray, y_cal: pd.Series, kind: str):
    if kind == 'sigmoid':
        cal = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000,
                                 random_state=RANDOM_STATE)
        cal.fit(p_cal.reshape(-1, 1), y_cal)
        return cal, lambda p: cal.predict_proba(p.reshape(-1, 1))[:, 1]
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(p_cal, y_cal)
    return cal, lambda p: cal.predict(p)


def main() -> None:
    t0 = time.time()
    frames, feature_cols = align_columns(load_frames())
    cat_cols = [c for c in feature_cols
                if str(frames['train'][0][c].dtype) == 'category']
    print(f'features: {len(feature_cols)} | categorical: {cat_cols}')

    X_fit, y_fit, X_es, y_es, X_cal, y_cal = fit_es_cal_split(
        *frames['train'])

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
            m = calculate_probability_metrics(y, proba)
            leaderboard.append({'model': f'{family}_raw',
                                'split': split_name, **m})
        valid_auc = lb_valid_auc(leaderboard, family)
        print(f'{family}: valid AUC = {valid_auc:.4f}')

    lb = pd.DataFrame(leaderboard)
    lb.to_csv(REPORT.with_name('scenario_b_leaderboard.csv'), index=False)

    # Stage A winner: highest valid AUC (ties -> lowest valid logloss)
    valid_lb = lb[lb['split'] == 'valid'].sort_values(
        ['roc_auc', 'log_loss'], ascending=[False, True])
    winner = valid_lb.iloc[0]['model'].replace('_raw', '')
    print(f'Stage A winner: {winner} (valid AUC '
          f'{valid_lb.iloc[0]["roc_auc"]:.4f})')

    model = fitted[winner]
    p_cal_raw = extract_positive_proba(model, X_cal)

    # calibration choice on validation Brier (raw vs sigmoid vs isotonic)
    p_valid_raw = extract_positive_proba(model, frames['valid'][0])
    y_valid = frames['valid'][1]
    candidates = {'raw': (None, p_valid_raw)}
    cal_leader = []
    for kind in ('sigmoid', 'isotonic'):
        cal, fn = fit_calibrator(p_cal_raw, y_cal, kind)
        candidates[kind] = (cal, fn(p_valid_raw))
    for kind, (_, p) in candidates.items():
        cal_leader.append({'version': kind,
                           **calculate_probability_metrics(y_valid, p)})
    cal_df = pd.DataFrame(cal_leader).sort_values('brier_score')
    cal_df.to_csv(REPORT.with_name('scenario_b_calibration.csv'),
                  index=False)
    best_version = cal_df.iloc[0]['version']
    calibrator, p_valid = candidates[best_version]
    print(f'calibration: {best_version} '
          f'(valid Brier {cal_df.iloc[0]["brier_score"]:.4f})')

    # expected-cost threshold tuned on validation, chosen prob type
    tau, cost = find_best_cost_threshold(y_valid, p_valid)
    approved = p_valid < tau
    appr_bad = float(np.mean(y_valid[approved])) if approved.any() else None

    # drift-monitoring reference: decile edges + per-bin shares on the
    # temporal validation split (numeric features only)
    X_valid_derived = frames['valid'][0]
    ref_edges, ref_probs = {}, {}
    for feat in feature_cols:
        s = X_valid_derived[feat]
        if str(s.dtype) == 'category':
            continue
        edges = np.unique(np.quantile(s.dropna(),
                                      np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        binned = pd.cut(s, bins=edges, include_lowest=True)
        ref_edges[feat] = edges.tolist()
        ref_probs[feat] = (binned.value_counts(normalize=True, sort=False)
                           .reindex(range(len(edges) - 1),
                                    fill_value=0.0).tolist())

    # production bundle
    bundle = {
        'scenario': 'B',
        'model_family': winner,
        'model': model,
        'calibrator_name': best_version if calibrator is not None else None,
        'calibrator': calibrator,
        'features': feature_cols,
        'cat_columns': cat_cols,
        'categories': {c: list(frames['train'][0][c].cat.categories)
                       for c in cat_cols},
        'threshold': float(tau),
        'metrics': {
            'valid': calculate_probability_metrics(y_valid, p_valid),
            'oot_test': calculate_probability_metrics(
                frames['test'][1],
                apply_calibrator(calibrator, best_version,
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(bundle, OUT_DIR / 'scenario_b_bundle.joblib')
    REPORT.write_text(json.dumps({
        'winner': winner, 'calibration': best_version,
        'threshold': float(tau), 'metrics': bundle['metrics'],
        'leaderboard_csv': str(REPORT.with_name('scenario_b_leaderboard.csv')),
        'elapsed_s': bundle['elapsed_s'],
    }, indent=2))
    print(f'Saved {OUT_DIR}/scenario_b_bundle.joblib + {REPORT}')
    print(f'valid AUC {bundle["metrics"]["valid"]["roc_auc"]:.4f} | '
          f'OOT AUC {bundle["metrics"]["oot_test"]["roc_auc"]:.4f} | '
          f'tau {tau:.4f}')


if __name__ == '__main__':
    main()
