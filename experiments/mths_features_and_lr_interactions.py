"""Experiment: do mths_since_last_* bureau features and LR interaction terms
add predictive value? Mirrors nb3/nb4 pipelines exactly; decision split = 'valid'.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42
MTHS = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']
NEVER = 999.0  # sentinel: feature never observed on bureau record

def L(p): return joblib.load(f'{ROOT}/{p}')
def ks_stat(y, p):
    order = np.argsort(-p); y_s = np.asarray(y)[order]; p_s = np.asarray(p)[order]
    P = y_s.sum(); N = len(y_s) - P
    tpr = np.cumsum(y_s) / P; fpr = np.cumsum(1 - y_s) / N
    return float(np.max(np.abs(tpr - fpr)))

def metrics(y, p):
    return dict(auc=roc_auc_score(y, p), ks=ks_stat(y, p),
                logloss=log_loss(y, p), brier=brier_score_loss(y, p))

t0 = time.time()
print('=== loading data ===')
mths = pd.read_csv(f'{ROOT}/data/raw/loan_data_2007_2014.csv',
                   usecols=['member_id'] + MTHS).set_index('member_id')

Xtr, ytr = L('data/interim/X_train_raw.pkl'), L('data/interim/y_train.pkl')
Xv,  yv  = L('data/interim/X_valid_raw.pkl'),  L('data/interim/y_valid.pkl')
Xt,  yt  = L('data/interim/X_test_raw.pkl'),   L('data/interim/y_test.pkl')

def add_mths(X):
    return pd.concat([X, mths[MTHS].reindex(X.index).fillna(NEVER)], axis=1)

def add_interactions(X):
    out = X.copy()
    out['dti_x_revol_util'] = out['dti'] * out['revol_util']
    lti = out['loan_amnt'] / out['annual_inc']
    cage = ((out['issue_d'].dt.year - out['earliest_cr_line'].dt.year) * 12
            + (out['issue_d'].dt.month - out['earliest_cr_line'].dt.month)) / 12.0
    out['lti_x_credit_age'] = lti * cage
    return out

# ---------------- PART A: TREE MODELS ----------------
print('=== PART A: tree models ===')
from src.utils import split_train_test
from src.binning_woe import add_derived_features

X_fit, X_tmp, y_fit, y_tmp = split_train_test(Xtr, ytr, test_size=0.15,
                                            stratify=ytr, seed=RANDOM_STATE)
X_es, X_cal, y_es, y_cal = split_train_test(X_tmp, y_tmp, test_size=0.5,
                                            stratify=y_tmp, seed=RANDOM_STATE)

raw_pieces = {'fit': X_fit, 'es': X_es, 'cal': X_cal, 'valid': Xv, 'test': Xt}
variants = {'base': {}, 'mths': {c: add_mths for c in [None]},
            }
# build variant frames
def build(feats_fn):
    d = {}
    for k, X in raw_pieces.items():
        X2 = feats_fn(X) if feats_fn else X.copy()
        X2 = add_derived_features(X2)
        d[k] = X2[sorted(X2.columns)]
    return d

D_base = build(None)
D_mths = build(add_mths)

def cat_features(Xf):
    return [c for c in Xf.columns if str(Xf[c].dtype) == 'category']

XGB_PARAMS = {
    'objective': 'binary:logistic', 'booster': 'gbtree', 'eval_metric': 'logloss',
    'tree_method': 'hist', 'enable_categorical': True, 'n_estimators': 5000,
    'early_stopping_rounds': 50, 'learning_rate': 0.01783339959316753,
    'max_depth': 8, 'min_child_weight': 20.0, 'gamma': 5.0, 'subsample': 0.65,
    'colsample_bytree': 0.6286729841272466, 'reg_alpha': 1e-08, 'reg_lambda': 1e-08,
    'max_delta_step': 0, 'max_bin': 507, 'grow_policy': 'depthwise',
    'n_jobs': -1, 'random_state': RANDOM_STATE,
}
CAT_PARAMS = {
    'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'iterations': 5000,
    'early_stopping_rounds': 50, 'bootstrap_type': 'Bernoulli',
    'task_type': 'CPU', 'thread_count': -1, 'border_count': 64,
    'learning_rate': 0.034042376046732846, 'depth': 6,
    'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 100.0, 'subsample': 0.65,
    'random_strength': 1e-09, 'min_data_in_leaf': 45, 'rsm': 0.6,
    'random_seed': RANDOM_STATE, 'verbose': False, 'allow_writing_files': False,
}

def fit_tree(kind, D):
    Xf, Xe, Xva, Xte = D['fit'], D['es'], D['valid'], D['test']
    yf, ye = y_fit, y_es
    if kind == 'cat':
        from catboost import CatBoostClassifier
        p = dict(CAT_PARAMS); p['cat_features'] = cat_features(Xf)
        m = CatBoostClassifier(**p)
        m.fit(Xf, yf, eval_set=(Xe, ye))
    else:
        from xgboost import XGBClassifier
        p = dict(XGB_PARAMS)
        m = XGBClassifier(**p)
        m.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
    return m

def eval_tree(m, D):
    out = {}
    for split, yv_ in [('valid', yv), ('test', yt)]:
        p = m.predict_proba(D[split])[:, 1]
        out[split] = metrics(np.asarray(yv_), p)
    return out

results = {}
models = {}
for name, D in [('base', D_base), ('mths', D_mths)]:
    for kind in ['cat', 'xgb']:
        t = time.time()
        m = fit_tree(kind, D)
        models[(name, kind)] = m
        r = eval_tree(m, D)
        results[f'{kind}_{name}'] = r
        print(f'{kind}_{name}: valid AUC={r["valid"]["auc"]:.5f} KS={r["valid"]["ks"]:.4f} '
              f'| test AUC={r["test"]["auc"]:.5f} KS={r["test"]["ks"]:.4f} ({time.time()-t:.0f}s)')

# paired bootstrap on valid: cat_mths - cat_base
rng = np.random.default_rng(0)
p_b = models[('base','cat')].predict_proba(D_base['valid'])[:,1]
p_m = models[('mths','cat')].predict_proba(D_mths['valid'])[:,1]
yv_arr = np.asarray(yv)
diffs = []
for _ in range(1000):
    idx = rng.integers(0, len(yv_arr), len(yv_arr))
    diffs.append(roc_auc_score(yv_arr[idx], p_m[idx]) - roc_auc_score(yv_arr[idx], p_b[idx]))
ci = np.percentile(diffs, [2.5, 97.5])
print(f'cat mths-vs-base valid AUC diff: mean={np.mean(diffs):+.5f} 95%CI=[{ci[0]:+.5f},{ci[1]:+.5f}]')

# ---------------- PART B: LOGISTIC REGRESSION ----------------
print('=== PART B: logistic regression (WoE pipeline) ===')
import src.binning_woe as bw

Xtr_y = Xtr.assign(y=ytr)

def lr_pipeline(tag, extra_num=(), fit_params_extra=None, Xtr_ext=None,
                Xv_ext=None, Xt_ext=None):
    """Fit binning on extended feature set, transform splits, train LR, eval."""
    saved = (bw.ALGORITHMIC_NUMERICAL.copy(), bw.ALGORITHMIC_FEATURES.copy(),
             dict(bw.ALGORITHMIC_FIT_PARAMS))
    try:
        bw.ALGORITHMIC_NUMERICAL = bw.ALGORITHMIC_NUMERICAL + list(extra_num)
        bw.ALGORITHMIC_FEATURES = bw.ALGORITHMIC_NUMERICAL + bw.ALGORITHMIC_CATEGORICAL
        if fit_params_extra:
            bw.ALGORITHMIC_FIT_PARAMS.update(fit_params_extra)
        tr = Xtr_ext if Xtr_ext is not None else Xtr_y
        art = bw.fit_binning(tr, target_col='y')
        Xl_tr = bw.transform_binning(Xtr_ext.drop(columns=['y'], errors='ignore') if Xtr_ext is not None else Xtr, art, metric='woe')
        Xl_v  = bw.transform_binning(Xv_ext if Xv_ext is not None else Xv, art, metric='woe')
        Xl_t  = bw.transform_binning(Xt_ext if Xt_ext is not None else Xt, art, metric='woe')
    finally:
        bw.ALGORITHMIC_NUMERICAL, bw.ALGORITHMIC_FEATURES, bw.ALGORITHMIC_FIT_PARAMS = saved

    # nb3 split: hold out 10/90 for calibration
    Xa, Xc, ya, yc = split_train_test(Xl_tr, ytr, test_size=10/90,
                                    stratify=ytr, seed=RANDOM_STATE)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=0.496024651248498, l1_ratio=0.0, max_iter=2000,
                            penalty='elasticnet', random_state=RANDOM_STATE,
                            solver='saga')
    lr.fit(Xa, ya)
    out = {}
    for name, Xl, y_ in [('valid', Xl_v, yv), ('test', Xl_t, yt)]:
        p = lr.predict_proba(Xl)[:, 1]
        out[name] = metrics(np.asarray(y_), p)
    print(f'lr_{tag}: valid AUC={out["valid"]["auc"]:.5f} KS={out["valid"]["ks"]:.4f} '
          f'| test AUC={out["test"]["auc"]:.5f} KS={out["test"]["ks"]:.4f} '
          f'| nfeat={Xl_tr.shape[1]} ({time.time()-t0:.0f}s elapsed)')
    return out, lr, Xl_v

# baseline LR: reuse existing pickles (already fitted pipeline)
Xl_tr0 = L('data/processed/logreg/X_train_lr.pkl')
Xa, Xc, ya, yc = split_train_test(Xl_tr0, ytr, test_size=10/90,
                                stratify=ytr, seed=RANDOM_STATE)
from sklearn.linear_model import LogisticRegression
lr0 = LogisticRegression(C=0.496024651248498, l1_ratio=0.0, max_iter=2000,
                         penalty='elasticnet', random_state=RANDOM_STATE, solver='saga')
lr0.fit(Xa, ya)
for name, p in [('valid', L('data/processed/logreg/X_valid_lr.pkl')),
                ('test',  L('data/processed/logreg/X_test_lr.pkl'))]:
    y_ = yv if name == 'valid' else yt
    m = metrics(np.asarray(y_), lr0.predict_proba(p)[:, 1])
    print(f'lr_base: {name} AUC={m["auc"]:.5f} KS={m["ks"]:.4f}')

# + mths features
mths_params = {c: dict(max_n_bins=8, min_bin_size=0.05, monotonic_trend='descending')
               for c in MTHS}
lr_pipeline('mths', extra_num=MTHS, fit_params_extra=mths_params,
            Xtr_ext=add_mths(Xtr).assign(y=ytr),
            Xv_ext=add_mths(Xv),
            Xt_ext=add_mths(Xt))

# + interactions
inter_params = {
    'dti_x_revol_util': dict(max_n_bins=8, min_bin_size=0.05, monotonic_trend='auto'),
    'lti_x_credit_age': dict(max_n_bins=8, min_bin_size=0.05, monotonic_trend='auto'),
}
lr_pipeline('interactions', extra_num=['dti_x_revol_util', 'lti_x_credit_age'],
            fit_params_extra=inter_params,
            Xtr_ext=Xtr.assign(y=ytr).pipe(add_interactions),
            Xv_ext=add_interactions(Xv), Xt_ext=add_interactions(Xt))

print(f'=== done in {time.time()-t0:.0f}s ===')
