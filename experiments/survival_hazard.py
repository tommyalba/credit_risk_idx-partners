"""Discrete-time hazard model (v2 experiment) — handles censoring explicitly.

Binary target says "BAD within M=33 months?". The hazard model instead learns
h(t|x) = P(default in month t | survived to t), giving PD(t) curves and a
PD(33) comparable to the binary model.

Event/censoring definitions (consistent with src/performance_window.py):
  - BAD loans      : event at months(issue_d -> last_pymnt_d)   [project proxy]
  - Fully Paid     : censored at months(issue_d -> last_pymnt_d) (payoff ~ last pmt)
  - Current        : censored at months(issue_d -> snapshot=2015-09-01)
All durations capped at M=33 to match the binary performance window.

Features: same 12 derived features as the tree models (numeric + one-hot
categoricals) + smooth period terms (t, t^2, log t) for the baseline hazard.

Run from repo root:  python experiments/survival_hazard.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.binning_woe import add_derived_features
from src.performance_window import DEFAULT_BAD_STATUSES, DEFAULT_RESOLVED_GOOD_STATUSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 33
SNAPSHOT = pd.Timestamp('2015-09-01')

def months_between(a, b):
    return ((b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)).clip(lower=0)

def load_split(split):
    X = joblib.load(f'{ROOT}/data/interim/X_{split}_raw.pkl')
    y = np.asarray(joblib.load(f'{ROOT}/data/interim/y_{split}.pkl'))
    return X, y

def durations_for(X_raw, y):
    """Attach loan_status + dates from raw CSV, return T (months at risk, 1..M)
    and event indicator per loan."""
    raw = pd.read_csv(f'{ROOT}/data/raw/loan_data_2007_2014.csv',
                      usecols=['member_id', 'loan_status', 'issue_d', 'last_pymnt_d'])
    raw['issue_d'] = pd.to_datetime(raw['issue_d'], format='%b-%y', errors='coerce')
    raw['last_pymnt_d'] = pd.to_datetime(raw['last_pymnt_d'], format='%b-%y', errors='coerce')
    raw = raw.set_index('member_id')
    df = X_raw[['issue_d']].join(raw[['loan_status', 'last_pymnt_d']])
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'])
    df['y'] = y

    is_bad = df['loan_status'].isin(DEFAULT_BAD_STATUSES)
    is_paid = df['loan_status'].isin(DEFAULT_RESOLVED_GOOD_STATUSES)
    # remaining = 'Current' and other open statuses -> censored at snapshot

    t_bad = months_between(df['issue_d'], df['last_pymnt_d'])
    t_paid = months_between(df['issue_d'], df['last_pymnt_d'])
    t_open = months_between(df['issue_d'], pd.Series(SNAPSHOT, index=df.index))

    T = pd.Series(np.select(
        [is_bad.to_numpy(), is_paid.to_numpy()],
        [t_bad.fillna(M).to_numpy(), t_paid.fillna(M).to_numpy()],
        default=t_open.to_numpy()), index=df.index)
    T = np.clip(np.ceil(T.fillna(M).astype(float)), 1, M).astype(int)
    event = (is_bad & (T <= M)).astype(int).to_numpy()
    # a bad loan whose proxy time exceeds M is treated as censored at M (no event in window)
    event = np.where(is_bad.to_numpy() & (t_bad.fillna(999) <= M).to_numpy(), 1, 0)
    return pd.Series(T, index=df.index), pd.Series(event, index=df.index)

def person_period(X_feat, T, event):
    """Expand to (loan, period) rows. Returns design matrix + duration/event."""
    n = len(X_feat)
    counts = T.to_numpy()
    idx = np.repeat(np.arange(n), counts)
    period = np.concatenate([np.arange(1, c + 1) for c in counts])
    feats = X_feat.iloc[idx].reset_index(drop=True)
    feats['period'] = period
    feats['period2'] = period ** 2
    feats['log_period'] = np.log(period)
    # event only in the loan's final period
    ev = np.zeros(len(idx), dtype=np.int8)
    ends = np.cumsum(counts) - 1
    ev[ends[event.to_numpy() == 1]] = 1
    return feats, ev

def fit_hazard(Xtr_pp, y_pp):
    lr = LogisticRegression(C=0.5, max_iter=300, solver='lbfgs')
    lr.fit(Xtr_pp, y_pp)
    return lr

def pd_within_M(model, X_loan, T_max=M):
    """PD(event within M months) per loan via cumulative hazard product."""
    rows = []
    n = len(X_loan)
    for t in range(1, T_max + 1):
        Xt = X_loan.copy()
        Xt['period'] = t; Xt['period2'] = t * t; Xt['log_period'] = np.log(t)
        h = model.predict_proba(Xt[model.feature_names_in_])[:, 1]
        rows.append(h)
    H = np.vstack(rows).T                      # n_loans x M hazards
    return 1.0 - np.prod(1.0 - H, axis=1)

def main():
    t0 = time.time()
    print('=== loading splits ===')
    data = {}
    for s in ['train', 'valid', 'test']:
        X, y = load_split(s)
        data[s] = (X, y)
        print(s, X.shape, 'bad', round(y.mean(), 3))

    print('=== durations ===')
    for s, (X, y) in data.items():
        T, ev = durations_for(X, y)
        data[s] = (X, y, T, ev)
        print(f'{s}: mean T={T.mean():.1f}mo, events={ev.sum()} ({ev.mean():.3f}), '
              f'max T={T.max()}')

    print('=== features (WoE transform, same as LR pipeline) ===')
    from src.binning_woe import transform_binning
    artifacts = joblib.load(f'{ROOT}/artifacts/binning_artifacts.pkl')
    feats = {}
    for s, (X, y, T, ev) in data.items():
        Xw = transform_binning(X, artifacts, metric='woe')
        feats[s] = Xw.astype(np.float32)
        print(s, Xw.shape)

    # align columns to train
    fcols = feats['train'].columns
    for s in ['valid', 'test']:
        feats[s] = feats[s].reindex(columns=fcols, fill_value=0.0)

    print('=== person-period expansion (train) ===')
    Xtr, ytr, Ttr, evtr = data['train']
    pp_X, pp_y = person_period(feats['train'], Ttr, evtr)
    print('train person-period:', pp_X.shape, 'events:', pp_y.sum(),
          f'({time.time()-t0:.0f}s)')

    print('=== fitting discrete-time hazard LR ===')
    model = fit_hazard(pp_X, pp_y)
    print(f'fit done ({time.time()-t0:.0f}s)')

    def ks(y, p):
        o = np.argsort(-p); ys = np.asarray(y)[o]
        P, N = ys.sum(), len(ys) - ys.sum()
        return float(np.max(np.abs(np.cumsum(ys)/P - np.cumsum(1-ys)/N)))

    print('\n=== PD(33mo) vs binary labels ===')
    for s in ['valid', 'test']:
        Xs, ys, Ts, evs = data[s]
        pd_m = pd_within_M(model, feats[s])
        auc = roc_auc_score(ys, pd_m)
        ll = log_loss(ys, np.clip(pd_m, 1e-6, 1-1e-6))
        br = brier_score_loss(ys, pd_m)
        print(f'{s}: PD33 AUC={auc:.5f} KS={ks(ys, pd_m):.4f} '
              f'logloss={ll:.4f} brier={br:.4f} | mean PD={pd_m.mean():.3f} vs bad={ys.mean():.3f}')

    # calibration-ish: decile table on test
    Xs, ys, Ts, evs = data['test']
    pd_m = pd_within_M(model, feats['test'])
    dec = pd.qcut(pd_m, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'dec': dec, 'y': ys, 'pd': pd_m}).groupby('dec').agg(
        emp=('y', 'mean'), pred=('pd', 'mean'), n=('y', 'size'))
    print('\ntest calibration deciles:'); print(cal.round(3).to_string())

if __name__ == '__main__':
    main()
