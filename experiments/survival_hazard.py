"""Discrete-time hazard model (v2 experiment) — handles censoring explicitly.

Binary target says "BAD within M=33 months?". The hazard model instead learns
h(t|x) = P(default in month t | survived to t), giving PD(t) curves and a
PD(33) comparable to the binary model.

Event/censoring definitions (consistent with src/performance_window.py):
  - BAD loans      : event at months(issue_d -> last_pymnt_d)   [project proxy]
  - Fully Paid     : censored at months(issue_d -> last_pymnt_d) (payoff ~ last pmt)
  - Current        : censored at months(issue_d -> snapshot=2016-01-01)
All durations capped at M=33 to match the binary performance window.

Features: same 12 derived features as the tree models (numeric + one-hot
categoricals) + smooth period terms (t, t^2, log t) for the baseline hazard.

Run from repo root:  python experiments/survival_hazard.py
"""
import sys, os, time, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.binning_woe import add_derived_features
from src.performance_window import DEFAULT_BAD_STATUSES, DEFAULT_RESOLVED_GOOD_STATUSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = 33
SNAPSHOT = pd.Timestamp('2016-01-01')  # canonical pipeline snapshot

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

def fit_hazard(Xtr_pp, y_pp, learner='lr'):
    if learner == 'lr':
        model = LogisticRegression(C=0.5, max_iter=300, solver='lbfgs')
    elif learner == 'hgb':
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=42)
    else:
        raise ValueError(f'unknown learner {learner!r}')
    model.fit(Xtr_pp, y_pp)
    return model

def hazard_matrix(model, X_loan, T_max=M):
    """Monthly hazard h(t|x) for t=1..T_max. Returns n_loans x T_max."""
    rows = []
    for t in range(1, T_max + 1):
        Xt = X_loan.copy()
        Xt['period'] = t; Xt['period2'] = t * t; Xt['log_period'] = np.log(t)
        rows.append(model.predict_proba(Xt[model.feature_names_in_])[:, 1])
    return np.vstack(rows).T                      # n_loans x M hazards

def cum_pd(H, t):
    """Cumulative PD(t) = 1 - prod_{s<=t}(1 - h_s) for each loan."""
    return 1.0 - np.prod(1.0 - H[:, :t], axis=1)

def pd_within_M(model, X_loan, T_max=M):
    """PD(event within M months) per loan via cumulative hazard product."""
    return cum_pd(hazard_matrix(model, X_loan, T_max), T_max)

def median_event_time(H, T_max=M):
    """Predicted median time-to-event per loan: first t with S(t) <= 0.5,
    else T_max + 1 (survives past window)."""
    S = np.cumprod(1.0 - H, axis=1)
    hit = S <= 0.5
    first = np.where(hit.any(axis=1),
                     hit.argmax(axis=1) + 1, T_max + 1)
    return first

def metrics_row(y, p, name):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        'split': name,
        'n': len(y),
        'bad_rate': float(np.mean(y)),
        'mean_pd': float(np.mean(p)),
        'auc': float(roc_auc_score(y, p)),
        'logloss': float(log_loss(y, p)),
        'brier': float(brier_score_loss(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--learner', choices=['lr', 'hgb'], default='lr')
    args = ap.parse_args()

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

    print('=== fitting discrete-time hazard model ===')
    model = fit_hazard(pp_X, pp_y, learner=args.learner)
    print(f'learner={args.learner} fit done ({time.time()-t0:.0f}s)')

    def ks(y, p):
        o = np.argsort(-p); ys = np.asarray(y)[o]
        P, N = ys.sum(), len(ys) - ys.sum()
        return float(np.max(np.abs(np.cumsum(ys)/P - np.cumsum(1-ys)/N)))

    # --- PD(33) per split, raw + isotonic-calibrated on a train tail ---
    # Calibration fit on the LAST 15% of train (temporally adjacent to
    # valid), mirroring the pipeline's temporal discipline; valid/test
    # stay untouched until scoring.
    # Hazard matrix computed once per split; all horizon summaries below
    # derive from it.
    H = {s: hazard_matrix(model, feats[s]) for s in ('train', 'valid', 'test')}
    pd33 = {s: cum_pd(H[s], M) for s in ('train', 'valid', 'test')}

    n_tr = len(feats['train'])
    cal_n = int(round(n_tr * 0.15))
    # train frame is sorted by issue_d upstream; tail = most recent
    cal_mask = np.zeros(n_tr, dtype=bool)
    cal_mask[-cal_n:] = True
    y_tr_all = data['train'][1]
    y_cal = np.asarray(y_tr_all)[cal_mask]
    p_cal = pd33['train'][cal_mask]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_cal, y_cal)
    print(f'calibrator: isotonic on last {cal_n} train loans '
          f'(bad rate {y_cal.mean():.3f})')

    rows = []
    print('\n=== PD(33mo) vs binary labels ===')
    for s in ['valid', 'test']:
        Xs, ys, Ts, evs = data[s]
        for tag, p in [('raw', pd33[s]), ('iso', iso.predict(pd33[s]))]:
            auc = roc_auc_score(ys, p)
            ll = log_loss(ys, np.clip(p, 1e-6, 1 - 1e-6))
            br = brier_score_loss(ys, p)
            rows.append(metrics_row(ys, p, f'{s}_{tag}'))
            print(f'{s} {tag}: PD33 AUC={auc:.5f} KS={ks(ys, p):.4f} '
                  f'logloss={ll:.4f} brier={br:.4f} | '
                  f'mean PD={np.mean(p):.3f} vs bad={ys.mean():.3f}')

    # binary CatBoost baseline for the same splits (Scenario A)
    base = {'valid': (0.646804, 0.402443, 0.121717),
            'test': (0.644239, 0.389742, 0.116929)}  # auc, logloss, brier
    print('\n=== vs binary CatBoost baseline (Scenario A) ===')
    for s in ('valid', 'test'):
        r = next(r for r in rows if r['split'] == f'{s}_iso')
        b = base['test' if s == 'test' else 'valid']
        print(f"{s}: hazard-iso AUC={r['auc']:.4f} vs cat {b[0]:.4f} "
              f"(delta {r['auc']-b[0]:+.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(f'{ROOT}/experiments/reports/hazard_model_{args.learner}.csv',
               index=False)

    # calibration deciles on test (isotonic version)
    Xs, ys, Ts, evs = data['test']
    p_iso = iso.predict(pd33['test'])
    dec = pd.qcut(p_iso, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'dec': dec, 'y': ys, 'pd': p_iso}).groupby('dec').agg(
        emp=('y', 'mean'), pred=('pd', 'mean'), n=('y', 'size'))
    print('\ntest calibration deciles (isotonic):')
    print(cal.round(3).to_string())
    cal.to_csv(f'{ROOT}/experiments/reports/hazard_model_{args.learner}_deciles.csv')

    # --- survival summaries on OOT (test) ---
    # Cohort PD(t) curves by predicted-risk decile; expected-loss term
    # structure with EAD = loan_amnt and canonical LGD = 0.55; median
    # time-to-event observed vs predicted.
    horizons = [6, 12, 24, M]
    Xs, ys, Ts, evs = data['test']
    p_iso = iso.predict(pd33['test'])
    dec = pd.qcut(p_iso, 10, labels=False, duplicates='drop')
    cohort_rows = []
    for d in sorted(set(dec)):
        m = np.asarray(dec == d)
        row = {'decile': int(d), 'n': int(m.sum()),
               'pd33_iso': float(p_iso[m].mean())}
        for t in horizons:
            row[f'pd_{t}m'] = float(cum_pd(H['test'][m], t).mean())
        cohort_rows.append(row)
    cohort = pd.DataFrame(cohort_rows)
    cohort.to_csv(f'{ROOT}/experiments/reports/hazard_pd_curves_{args.learner}.csv',
                  index=False)

    lgd = 0.55
    if 'loan_amnt' in Xs.columns:
        ead = Xs['loan_amnt'].astype(float).to_numpy()
    else:
        ead = np.full(len(ys), 15e6)
    el_rows = []
    el_cum_prev = 0.0
    for t in range(1, M + 1):
        el_cum = float((ead * lgd * cum_pd(H['test'], t)).sum())
        el_rows.append({'month': t,
                        'el_cumulative_usd': el_cum,
                        'el_marginal_usd': el_cum - el_cum_prev})
        el_cum_prev = el_cum
    el = pd.DataFrame(el_rows)
    el.to_csv(f'{ROOT}/experiments/reports/hazard_el_term_{args.learner}.csv',
              index=False)

    med_obs = float(Ts[evs.to_numpy() == 1].median()) \
        if (evs.to_numpy() == 1).any() else np.nan
    med_pred_bads = float(np.median(median_event_time(H['test'][ys == 1])))
    med_pred_all = float(np.median(median_event_time(H['test'])))
    print(f'\n=== survival summaries (OOT, learner={args.learner}) ===')
    print('cohort PD(t) by risk decile:')
    print(cohort.round(4).to_string(index=False))
    peak_m = int(el.loc[el.el_marginal_usd.idxmax(), 'month'])
    print(f'EL cumulative at {M}m: '
          f'{el.el_cumulative_usd.iloc[-1]/1e9:.2f}B USD '
          f'(EAD=loan_amnt USD, LGD={lgd}); peak marginal month={peak_m}')
    print(f'median time-to-event months: observed(bads)={med_obs:.1f} | '
          f'predicted(bads)={med_pred_bads:.1f} | '
          f'predicted(all)={med_pred_all:.1f}')

if __name__ == '__main__':
    main()
