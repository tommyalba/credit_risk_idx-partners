"""Per-vintage recalibration drift check.

rolling_oot.py showed discrimination (AUC) decays gradually across
quarters. This script asks the companion question: does *calibration*
decay faster? For each quarterly window it refits the production
CatBoost recipe on expanding history, then on the test quarter reports

  - raw calibration diagnostics (logit slope/intercept, ECE, Brier)
  - performance of two recalibrators fit on the LAST 15% of the
    training window (temporally adjacent to the test quarter):
    isotonic and logit (a + b*logit(p)).

If raw ECE/Brier drift upward faster than AUC drifts downward, the
recalibration cadence should be shorter than the retraining cadence.

Only the temporal ordering is used; OOT is just the last quarter of
the same rolling evaluation (no separate holdout exists here).

Usage:
    python experiments/recalibration_drift.py --scenario a
    python experiments/recalibration_drift.py --scenario b

Writes experiments/reports/recalibration_drift_{a,b}.csv and prints a summary.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba          # noqa: E402
from src.calibration import (                              # noqa: E402
    calibration_slope_intercept, expected_calibration_error,
    fit_logit_recalibration, apply_logit_recalibration)
from src.training import FIT_STRATEGIES                    # noqa: E402
from src.utils import deserialize_data, split_train_test   # noqa: E402

from pipeline.core import BASE_PARAMS, RANDOM_STATE        # noqa: E402
from rolling_oot import (                      # noqa: E402
    SCENARIOS, MIN_TRAIN, MIN_TEST, quarter_windows,
    load_population)

CAL_TAIL_SHARE = 0.15   # last 15% of each training window = recal fit


def _metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    d = calibration_slope_intercept(y, p)
    return {
        'auc': roc_auc_score(y, p),
        'logloss': log_loss(y, p),
        'brier': brier_score_loss(y, p),
        'ece': expected_calibration_error(y, p),
        'slope': d['slope'], 'intercept': d['intercept'],
    }


def run_scenario(key):
    cfg = SCENARIOS[key]
    X, y, issue = load_population(cfg)
    print(f'{cfg["stem"]}: population {len(X)} | '
          f'{issue.min().date()} -> {issue.max().date()}')

    cat_cols = [c for c in X.columns
                if str(X[c].dtype) == 'category']
    params = dict(BASE_PARAMS['cat'])
    params['cat_features'] = cat_cols

    rows = []
    for label, q_start, q_end in quarter_windows(issue):
        test_mask = (issue >= q_start) & (issue < q_end)
        train_mask = issue < q_start
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if n_train < MIN_TRAIN or n_test < MIN_TEST:
            continue

        t0 = time.time()
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        X_fit, X_es, y_fit, y_es = split_train_test(
            X_tr, y_tr, test_size=0.15, stratify=y_tr,
            seed=RANDOM_STATE)

        import importlib
        est = getattr(importlib.import_module('catboost'),
                      'CatBoostClassifier')(**params)
        est = FIT_STRATEGIES['cat'](est, X_fit, y_fit, X_es, y_es)

        # recalibration fit set: temporally LAST 15% of the training
        # window (adjacent to the test quarter), labels = binary target
        tr_issue = issue[train_mask]
        tail_idx = tr_issue.sort_values().index[
            -int(round(n_train * CAL_TAIL_SHARE)):]
        p_cal = extract_positive_proba(est, X_tr.loc[tail_idx])
        y_cal = np.asarray(y_tr.loc[tail_idx])

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_cal, y_cal)
        a, b = fit_logit_recalibration(p_cal, y_cal)

        p_te = extract_positive_proba(est, X_te)
        y_te_arr = np.asarray(y_te)

        raw = _metrics(y_te_arr, p_te)
        iso_m = _metrics(y_te_arr, iso.predict(p_te))
        logit_m = _metrics(y_te_arr, apply_logit_recalibration(p_te, a, b))

        rows.append({
            'window': label, 'n_train': n_train, 'n_test': n_test,
            'test_bad_rate': float(np.mean(y_te_arr)),
            'fit_s': round(time.time() - t0, 1),
            'auc': raw['auc'],
            'slope_raw': raw['slope'], 'int_raw': raw['intercept'],
            'ece_raw': raw['ece'], 'brier_raw': raw['brier'],
            'logloss_raw': raw['logloss'],
            'ece_iso': iso_m['ece'], 'brier_iso': iso_m['brier'],
            'logloss_iso': iso_m['logloss'],
            'ece_logit': logit_m['ece'], 'brier_logit': logit_m['brier'],
            'logloss_logit': logit_m['logloss'],
        })
        print(f"  {label}: AUC={raw['auc']:.4f} slope={raw['slope']:.3f} "
              f"ECE raw/iso/logit={raw['ece']:.4f}/"
              f"{iso_m['ece']:.4f}/{logit_m['ece']:.4f} "
              f"({rows[-1]['fit_s']}s)", flush=True)

    out = pd.DataFrame(rows)
    out_path = Path('experiments/reports') / f'recalibration_drift_{cfg["stem"][-1]}.csv'
    out.to_csv(out_path, index=False)
    print(f'\n{cfg["stem"]}: wrote {out_path} ({len(out)} windows)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['a', 'b'], required=True)
    args = ap.parse_args()
    run_scenario(args.scenario)


if __name__ == '__main__':
    main()
