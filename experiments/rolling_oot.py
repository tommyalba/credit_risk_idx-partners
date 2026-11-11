"""Rolling-origin out-of-time evaluation.

The fixed split evaluates the model once on 2013Q1. This script instead
refits the production CatBoost recipe on an expanding window and scores
the following calendar quarter, for every quarter with enough history.
It answers: does discrimination decay gradually across vintages, or is
the train->valid gap a one-time regime break?

Only CatBoost is evaluated (the deployed winner in both scenarios);
XGBoost/LightGBM tracked within ~0.002 AUC in the fixed split and would
multiply runtime ~3x on this CPU.

Usage:
    python experiments/rolling_oot.py --scenario a
    python experiments/rolling_oot.py --scenario b

Writes reports/rolling_oot_{a,b}.csv and prints a summary table.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba          # noqa: E402
from src.metrics import calculate_probability_metrics      # noqa: E402
from src.training import FIT_STRATEGIES                    # noqa: E402
from src.utils import deserialize_data, split_train_test   # noqa: E402

from pipeline.core import BASE_PARAMS, RANDOM_STATE        # noqa: E402
from pipeline.scenario_a.features import (                 # noqa: E402
    add_scenario_a_features, SCENARIO_A_FEATURE_COLUMNS)
from pipeline.scenario_b.features import add_scenario_b_features  # noqa: E402

SCENARIOS = {
    'a': dict(interim_dir='data/interim',
              derive_fn=add_scenario_a_features,
              feature_columns=SCENARIO_A_FEATURE_COLUMNS,
              stem='scenario_a'),
    'b': dict(interim_dir='data/interim/scenario_b',
              derive_fn=add_scenario_b_features,
              feature_columns=None,   # scenario B pickles already derived scope
              stem='scenario_b'),
}

MIN_TRAIN = 3000      # skip origins with less history than this
MIN_TEST = 300        # skip windows too small to score meaningfully


def load_population(cfg):
    """Concat the three canonical splits back into one eligible frame."""
    interim = Path(cfg['interim_dir'])
    X = pd.concat([
        deserialize_data(str(interim / f'X_{s}_raw.pkl'))
        for s in ('train', 'valid', 'test')
    ])
    y = pd.concat([
        deserialize_data(str(interim / f'y_{s}.pkl'))
        for s in ('train', 'valid', 'test')
    ])
    y = y.loc[X.index]
    issue = X['issue_d']
    X_feat = cfg['derive_fn'](X)
    return X_feat, y, issue


def quarter_windows(issue):
    """Yield (label, start, end) for each calendar quarter in the data."""
    start = issue.min().to_period('Q').start_time
    end = issue.max().to_period('Q').end_time
    for p in pd.period_range(start, end, freq='Q'):
        yield f'{p.start_time.date()}..{p.end_time.date()}', \
            p.start_time, p.end_time + pd.Timedelta(days=1)


def run_scenario(key):
    cfg = SCENARIOS[key]
    X, y, issue = load_population(cfg)
    print(f'{cfg["stem"]}: population {len(X)} | '
          f'{issue.min().date()} -> {issue.max().date()}')

    cat_cols = [c for c in X.columns
                if str(X[c].dtype) == 'category']

    feature_cols = sorted(X.columns)
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

        # Same 85/15 fit/early-stop split shape as pipeline.core.
        X_fit, X_es, y_fit, y_es = split_train_test(
            X_tr, y_tr, test_size=0.15, stratify=y_tr,
            seed=RANDOM_STATE)

        import importlib
        est = getattr(importlib.import_module('catboost'),
                      'CatBoostClassifier')(**params)
        est = FIT_STRATEGIES['cat'](est, X_fit, y_fit, X_es, y_es)

        p_te = extract_positive_proba(est, X_te)
        m = calculate_probability_metrics(y_te, p_te)
        rows.append({
            'window': label,
            'n_train': n_train, 'n_test': n_test,
            'train_bad_rate': float(y_tr.mean()),
            'test_bad_rate': float(y_te.mean()),
            'fit_s': round(time.time() - t0, 1),
            **m,
        })
        print(f'  {label}: n_test={n_test} AUC={m["roc_auc"]:.4f} '
              f'KS={m["ks_statistic"]:.4f} ({rows[-1]["fit_s"]}s)',
              flush=True)

    out = pd.DataFrame(rows)
    out_path = Path('reports') / f'rolling_oot_{cfg["stem"][-1]}.csv'
    out.to_csv(out_path, index=False)
    print(f'\n{cfg["stem"]}: wrote {out_path} '
          f'({len(out)} windows)')
    print(out[['window', 'n_test', 'roc_auc', 'ks_statistic',
               'log_loss', 'brier_score']].to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['a', 'b'], required=True)
    args = ap.parse_args()
    run_scenario(args.scenario)


if __name__ == '__main__':
    main()
