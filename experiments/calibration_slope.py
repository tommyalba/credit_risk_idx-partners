"""Calibration slope/intercept diagnostic for both scenarios.

Fits the logistic recalibration map logit(y) ~ a + b * logit(p) on the
temporal validation split and asks two questions:

1. Diagnostic: what are a and b on validation, and does the same fit
   hold on OOT (refit there)? Slope b < 1 => overconfident scores.
2. Transfer: does a recalibrator fit on validation improve OOT
   log-loss/Brier, or does it just fit validation noise?

Also reports ECE (expected calibration error) per version.

Usage: python experiments/calibration_slope.py
Writes experiments/reports/calibration_slope.csv and prints a summary table.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (                          # noqa: E402
    apply_logit_recalibration,
    calibration_slope_intercept,
    expected_calibration_error,
    fit_logit_recalibration,
)
from src.evaluation import extract_positive_proba      # noqa: E402
from src.metrics import calculate_probability_metrics  # noqa: E402
from src.utils import deserialize_data                 # noqa: E402

from pipeline.core import _apply_cal, prepare_scoring_frame  # noqa: E402
from pipeline.scenario_a.features import add_scenario_a_features  # noqa: E402
from pipeline.scenario_b.features import add_scenario_b_features  # noqa: E402

SCENARIOS = {
    'a': dict(bundle='models/scenario_a/scenario_a_bundle.joblib',
              interim='data/interim', derive=add_scenario_a_features),
    'b': dict(bundle='models/scenario_b/scenario_b_bundle.joblib',
              interim='data/interim/scenario_b',
              derive=add_scenario_b_features),
}


def split_probs(bundle, derive, interim, split):
    X_raw = deserialize_data(str(Path(interim) / f'X_{split}_raw.pkl'))
    y = np.asarray(deserialize_data(str(Path(interim) / f'y_{split}.pkl')))
    X = prepare_scoring_frame(X_raw, bundle, derive)
    p = extract_positive_proba(bundle['model'], X)
    return y, _apply_cal(bundle.get('calibrator'),
                         bundle.get('calibrator_name'), p)


def main():
    rows = []
    for key, cfg in SCENARIOS.items():
        bundle = joblib.load(cfg['bundle'])
        probs = {s: split_probs(bundle, cfg['derive'], cfg['interim'], s)
                 for s in ('valid', 'test')}
        yv, pv = probs['valid']
        yt, pt = probs['test']

        # diagnostic fits on each split's own data
        sv = calibration_slope_intercept(yv, pv)
        st = calibration_slope_intercept(yt, pt)
        # transfer: fit on valid, apply to both
        a, b = fit_logit_recalibration(pv, yv)
        pv_re = apply_logit_recalibration(pv, a, b)
        pt_re = apply_logit_recalibration(pt, a, b)

        for split_name, y, versions in [
            ('valid', yv, {'deployed': pv, 'recal_from_valid': pv_re}),
            ('test', yt, {'deployed': pt,
                          'recal_from_valid': pt_re}),
        ]:
            diag = sv if split_name == 'valid' else st
            for vname, p in versions.items():
                m = calculate_probability_metrics(y, p)
                rows.append({
                    'scenario': key.upper(), 'split': split_name,
                    'version': vname,
                    'slope_own_fit': diag['slope'],
                    'intercept_own_fit': diag['intercept'],
                    'ece': expected_calibration_error(y, p),
                    'roc_auc': m['roc_auc'], 'log_loss': m['log_loss'],
                    'brier_score': m['brier_score'],
                })
        rows.append({
            'scenario': key.upper(), 'split': 'transfer',
            'version': 'logit_recal_valid->test',
            'slope_own_fit': b, 'intercept_own_fit': a,
            'ece': np.nan, 'roc_auc': np.nan,
            'log_loss': np.nan, 'brier_score': np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv('experiments/reports/calibration_slope.csv', index=False)
    cols = ['scenario', 'split', 'version', 'slope_own_fit',
            'intercept_own_fit', 'ece', 'roc_auc', 'log_loss',
            'brier_score']
    print(df[cols].round(4).to_string(index=False))
    print('\nwrote experiments/reports/calibration_slope.csv')


if __name__ == '__main__':
    main()
