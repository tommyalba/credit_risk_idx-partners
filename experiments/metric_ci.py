"""Bootstrap confidence intervals for the headline metrics, both scenarios.

Resamples rows within each temporal split (never across splits), so the
CIs quantify sampling variability without breaking the temporal design.
Answers questions like "is valid AUC 0.6468 vs OOT 0.6442 a real
difference?" — compare interval overlap, not point estimates.

Usage:
    python experiments/metric_ci.py [--boot 1000]

Writes experiments/reports/metric_ci.csv and prints a summary table.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import extract_positive_proba          # noqa: E402
from src.metrics import bootstrap_metric_ci                # noqa: E402
from src.utils import deserialize_data                     # noqa: E402

from pipeline.core import _apply_cal, prepare_scoring_frame  # noqa: E402
from pipeline.scenario_a.features import (                 # noqa: E402
    FORBIDDEN_FEATURES, SCENARIO_A_FEATURE_COLUMNS,
    add_scenario_a_features)
from pipeline.scenario_b.features import add_scenario_b_features  # noqa: E402

SCENARIOS = {
    'a': dict(bundle='models/scenario_a/scenario_a_bundle.joblib',
              interim='data/interim',
              derive=add_scenario_a_features),
    'b': dict(bundle='models/scenario_b/scenario_b_bundle.joblib',
              interim='data/interim/scenario_b',
              derive=add_scenario_b_features),
}


def split_proba(bundle, derive, interim, split):
    X_raw = deserialize_data(str(Path(interim) / f'X_{split}_raw.pkl'))
    y = deserialize_data(str(Path(interim) / f'y_{split}.pkl'))
    X = prepare_scoring_frame(X_raw, bundle, derive)
    p = extract_positive_proba(bundle['model'], X)
    p = _apply_cal(bundle.get('calibrator'),
                   bundle.get('calibrator_name'), p)
    return np.asarray(y), np.asarray(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = []
    for key, cfg in SCENARIOS.items():
        bundle = joblib.load(cfg['bundle'])
        for split in ('valid', 'test'):
            y, p = split_proba(bundle, cfg['derive'], cfg['interim'],
                               split)
            cis = bootstrap_metric_ci(y, p, n_boot=args.boot,
                                      seed=args.seed)
            for metric, r in cis.items():
                rows.append({
                    'scenario': key.upper(), 'split': split,
                    'metric': metric, 'point': r['point'],
                    'ci_lo': r['lo'], 'ci_hi': r['hi'],
                    'boot_mean': r['mean'],
                    'n_boot': r['n_boot_used'],
                })
            a = cis['roc_auc']
            print(f"{key.upper()} {split}: AUC {a['point']:.4f} "
                  f"[{a['lo']:.4f}, {a['hi']:.4f}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('experiments/reports/metric_ci.csv', index=False)
    print('\nwrote experiments/reports/metric_ci.csv')
    print(df.pivot_table(index=['scenario', 'split', 'metric'],
                         values=['point', 'ci_lo', 'ci_hi'])
          .round(4).to_string())


if __name__ == '__main__':
    main()
