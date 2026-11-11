"""Bootstrap confidence interval for the cost-optimal decision threshold.

The deployed tau is a point estimate tuned on the temporal validation
split. This script resamples validation rows, recomputes the
E[C(tau)] curve each time, and reports:

- the distribution of tau* (is the optimum flat or sharp?)
- expected-cost regret at the deployed bundle tau vs the per-resample
  optimum (how much does a fixed tau cost on average?)
- deterministic sensitivity: E[C] at tau +/- 0.02 / 0.05 on the full
  validation curve

Usage: python experiments/tau_ci.py [--boot 500]
Writes reports/threshold_ci.csv and prints a summary.
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
from src.expected_cost_threshold import (                  # noqa: E402
    expected_cost_curve_scalar)
from src.thresholding import find_best_cost_threshold      # noqa: E402
from src.utils import deserialize_data                     # noqa: E402

from pipeline.core import _apply_cal, prepare_scoring_frame  # noqa: E402
from pipeline.scenario_a.features import add_scenario_a_features  # noqa: E402
from pipeline.scenario_b.features import add_scenario_b_features  # noqa: E402

C_FN, C_FP = 8_250_000.0, 2_700_000.0

SCENARIOS = {
    'a': dict(bundle='models/scenario_a/scenario_a_bundle.joblib',
              interim='data/interim', derive=add_scenario_a_features),
    'b': dict(bundle='models/scenario_b/scenario_b_bundle.joblib',
              interim='data/interim/scenario_b',
              derive=add_scenario_b_features),
}


def valid_probs(bundle, derive, interim):
    X_raw = deserialize_data(str(Path(interim) / 'X_valid_raw.pkl'))
    y = np.asarray(deserialize_data(str(Path(interim) / 'y_valid.pkl')))
    X = prepare_scoring_frame(X_raw, bundle, derive)
    p = extract_positive_proba(bundle['model'], X)
    return y, _apply_cal(bundle.get('calibrator'),
                         bundle.get('calibrator_name'), p)


def curve_at(y, p, tau):
    """E[C] at a single threshold, same formula as the curve."""
    res = expected_cost_curve_scalar(y, p, C_FN, C_FP, n_thresholds=1001)
    idx = np.argmin(np.abs(res['thresholds'] - tau))
    return float(res['expected_costs'][idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    rows = []
    for key, cfg in SCENARIOS.items():
        bundle = joblib.load(cfg['bundle'])
        y, p = valid_probs(bundle, cfg['derive'], cfg['interim'])
        n = len(y)
        tau_bundle = float(bundle['threshold'])

        # point estimate on full valid
        tau_hat, cost_hat = find_best_cost_threshold(y, p)

        # bootstrap tau* distribution + regret at bundle tau
        taus_star, regrets = [], []
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            res = expected_cost_curve_scalar(
                y[idx], p[idx], C_FN, C_FP, n_thresholds=501)
            i_opt = int(np.argmin(res['expected_costs']))
            taus_star.append(res['thresholds'][i_opt])
            i_b = int(np.argmin(np.abs(res['thresholds'] - tau_bundle)))
            regrets.append(res['expected_costs'][i_b]
                           - res['expected_costs'][i_opt])
        taus_star = np.asarray(taus_star)
        regrets = np.asarray(regrets)

        # deterministic sensitivity on the full curve
        res_full = expected_cost_curve_scalar(y, p, C_FN, C_FP,
                                              n_thresholds=1001)
        c_min = res_full['optimal_cost']
        sens = {}
        for d in (-0.05, -0.02, 0.0, 0.02, 0.05):
            sens[d] = curve_at(y, p, tau_bundle + d) / c_min - 1.0

        pct = np.percentile(taus_star, [5, 25, 50, 75, 95])
        rows.append({
            'scenario': key.upper(), 'n_valid': n,
            'tau_bundle': tau_bundle, 'tau_hat_valid': float(tau_hat),
            'tau_star_p05': pct[0], 'tau_star_p25': pct[1],
            'tau_star_p50': pct[2], 'tau_star_p75': pct[3],
            'tau_star_p95': pct[4],
            'p_within_0.02': float(np.mean(np.abs(taus_star - tau_bundle) <= 0.02)),
            'regret_mean': float(regrets.mean()),
            'regret_p90': float(np.percentile(regrets, 90)),
            'cost_rel_tau-0.05': sens[-0.05], 'cost_rel_tau-0.02': sens[-0.02],
            'cost_rel_tau+0.02': sens[0.02], 'cost_rel_tau+0.05': sens[0.05],
        })
        r = rows[-1]
        print(f"{key.upper()}: tau bundle={tau_bundle:.4f} "
              f"hat={tau_hat:.4f} | tau* CI95 "
              f"[{pct[0]:.3f},{pct[4]:.3f}] med {pct[2]:.3f} | "
              f"regret mean {regrets.mean():,.0f} p90 "
              f"{np.percentile(regrets,90):,.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('reports/threshold_ci.csv', index=False)
    print('\nwrote reports/threshold_ci.csv')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
