"""PSI-triggered retraining watch.

Checks recent shadow-score batches for population drift and, on a
'Significant shift' verdict, invokes the gated retrain
(`python -m pipeline.retrain --promote`). Moderate verdicts are
reported for human review; Stable exits quietly.

Designed for cron/systemd timers — exit codes:
    0 = stable or moderate (no action)
    2 = significant drift (retrain attempted if --auto-retrain, else alert)
    1 = error (missing data, failed retrain)

Usage:
    # check last 30 days of shadow scores for scenario A
    python -m pipeline.retrain_watch --scenario a --days 30

    # auto-retrain on significant drift (gate still applies)
    python -m pipeline.retrain_watch --scenario a --auto-retrain

    # or check any batch CSV instead of the shadow log
    python -m pipeline.retrain_watch --scenario a --input batch.csv

Cron example (daily 06:00):
    0 6 * * * cd /path/to/credit_risk_modeling && \
      /path/to/python -m pipeline.retrain_watch --scenario a \
      --auto-retrain >> logs/watch_a.log 2>&1
"""
import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.core import drift_report  # noqa: E402
from pipeline.scenario_a import score as score_a  # noqa: E402
from pipeline.scenario_a import features as feat_a  # noqa: E402
from pipeline.scenario_b import score as score_b  # noqa: E402
from pipeline.scenario_b import features as feat_b  # noqa: E402
from pipeline.shadow import load_shadow  # noqa: E402

SPECS = {
    'a': dict(label='Scenario A', derive=feat_a.add_scenario_a_features,
              forbidden=feat_a.FORBIDDEN_FEATURES,
              required=score_a.REQUIRED_RAW,
              interim='data/interim',
              bundle=score_a.BUNDLE_PATH),
    'b': dict(label='Scenario B', derive=feat_b.add_scenario_b_features,
              forbidden=feat_b.FORBIDDEN_FEATURES,
              required=score_b.REQUIRED_RAW,
              interim='data/interim/scenario_b',
              bundle=score_b.BUNDLE_PATH),
}

SIGNIFICANT = 'Significant shift'


def recent_shadow_frame(shadow_dir, scenario, days):
    """Shadow-log rows scored within the last `days` days."""
    log = load_shadow(shadow_dir, scenario)
    if log.empty or 'scored_at' not in log.columns:
        return pd.DataFrame()
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=days)).isoformat()
    log['scored_at'] = pd.to_datetime(log['scored_at'], utc=True)
    return log[log['scored_at'] >= pd.Timestamp(cutoff)]


def check_drift(df, spec, bundle):
    return drift_report(df, bundle, spec['derive'], spec['forbidden'],
                        spec['required'], spec['interim'])


def run_retrain(scenario, promote=True):
    cmd = [sys.executable, '-m', 'pipeline.retrain',
           '--scenario', scenario]
    if promote:
        cmd.append('--promote')
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description='PSI-triggered retrain watch')
    ap.add_argument('--scenario', choices=['a', 'b'], required=True)
    ap.add_argument('--days', type=int, default=30,
                    help='shadow window in days (default 30)')
    ap.add_argument('--shadow-dir', default='data/shadow')
    ap.add_argument('--input', default=None,
                    help='check this CSV/parquet instead of the shadow log')
    ap.add_argument('--auto-retrain', action='store_true',
                    help='run gated retrain when drift is Significant')
    args = ap.parse_args()

    spec = SPECS[args.scenario]

    if args.input:
        path = Path(args.input)
        df = (pd.read_parquet(path) if path.suffix == '.parquet'
               else pd.read_csv(path))
        source = str(path)
    else:
        df = recent_shadow_frame(args.shadow_dir, args.scenario,
                                 args.days)
        source = f'{args.shadow_dir} last {args.days}d'

    if df.empty:
        print(f'[{spec["label"]}] no data in {source}; nothing to check')
        return 0
    print(f'[{spec["label"]}] checking {len(df)} records from {source}')

    import joblib
    bundle = joblib.load(spec['bundle'])
    rep = check_drift(df, spec, bundle)
    verdict = rep['worst_verdict']
    print(f'worst verdict: {verdict} | score PSI {rep["score_psi"]:.4f} '
          f'({rep["score_verdict"]})')

    if SIGNIFICANT not in verdict:
        return 0

    print(f'[{spec["label"]}] SIGNIFICANT DRIFT — '
          + ('running gated retrain' if args.auto_retrain
             else 'alert: run retrain manually'))
    if args.auto_retrain:
        rc = run_retrain(args.scenario, promote=True)
        return 0 if rc == 0 else 1
    return 2


if __name__ == '__main__':
    sys.exit(main())
