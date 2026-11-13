"""Shadow-mode scoring: log live PDs, reconcile against outcomes later.

Shadow mode = score every live applicant and store the inputs + outputs,
without acting on the decision. Weeks/months later, when realized
outcomes arrive (bureau follow-up or loan tape), `shadow_report` joins
the log to outcomes and measures live-portfolio calibration — the
validation step the funded-only test split cannot provide.

Storage: append-only monthly CSVs under `data/shadow/`
(`{scenario}_YYYY-MM.csv`), each row = raw input columns + scored_at,
scenario, id, pd_score, decision. Raw inputs are kept so feature-level
drift can be analyzed on the live stream later.

Usage:
    # log during batch scoring
    python -m pipeline.scenario_a.score live.csv -o scores.csv --shadow

    # reconcile once outcomes exist (member_id + target columns)
    python -m pipeline.shadow report --outcomes outcomes.csv
    python -m pipeline.shadow report --outcomes outcomes.csv \
        --scenario b -o experiments/reports/shadow_report_b.json
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHADOW_DIR = Path(os.environ.get('SHADOW_DIR', 'data/shadow'))


def record_scores(input_df: pd.DataFrame, scored_df: pd.DataFrame,
                  scenario: str, shadow_dir=SHADOW_DIR,
                  id_col='member_id', score_col='pd_score',
                  decision_col='decision') -> Path:
    """Append one scoring batch to the monthly shadow log.

    `scored_df` must contain score_col/decision_col (and id_col if the
    input lacks it). Returns the CSV path written.
    """
    month = datetime.now(timezone.utc).strftime('%Y-%m')
    out = pd.concat(
        [input_df.reset_index(drop=True),
         scored_df.drop(columns=[c for c in (id_col,) if c in scored_df.columns
                                 and c in input_df.columns],
                        errors='ignore').reset_index(drop=True)],
        axis=1)
    row = out.copy()
    row.insert(0, 'scenario', scenario)
    row.insert(0, 'scored_at',
               datetime.now(timezone.utc).isoformat(timespec='seconds'))

    shadow_dir = Path(shadow_dir)
    shadow_dir.mkdir(parents=True, exist_ok=True)
    path = shadow_dir / f'{scenario}_{month}.csv'
    header = not path.exists()
    row.to_csv(path, mode='a', header=header, index=False)
    return path


def load_shadow(shadow_dir=SHADOW_DIR, scenario=None) -> pd.DataFrame:
    """Read all monthly shadow logs (optionally one scenario)."""
    shadow_dir = Path(shadow_dir)
    files = sorted(shadow_dir.glob('*.csv'))
    if scenario:
        files = [f for f in files if f.name.startswith(f'{scenario}_')]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def shadow_report(shadow_dir=SHADOW_DIR, outcomes=None,
                  scenario='a', id_col='member_id',
                  label_col='target', score_col='pd_score',
                  decision_col='decision') -> dict:
    """Join shadow scores to realized outcomes and measure live PD quality.

    `outcomes`: path to CSV or a DataFrame with id_col + label_col
    (1 = default/BAD, 0 = good). Rows without a realized outcome yet are
    excluded; coverage is reported.
    """
    from src.calibration import (calibration_slope_intercept,
                                 expected_calibration_error)
    from src.metrics import calculate_probability_metrics

    log = load_shadow(shadow_dir, scenario)
    if log.empty:
        raise ValueError(f'no shadow scores found in {shadow_dir}')
    if id_col not in log.columns:
        raise ValueError(f'shadow log missing id column {id_col!r}')
    if score_col not in log.columns:
        raise ValueError(f'shadow log missing score column {score_col!r}')

    if isinstance(outcomes, (str, Path)):
        path = Path(outcomes)
        out = (pd.read_parquet(path) if path.suffix == '.parquet'
               else pd.read_csv(path))
    else:
        out = outcomes
    for c in (id_col, label_col):
        if c not in out.columns:
            raise ValueError(f'outcomes missing column {c!r}')

    # one row per applicant: keep the most recent score
    log = (log.sort_values('scored_at').drop_duplicates(id_col,
                                                      keep='last'))
    merged = log[[id_col, score_col, decision_col,
                  'scored_at']].merge(
        out[[id_col, label_col]].drop_duplicates(id_col),
        on=id_col, how='inner')

    rep = {
        'scenario': scenario,
        'shadow_rows': int(len(log)),
        'matched': int(len(merged)),
        'coverage': round(len(merged) / len(log), 4),
        'window': [str(log['scored_at'].min()),
                   str(log['scored_at'].max())],
    }
    if merged.empty:
        rep['note'] = 'no scored applicants have realized outcomes yet'
        return rep

    y = merged[label_col].astype(int)
    p = merged[score_col].astype(float)
    rep['n'] = int(len(merged))
    rep['realized_bad_rate'] = round(float(y.mean()), 4)
    rep['mean_predicted_pd'] = round(float(p.mean()), 4)
    rep['pd_gap'] = round(float(p.mean() - y.mean()), 4)
    rep.update({k: round(v, 4) for k, v in
                calculate_probability_metrics(y, p.values).items()})
    cal = calibration_slope_intercept(y, p.values)
    rep['calibration_slope'] = round(cal['slope'], 4)
    rep['calibration_intercept'] = round(cal['intercept'], 4)
    rep['ece'] = round(expected_calibration_error(y, p.values), 4)

    if decision_col in merged.columns:
        for dec in merged[decision_col].unique():
            sub = merged[merged[decision_col] == dec]
            rep[f'bad_rate_{dec}'] = round(float(sub[label_col].mean()), 4)
            rep[f'n_{dec}'] = int(len(sub))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description='Shadow-mode reconciliation')
    sub = ap.add_subparsers(dest='cmd', required=True)
    rp = sub.add_parser('report', help='join shadow log to outcomes')
    rp.add_argument('--outcomes', required=True,
                    help='CSV/parquet with id_col + label_col')
    rp.add_argument('--scenario', default='a', choices=['a', 'b'])
    rp.add_argument('--shadow-dir', default=str(SHADOW_DIR))
    rp.add_argument('--id-col', default='member_id')
    rp.add_argument('--label-col', default='target')
    rp.add_argument('--score-col', default='pd_score')
    rp.add_argument('--decision-col', default='decision')
    rp.add_argument('-o', '--output', help='write JSON report here')
    args = ap.parse_args()

    rep = shadow_report(
        args.shadow_dir, args.outcomes, scenario=args.scenario,
        id_col=args.id_col, label_col=args.label_col,
        score_col=args.score_col, decision_col=args.decision_col)
    print(json.dumps(rep, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(rep, indent=2))
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
