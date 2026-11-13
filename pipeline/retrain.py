"""Gated retraining: refit a scenario and promote only if OOT does not regress.

The incumbent bundle is never overwritten during training — the challenger
trains into a staging directory. On success the staging bundle is copied
over the live path; on gate failure the incumbent bundle and reports are
restored from an archive taken before the run.

Gate (default): challenger OOT ROC-AUC must be >= incumbent - 0.005
(non-inferiority margin). Override with --min-auc-margin.

Usage:
    python -m pipeline.retrain --scenario a --data data/raw/new_tape.csv
    python -m pipeline.retrain --scenario b --promote-only-check   # dry gate

Without --promote the run still trains and reports the gate verdict but
leaves the live bundle untouched (reports/ is still refreshed to the
challenger's numbers — restore from the archive if you discard it).
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.core import run_prepare, run_training  # noqa: E402
from pipeline.scenario_a import features as feat_a  # noqa: E402
from pipeline.scenario_b import features as feat_b  # noqa: E402

REPORTS = ['{stem}_metrics.json', '{stem}_leaderboard.csv',
           '{stem}_calibration.csv']

SPECS = {
    'a': {
        'feature_columns': feat_a.SCENARIO_A_FEATURE_COLUMNS,
        'derive_fn': feat_a.add_scenario_a_features,
        'interim_dir': 'data/interim',
        'live_bundle': Path('models/scenario_a/scenario_a_bundle.joblib'),
        'staging_dir': Path('models/scenario_a/staging'),
        'report_stem': 'scenario_a',
        'scenario_label': 'A',
    },
    'b': {
        'feature_columns': feat_b.SCENARIO_B_FEATURE_COLUMNS,
        'derive_fn': feat_b.add_scenario_b_features,
        'interim_dir': 'data/interim/scenario_b',
        'live_bundle': Path('models/scenario_b/scenario_b_bundle.joblib'),
        'staging_dir': Path('models/scenario_b/staging'),
        'report_stem': 'scenario_b',
        'scenario_label': 'B',
    },
}


def incumbent_auc(report_path: Path) -> float:
    rep = json.loads(report_path.read_text())
    return float(rep['metrics']['oot_test']['roc_auc'])


def challenger_auc(report_path: Path) -> float:
    return incumbent_auc(report_path)


def gate_passes(new_auc: float, old_auc: float, margin: float) -> bool:
    """Non-inferiority gate: challenger OOT AUC must not drop below
    incumbent AUC minus margin."""
    return new_auc >= old_auc - margin


def _archive(spec: dict, archive_dir: Path) -> list[Path]:
    """Copy live bundle + report artifacts into archive_dir."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    stem = spec['report_stem']
    candidates = [spec['live_bundle']] + [
        Path('reports') / r.format(stem=stem) for r in REPORTS]
    for src in candidates:
        if src.exists():
            dst = archive_dir / src.name
            shutil.copy2(src, dst)
            saved.append(dst)
    return saved


def _restore(spec: dict, archive_dir: Path) -> None:
    stem = spec['report_stem']
    for name in [spec['live_bundle'].name] + [
            r.format(stem=stem) for r in REPORTS]:
        src = archive_dir / name
        if not src.exists():
            continue
        if name.endswith('_bundle.joblib'):
            shutil.copy2(src, spec['live_bundle'])
        else:
            shutil.copy2(src, Path('reports') / name)


def main() -> int:
    ap = argparse.ArgumentParser(description='Gated scenario retraining')
    ap.add_argument('--scenario', choices=sorted(SPECS), required=True)
    ap.add_argument('--data', default='data/raw/loan_data_2007_2014.csv',
                    help='raw loan tape CSV for the prepare step')
    ap.add_argument('--promote', action='store_true',
                    help='copy the challenger bundle over the live one when '
                         'the gate passes (default: report only)')
    ap.add_argument('--min-auc-margin', type=float, default=0.005,
                    help='max allowed OOT AUC drop vs incumbent (default 0.005)')
    args = ap.parse_args()

    spec = SPECS[args.scenario]
    stem = spec['report_stem']
    metrics_path = Path('reports') / f'{stem}_metrics.json'
    staging_bundle = spec['staging_dir'] / f'{stem}_bundle.joblib'

    if not metrics_path.exists():
        print(f'no incumbent metrics at {metrics_path}; run train first')
        return 2
    old_auc = incumbent_auc(metrics_path)
    print(f'incumbent OOT AUC: {old_auc:.4f}')

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive_dir = Path('models/archive') / f'scenario_{args.scenario}' / ts
    _archive(spec, archive_dir)
    print(f'archived incumbent -> {archive_dir}')

    print('prepare...')
    run_prepare(feature_columns=spec['feature_columns'],
                out_dir=spec['interim_dir'], data_path=args.data)

    print('train (staging)...')
    spec['staging_dir'].mkdir(parents=True, exist_ok=True)
    run_training(interim_dir=spec['interim_dir'],
                 derive_fn=spec['derive_fn'],
                 out_dir=str(spec['staging_dir']),
                 report_stem=stem,
                 scenario_label=spec['scenario_label'])

    new_auc = challenger_auc(metrics_path)
    ok = gate_passes(new_auc, old_auc, args.min_auc_margin)
    print(f'challenger OOT AUC: {new_auc:.4f} | '
          f'gate (margin {args.min_auc_margin}): {"PASS" if ok else "FAIL"}')

    if not ok:
        _restore(spec, archive_dir)
        print('gate failed: incumbent bundle + reports restored from archive')
        return 1

    if args.promote:
        shutil.copy2(staging_bundle, spec['live_bundle'])
        print(f'promoted -> {spec["live_bundle"]}')
    else:
        print('gate passed but --promote not set: live bundle unchanged; '
              'staging bundle at ' + str(staging_bundle))
    return 0


if __name__ == '__main__':
    sys.exit(main())
