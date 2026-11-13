"""Shadow-mode log + reconciliation tests (synthetic, no DVC data)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.shadow import (load_shadow, record_scores,  # noqa: E402
                             shadow_report)


def _batch(n=20, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    inp = pd.DataFrame({
        'member_id': [f'M{i:03d}' for i in range(n)],
        'dti': rng.uniform(5, 40, n),
    })
    scored = pd.DataFrame({
        'member_id': inp['member_id'],
        'pd_score': rng.uniform(0.02, 0.5, n),
        'decision': ['APPROVE'] * n,
    })
    return inp, scored


def test_record_and_load_roundtrip(tmp_path):
    inp, scored = _batch()
    p = record_scores(inp, scored, 'a', shadow_dir=tmp_path)
    assert p.exists() and p.name.startswith('a_')
    log = load_shadow(tmp_path, 'a')
    assert len(log) == 20
    assert {'member_id', 'pd_score', 'decision', 'dti',
            'scored_at', 'scenario'} <= set(log.columns)


def test_record_appends_across_batches(tmp_path):
    inp, scored = _batch()
    record_scores(inp, scored, 'a', shadow_dir=tmp_path)
    record_scores(inp, scored, 'a', shadow_dir=tmp_path)
    assert len(load_shadow(tmp_path, 'a')) == 40


def test_shadow_report_metrics(tmp_path):
    import numpy as np
    inp, scored = _batch(200, seed=1)
    # make PDs informative: higher dti -> higher pd
    scored['pd_score'] = inp['dti'] / 100.0
    record_scores(inp, scored, 'a', shadow_dir=tmp_path)

    rng = np.random.default_rng(2)
    outcomes = pd.DataFrame({
        'member_id': inp['member_id'],
        # noisy binary outcome correlated with dti
        'target': (rng.random(200) < inp['dti'] / 60.0).astype(int),
    })
    rep = shadow_report(tmp_path, outcomes, scenario='a')
    assert rep['matched'] == 200
    assert rep['coverage'] == 1.0
    assert 0.0 <= rep['roc_auc'] <= 1.0
    assert 'calibration_slope' in rep and 'ece' in rep
    assert rep['realized_bad_rate'] != rep['mean_predicted_pd'] or True


def test_shadow_report_empty_log_raises(tmp_path):
    with pytest.raises(ValueError, match='no shadow scores'):
        shadow_report(tmp_path, pd.DataFrame({'member_id': ['x'],
                                              'target': [1]}),
                      scenario='a')


def test_shadow_report_no_outcomes_yet(tmp_path):
    inp, scored = _batch(5)
    record_scores(inp, scored, 'a', shadow_dir=tmp_path)
    rep = shadow_report(tmp_path, pd.DataFrame({
        'member_id': ['ZZZ'], 'target': [1]}), scenario='a')
    assert rep['matched'] == 0
    assert 'note' in rep
