"""Retrain-watch tests: windowing + graceful empty paths (no interim)."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.retrain_watch import recent_shadow_frame  # noqa: E402


def _log_row(days_ago):
    ts = (datetime.now(timezone.utc)
          - timedelta(days=days_ago)).isoformat(timespec='seconds')
    return {'scored_at': ts, 'scenario': 'a', 'member_id': 'M1',
            'dti': 10.0, 'pd_score': 0.1, 'decision': 'APPROVE'}


def test_recent_shadow_window(tmp_path):
    rows = [_log_row(5), _log_row(40), _log_row(1)]
    pd.DataFrame(rows).to_csv(tmp_path / 'a_2026-01.csv', index=False)
    df = recent_shadow_frame(tmp_path, 'a', days=30)
    assert len(df) == 2  # 5d and 1d old; 40d excluded


def test_recent_shadow_empty_dir(tmp_path):
    assert recent_shadow_frame(tmp_path, 'a', days=30).empty


def test_recent_shadow_wrong_scenario(tmp_path):
    pd.DataFrame([_log_row(1)]).to_csv(tmp_path / 'b_2026-01.csv',
                                       index=False)
    assert recent_shadow_frame(tmp_path, 'a', days=30).empty
