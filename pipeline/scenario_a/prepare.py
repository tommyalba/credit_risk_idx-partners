"""Scenario A preparation — .py equivalent of canonical nb2.

Writes the SAME interim pickles nb2 produces (data/interim/X_*_raw.pkl),
so autoresearch/experiment.py, the dashboard and nb4 all consume
identical artifacts.

Usage:
    python -m pipeline.scenario_a.prepare
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import run_prepare  # noqa: E402
from .features import SCENARIO_A_FEATURE_COLUMNS  # noqa: E402


def main() -> None:
    run_prepare(
        feature_columns=SCENARIO_A_FEATURE_COLUMNS,
        out_dir='data/interim',
    )


if __name__ == '__main__':
    main()
