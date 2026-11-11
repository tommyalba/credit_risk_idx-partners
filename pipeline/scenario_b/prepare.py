"""Scenario B preparation — thin wrapper over pipeline.core.run_prepare.

Usage:
    python -m pipeline.scenario_b.prepare
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import run_prepare  # noqa: E402
from .features import SCENARIO_B_FEATURE_COLUMNS  # noqa: E402


def main() -> None:
    run_prepare(
        feature_columns=SCENARIO_B_FEATURE_COLUMNS,
        out_dir='data/interim/scenario_b',
    )


if __name__ == '__main__':
    main()
