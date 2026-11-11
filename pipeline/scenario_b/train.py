"""Scenario B model training — thin wrapper over pipeline.core.run_training.

Usage:
    python -m pipeline.scenario_b.train
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import run_training  # noqa: E402
from .features import add_scenario_b_features  # noqa: E402


def main() -> None:
    run_training(
        interim_dir='data/interim/scenario_b',
        derive_fn=add_scenario_b_features,
        out_dir='models/scenario_b',
        report_stem='scenario_b',
        scenario_label='B',
    )


if __name__ == '__main__':
    main()
