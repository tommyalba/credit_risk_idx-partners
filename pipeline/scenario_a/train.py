"""Scenario A model training — .py equivalent of nb4 tree stage.

Usage:
    python -m pipeline.scenario_a.train
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.core import run_training  # noqa: E402
from .features import add_scenario_a_features  # noqa: E402


def main() -> None:
    run_training(
        interim_dir='data/interim',
        derive_fn=add_scenario_a_features,
        out_dir='models/scenario_a',
        report_stem='scenario_a',
        scenario_label='A',
    )


if __name__ == '__main__':
    main()
