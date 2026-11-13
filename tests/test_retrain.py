"""Retrain promotion-gate tests (pure logic, no data needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.retrain import gate_passes  # noqa: E402


def test_gate_passes_when_challenger_equal():
    assert gate_passes(0.6442, 0.6442, 0.005)


def test_gate_passes_within_margin():
    assert gate_passes(0.6400, 0.6442, 0.005)


def test_gate_fails_below_margin():
    assert not gate_passes(0.6300, 0.6442, 0.005)


def test_gate_passes_when_challenger_better():
    assert gate_passes(0.6600, 0.6442, 0.005)
