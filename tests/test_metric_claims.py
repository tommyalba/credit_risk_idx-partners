"""Tests pinning the mathematical claims of src/ metrics on synthetic data.

Each test constructs data where the correct answer is known analytically,
so a refactor that breaks the formula fails loudly rather than drifting.
"""
import numpy as np
import pandas as pd
import pytest

from src.population_stability_index import calculate_psi, psi_verdict, psi_report
from src.calibration_selection import brier_skill_score, reliability_table
from src.calibration import CALIBRATION_STRATEGIES
from src.performance_window import apply_performance_window_filter
from src.metrics import calculate_probability_metrics
from src.ks_statistic_evaluation import compute_ks_statistic


# ---------------------------------------------------------------------------
# PSI: identical -> ~0, disjoint -> large, verdict bands
# ---------------------------------------------------------------------------

def test_psi_identical_distribution_is_zero():
    s = pd.Series(["a"] * 50 + ["b"] * 30 + ["c"] * 20)
    res = calculate_psi(s, s.copy())
    assert res.attrs["psi_total"] == pytest.approx(0.0, abs=1e-9)
    assert psi_verdict(res.attrs["psi_total"]) == "Stable"


def test_psi_disjoint_distribution_is_large():
    expected = pd.Series(["a"] * 100)
    actual = pd.Series(["b"] * 100)
    res = calculate_psi(expected, actual)
    assert res.attrs["psi_total"] > 0.25
    assert "Significant" in psi_verdict(res.attrs["psi_total"])


def test_psi_verdict_bands():
    assert psi_verdict(0.05).startswith("Stable")
    assert "Moderate" in psi_verdict(0.15)
    assert "Significant" in psi_verdict(0.30)


def test_psi_report_ranks_drifted_feature_first():
    base = pd.DataFrame({
        "stable": ["x"] * 90 + ["y"] * 10,
        "drifted": ["x"] * 90 + ["y"] * 10,
    })
    comp = pd.DataFrame({
        "stable": ["x"] * 90 + ["y"] * 10,
        "drifted": ["z"] * 100,
    })
    report = psi_report(base, comp, ["stable", "drifted"])
    assert report.iloc[0]["feature"] == "drifted"


# ---------------------------------------------------------------------------
# Calibration: perfect calibration -> reliability table exact, skill = 1,
# isotonic recovers step function on separable data
# ---------------------------------------------------------------------------

def _perfectly_calibrated():
    """p and y where observed rate in each p-group equals p exactly."""
    ys, ps = [], []
    for p, k in [(0.1, 100), (0.5, 100), (0.9, 100)]:
        n_bad = int(round(p * k))
        ys += [1] * n_bad + [0] * (k - n_bad)
        ps += [p] * k
    return pd.Series(ys), np.array(ps)


def test_reliability_table_perfect_calibration():
    y, p = _perfectly_calibrated()
    table = reliability_table(y, p, n_bins=3)
    assert (table["mean_predicted"] - table["observed_rate"]).abs().max() < 1e-9


def test_brier_skill_score_perfect_predictor_is_one():
    y, _ = _perfectly_calibrated()
    p = y.to_numpy().astype(float)
    assert brier_skill_score(y, p)["skill_score"] == pytest.approx(1.0)


def test_isotonic_recovers_separable_step():
    p_train = np.linspace(0.0, 1.0, 201)
    y_train = pd.Series((p_train >= 0.5).astype(int))
    p_target = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    calibrated = CALIBRATION_STRATEGIES["isotonic"](p_train, y_train, p_target)
    assert calibrated[0] < 0.05 and calibrated[1] < 0.05
    assert calibrated[3] > 0.95 and calibrated[4] > 0.95


# ---------------------------------------------------------------------------
# Metrics: KS = 1 on separation, AUC ~ 0.5 on noise
# ---------------------------------------------------------------------------

def test_ks_statistic_perfect_separation():
    y = np.array([0] * 50 + [1] * 50)
    p = y.astype(float)
    result = compute_ks_statistic(y, p)
    assert result["ks_statistic_roc"] == pytest.approx(1.0, abs=1e-6)


def test_auc_random_scores_near_half():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 2, size=2000))
    p = rng.random(2000)
    metrics = calculate_probability_metrics(y, p)
    assert 0.45 < metrics["roc_auc"] < 0.55
    assert 0.25 < metrics["brier_score"] < 0.4  # ~1/3 for uniform p, 50/50 y


# ---------------------------------------------------------------------------
# Performance window: Current eligible iff age >= M (month granularity)
# ---------------------------------------------------------------------------

def test_performance_window_boundary_at_M():
    snapshot = pd.Timestamp("2016-01-01")
    M = 33
    df = pd.DataFrame({
        "issue_d": pd.to_datetime([
            "2013-04-01",  # Current, age exactly 33 -> eligible good
            "2013-05-01",  # Current, age 32 -> excluded
            "2014-01-01",  # Charged Off, age 24 -> eligible bad (asymmetric)
            "2014-06-01",  # Fully Paid, age 19 -> eligible good
            "2014-08-01",  # In Grace Period -> always excluded
        ]),
        "loan_status": [
            "Current", "Current", "Charged Off", "Fully Paid", "In Grace Period",
        ],
    })
    result = apply_performance_window_filter(df, snapshot, M)
    out = result.df.set_index("loan_status")

    assert result.n_input == 5
    assert result.n_eligible == 3
    assert result.n_excluded_immature == 1
    assert result.n_excluded_indeterminate == 1

    assert out.loc["Current", "target"] == 0
    assert out.loc["Charged Off", "target"] == 1
    assert out.loc["Fully Paid", "target"] == 0
