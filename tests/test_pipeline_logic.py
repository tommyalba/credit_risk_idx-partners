"""Tests for the modeling-critical logic: performance window, cost threshold,
and WoE binning fit/transform round-trip."""
import numpy as np
import pandas as pd
import pytest

from src.performance_window import (
    compute_time_to_bad_distribution,
    recommend_performance_window,
    apply_performance_window_filter,
    safe_horizon_cutoff,
    truncate_to_safe_horizon,
)
from src.expected_cost_threshold import (
    expected_cost_curve_scalar,
    expected_cost_curve_per_observation,
)


# ---------------------------------------------------------------------------
# performance_window
# ---------------------------------------------------------------------------

def _pw_df():
    return pd.DataFrame({
        "issue_d": pd.to_datetime([
            "2012-01-01",  # bad, any age -> eligible
            "2012-12-01",  # current, age 1 -> excluded
            "2012-06-01",  # current, age 7 >= M -> eligible good
            "2012-03-01",  # fully paid -> eligible good
            "2012-10-01",  # in grace period -> excluded
        ]),
        "last_pymnt_d": pd.to_datetime([
            "2012-07-01", None, None, "2012-09-01", None,
        ]),
        "loan_status": [
            "Charged Off", "Current", "Current", "Fully Paid", "In Grace Period",
        ],
    })


def test_time_to_bad_distribution():
    df = _pw_df()
    months = compute_time_to_bad_distribution(df)
    assert len(months) == 1
    assert months.iloc[0] == 6  # Jan -> Jul


def test_recommend_performance_window():
    s = pd.Series([float(m) for m in range(1, 11)])
    assert recommend_performance_window(s, coverage=0.90) == 10  # ceil(9.1)


def test_apply_performance_window_filter_asymmetric():
    snap = pd.Timestamp("2013-01-01")
    res = apply_performance_window_filter(_pw_df(), snapshot_date=snap, M=6)
    statuses = res.df["loan_status"].tolist()
    # bad always eligible; resolved good always eligible; mature Current eligible;
    # immature Current and indeterminate excluded
    assert sorted(statuses) == ["Charged Off", "Current", "Fully Paid"]
    assert res.df.loc[res.df["loan_status"] == "Charged Off", "target"].iloc[0] == 1
    assert res.df.loc[res.df["loan_status"] == "Fully Paid", "target"].iloc[0] == 0
    assert res.n_excluded_immature == 1
    assert res.n_excluded_indeterminate == 1


def test_safe_horizon_cutoff_and_truncate():
    assert safe_horizon_cutoff(pd.Timestamp("2016-01-01"), M=33) == pd.Timestamp("2013-04-01")
    df = _pw_df()
    out = truncate_to_safe_horizon(df, pd.Timestamp("2013-01-01"), M=6)
    # cutoff = 2012-07-01; rows issued before it survive
    assert (out["issue_d"] < pd.Timestamp("2012-07-01")).all()


# ---------------------------------------------------------------------------
# expected_cost_threshold
# ---------------------------------------------------------------------------

def test_scalar_cost_curve_argmin():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 500)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.05, 500), 0.01, 0.99)
    res = expected_cost_curve_scalar(y, p, c_fn=8_250_000.0, c_fp=2_700_000.0)
    assert 0.0 <= res["optimal_threshold"] <= 1.0
    assert res["optimal_cost"] == pytest.approx(res["expected_costs"].min())
    # a real threshold exists between the two class probability clusters
    assert 0.2 < res["optimal_threshold"] < 0.8


def test_per_observation_cost_curve():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    cfn = np.array([500.0, 1000.0, 500.0, 1000.0])
    cfp = np.array([200.0, 1000.0, 200.0, 1000.0])
    res = expected_cost_curve_per_observation(y, p, cfn, cfp)
    assert res["optimal_cost_per_observation"] >= 0


# ---------------------------------------------------------------------------
# binning_woe round-trip
# ---------------------------------------------------------------------------

def _raw_synth(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    issue = pd.to_datetime("2010-01-01") + pd.to_timedelta(rng.randint(0, 1500, n), unit="D")
    earliest = issue - pd.to_timedelta(rng.randint(1, 300, n), unit="D")
    emp = rng.choice(["< 1 year", "2 years", "5 years", "10+ years"], n)
    return pd.DataFrame({
        "issue_d": issue,
        "earliest_cr_line": earliest,
        "emp_length": emp,
        "collections_12_mths_ex_med": rng.poisson(0.05, n),
        "delinq_2yrs": rng.poisson(0.2, n),
        "pub_rec": rng.poisson(0.05, n),
        "loan_amnt": rng.uniform(1000, 35000, n),
        "annual_inc": rng.uniform(20000, 150000, n),
        "open_acc": rng.randint(1, 20, n),
        "total_acc": rng.randint(5, 40, n),
        "term": rng.choice([" 36 months", " 60 months"], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "purpose": rng.choice(["credit_card", "debt_consolidation", "home_improvement",
                               "small_business", "other"], n),
        "revol_util": rng.uniform(0, 1.5, n),
        "revol_bal": rng.uniform(0, 30000, n),
        "dti": rng.uniform(0, 40, n),
        "inq_last_6mths": rng.poisson(0.5, n),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "target": rng.binomial(1, 0.15, n),
    })


@pytest.fixture(scope="module")
def binning_artifacts():
    pytest.importorskip("optbinning")
    from src.binning_woe import fit_binning
    return fit_binning(_raw_synth(), target_col="target")


def test_binning_roundtrip_woe(binning_artifacts):
    from src.binning_woe import transform_binning, ALGORITHMIC_FEATURES
    X = _raw_synth(n=200, seed=1)
    out = transform_binning(X, binning_artifacts, metric="woe")
    for col in ALGORITHMIC_FEATURES:
        assert f"{col}_woe" in out.columns
        assert np.isfinite(out[f"{col}_woe"]).all()


def test_binning_bin_labels(binning_artifacts):
    from src.binning_woe import transform_bin_labels, ALGORITHMIC_FEATURES
    X = _raw_synth(n=200, seed=2)
    labels = transform_bin_labels(X, binning_artifacts)
    assert list(labels.columns) == ALGORITHMIC_FEATURES
    assert len(labels) == len(X)
    # unseen categorical value maps to 'Unknown', never NaN
    X_unseen = X.copy()
    X_unseen.loc[X_unseen.index[0], "purpose"] = "never_seen_category"
    labels2 = transform_bin_labels(X_unseen, binning_artifacts)
    assert labels2.loc[X_unseen.index[0], "purpose"] == "Unknown"
