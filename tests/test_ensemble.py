import numpy as np
import pandas as pd
import joblib

from src.ensemble import AverageProbabilityEnsemble
from src.expected_cost_threshold import (
    estimate_cost_components,
    expected_cost_curve_per_observation,
)


class _ConstProba:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def test_ensemble_weighted_average():
    ens = AverageProbabilityEnsemble([_ConstProba(0.2), _ConstProba(0.6)], weights=[1, 3])
    proba = ens.predict_proba(pd.DataFrame({"a": [1, 2]}))
    assert np.allclose(proba[:, 1], 0.5)


def test_ensemble_pickle_roundtrip(tmp_path):
    ens = AverageProbabilityEnsemble([_ConstProba(0.3), _ConstProba(0.7)])
    path = tmp_path / "ens.joblib"
    joblib.dump(ens, path)
    loaded = joblib.load(path)
    X = pd.DataFrame({"a": [1]})
    assert np.allclose(loaded.predict_proba(X), ens.predict_proba(X))


def test_cost_components_array_mode():
    out = estimate_cost_components(
        ead=np.array([1000.0, 2000.0]),
        lgd=0.5,
        interest_spread=0.1,
        loan_duration_years=np.array([2.0, 5.0]),
    )
    assert np.allclose(out["C_FN"], [500.0, 1000.0])
    assert np.allclose(out["C_FP"], [200.0, 1000.0])


def test_per_observation_cost_curve():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    cfn = np.array([500.0, 1000.0, 500.0, 1000.0])
    cfp = np.array([200.0, 1000.0, 200.0, 1000.0])
    res = expected_cost_curve_per_observation(y, p, cfn, cfp)
    assert 0.0 <= res["optimal_threshold"] <= 1.0
    assert res["optimal_cost_per_observation"] >= 0
