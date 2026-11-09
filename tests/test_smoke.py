"""Smoke tests for the src helpers the notebooks depend on."""
import numpy as np
import pandas as pd

from src.utils import missing_summary, split_input_output, calculate_all_iv


def test_missing_summary_counts():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1, 2, 3]})
    out = missing_summary(df)
    assert out.loc["a", "Missing Values"] == 1
    assert "b" not in out.index


def test_split_input_output():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    X, y = split_input_output(df, target_col="y")
    assert list(X.columns) == ["x"]
    assert list(y) == [0, 1, 0]


def test_calculate_all_iv_runs():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame({"feat": x, "target": y})
    iv = calculate_all_iv(df, target_col="target", num_bins=5)
    assert "feat" in iv["Feature"].values
    assert iv["Information Value (IV)"].iloc[0] > 0
