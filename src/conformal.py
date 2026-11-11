"""Conformal prediction intervals for binary probabilities.

Venn-Abers conformal inference: for each scored observation, fit the
calibration map twice — once with the candidate label forced to 0 and
once forced to 1 — and the two resulting probability estimates bracket
a valid predictive probability. Under exchangeability between the
calibration set and the scored population, the interval

    [min(p0, p1), max(p0, p1)]

contains a well-calibrated probability for the observation with
marginal coverage guarantees, without distributional assumptions.

This is the right tool for "how uncertain is this PD?" near decision
thresholds: a narrow interval around tau is a different decision than a
wide one. It is NOT a confidence interval on the metric — see
src.metrics.bootstrap_metric_ci for population-level uncertainty.

Cost: two isotonic fits per scored row. Fine for application batches
(hundreds/thousands of rows); subsample large evaluation frames.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def venn_abers_intervals(
    p_cal: np.ndarray,
    y_cal: np.ndarray,
    p_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Venn-Abers intervals for binary outcomes.

    Parameters
    ----------
    p_cal : model scores on the calibration set (any monotone score;
        isotonic recalibrates it internally).
    y_cal : binary labels for the calibration set.
    p_test : scores to explain.

    Returns
    -------
    (lower, upper, midpoint): per-observation interval bounds and the
    Venn-Abers point predictor (p0 + p1) / 2.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    p_test = np.asarray(p_test, dtype=float)
    n_test = len(p_test)

    lo = np.empty(n_test)
    hi = np.empty(n_test)
    mid = np.empty(n_test)

    for i, s in enumerate(p_test):
        # candidate label 0
        x0 = np.append(p_cal, s)
        y0 = np.append(y_cal, 0)
        iso0 = IsotonicRegression(out_of_bounds='clip')
        iso0.fit(x0, y0)
        p0 = float(iso0.predict([s])[0])

        # candidate label 1
        x1 = np.append(p_cal, s)
        y1 = np.append(y_cal, 1)
        iso1 = IsotonicRegression(out_of_bounds='clip')
        iso1.fit(x1, y1)
        p1 = float(iso1.predict([s])[0])

        lo[i], hi[i] = min(p0, p1), max(p0, p1)
        mid[i] = 0.5 * (p0 + p1)

    return lo, hi, mid


def interval_summary(lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame:
    """Quick descriptive stats for a batch of intervals."""
    width = hi - lo
    return pd.DataFrame({
        'n': len(width),
        'width_mean': float(width.mean()),
        'width_median': float(np.median(width)),
        'width_p90': float(np.quantile(width, 0.9)),
        'frac_touching_bounds': float(np.mean((lo <= 1e-9) | (hi >= 1 - 1e-9))),
    }, index=[0])
