"""
stability.py

Population Stability Index (PSI) utilities for the credit risk pipeline.

Two complementary applications:

1. Characteristic-level PSI -- run on each already-binned/categorical
   feature (output of `transform_binning`) to diagnose WHICH variables are
   drifting between train/validation/test, or between development and a
   production scoring batch.

2. Score-level PSI -- run on the model's predicted probability (or point
   score), bucketed into deciles fitted on the training/development
   population via `QuantileBinner` (see binning.py) and re-applied
   statically to any comparison population. This is the metric typically
   monitored on a recurring cadence (e.g. monthly/quarterly) in production
   to trigger model review or redevelopment.

Interpretation thresholds follow standard scorecard-development practice
(Siddiqi, "Credit Risk Scorecards", 2006):
    PSI < 0.10            -> stable, no action required
    0.10 <= PSI < 0.25     -> moderate shift, investigate
    PSI >= 0.25            -> significant shift, recalibration/redevelopment
"""

from typing import List

import numpy as np
import pandas as pd


def calculate_psi(expected: pd.Series, actual: pd.Series) -> pd.DataFrame:
    """
    Compute per-bin and total PSI between a baseline and a comparison
    distribution of an already-binned/categorical (or discrete) feature.

    Parameters
    ----------
    expected : pd.Series
        Baseline / development population (typically the binned training
        column, e.g. `df_train_binned["annual_inc_group"]`).
    actual : pd.Series
        Comparison population (validation, test, or a production scoring
        batch), using the SAME bin categories as `expected` -- i.e. it
        must have been produced via the same fitted `QuantileBinner` /
        fixed-bin spec, not re-fitted independently.

    Returns
    -------
    pd.DataFrame
        Per-bin breakdown: ['bin', 'expected_pct', 'actual_pct', 'psi'].
        The scalar total is attached as `.attrs['psi_total']`.
    """
    eps = 1e-6  # guards against log(0) / division by zero on empty bins

    exp_dist = expected.value_counts(normalize=True)
    act_dist = actual.value_counts(normalize=True)

    all_bins = sorted(set(exp_dist.index) | set(act_dist.index), key=str)
    exp_pct = exp_dist.reindex(all_bins, fill_value=0).clip(lower=eps)
    act_pct = act_dist.reindex(all_bins, fill_value=0).clip(lower=eps)

    psi_per_bin = (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    result = pd.DataFrame(
        {
            "bin": all_bins,
            "expected_pct": exp_pct.values,
            "actual_pct": act_pct.values,
            "psi": psi_per_bin.values,
        }
    )
    result.attrs["psi_total"] = float(psi_per_bin.sum())
    return result


def psi_verdict(psi_total: float) -> str:
    """Map a scalar PSI value to the standard industry interpretation band."""
    if psi_total < 0.10:
        return "Stable"
    elif psi_total < 0.25:
        return "Moderate shift -- investigate"
    return "Significant shift -- recalibration/redevelopment likely required"


def psi_report(
    df_baseline: pd.DataFrame, df_comparison: pd.DataFrame, bin_columns: List[str]
) -> pd.DataFrame:
    """
    Batch characteristic-level PSI across multiple binned columns.

    Parameters
    ----------
    df_baseline : pd.DataFrame
        Development/training population (already passed through
        `transform_binning`).
    df_comparison : pd.DataFrame
        Validation, test, or production scoring batch (already passed
        through `transform_binning` using the SAME `BinningArtifacts` as
        `df_baseline`).
    bin_columns : list of str
        Names of the binned/categorical columns to evaluate, e.g.
        ['annual_inc_group', 'revol_util_group', 'inq_bin', ...].

    Returns
    -------
    pd.DataFrame
        One row per feature: ['feature', 'psi_total', 'verdict'], sorted
        descending by psi_total so the most-drifted features surface first.
    """
    rows = []
    for col in bin_columns:
        res = calculate_psi(df_baseline[col], df_comparison[col])
        rows.append(
            {
                "feature": col,
                "psi_total": res.attrs["psi_total"],
                "verdict": psi_verdict(res.attrs["psi_total"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("psi_total", ascending=False)
        .reset_index(drop=True)
    )