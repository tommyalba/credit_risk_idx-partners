# Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

# roc curve from sci-kit learn
from sklearn.metrics import roc_curve

"""
Expected-cost decision threshold for binary credit-risk classifiers.

Theoretical framework (Elkan, 2001; BCBS, 2006 for LGD/EAD components):

    E[C(tau)] = C_FN * P(y=1) * P(yhat=0 | y=1, tau)
              + C_FP * P(y=0) * P(yhat=1 | y=0, tau)

    where:
        P(yhat=0 | y=1, tau) = 1 - Recall_BAD(tau)      [False Negative Rate]
        P(yhat=1 | y=0, tau) = FPR(tau)                  [False Positive Rate]

    C_FN (cost of a false negative -- lending to a defaulting borrower)
        decomposes as:  C_FN ≈ EAD * LGD

    C_FP (cost of a false positive -- rejecting a genuinely good borrower)
        decomposes as the opportunity cost of foregone interest margin:
        C_FP ≈ EAD * interest_spread * loan_duration

    tau* = argmin_tau E[C(tau)]

Important epistemic note:
    C_FN and C_FP here are PORTFOLIO-LEVEL ESTIMATES, not per-individual
    certainties. This module provides two modes: (a) fixed scalar costs
    (constant C_FN/C_FP ratio across observations), and (b) per-observation
    costs when EAD varies across borrowers (e.g. different loan ceilings) --
    mode (b) is more realistic for heterogeneous consumer credit portfolios.
"""

# 1. ESTIMATING C_FN AND C_FP FROM EAD / LGD / SPREAD COMPONENTS
def estimate_cost_components(
    ead,
    lgd,
    interest_spread=None,
    loan_duration_years=None,
    fixed_cfp=None,
):
    """
    Derive C_FN and C_FP from primitive economic components.

    Parameters
    ----------
    ead : float or array-like
        Exposure at Default -- outstanding balance at default. Scalar for
        portfolio-level estimation, or an array of length n_observations
        when loan ceilings vary per individual.
    lgd : float or array-like
        Loss Given Default, the unrecovered share of EAD (0-1).
        Historical estimate: LGD = 1 - (recovery_present_value / EAD).
    interest_spread : float, optional
        Annual interest margin (e.g. 0.08 for an 8% spread), used to
        compute C_FP as opportunity cost. Required unless fixed_cfp
        is given.
    loan_duration_years : float, optional
        Estimated average loan duration in years.
    fixed_cfp : float, optional
        Alternative: supply C_FP directly (e.g. from an internal business
        unit estimate) without deriving it from spread*duration.

    Returns
    -------
    dict with C_FN, C_FP (scalar or array, following the type of ead), and
    the C_FN/C_FP ratio -- the ratio is what actually determines the
    position of tau*, not the absolute magnitude of C_FN and C_FP.
    """
    ead = np.asarray(ead, dtype=float)
    lgd = np.asarray(lgd, dtype=float)

    if np.any((lgd < 0) | (lgd > 1)):
        raise ValueError("LGD must be in the range [0, 1].")

    c_fn = ead * lgd

    if fixed_cfp is not None:
        c_fp = np.asarray(fixed_cfp, dtype=float)
    else:
        if interest_spread is None or loan_duration_years is None:
            raise ValueError(
                "Provide fixed_cfp, OR interest_spread + loan_duration_years."
            )
        c_fp = ead * interest_spread * loan_duration_years

    ratio = c_fn / c_fp if np.isscalar(c_fn) or c_fn.shape == () else np.divide(
        c_fn, c_fp, out=np.full_like(c_fn, np.nan), where=(c_fp != 0)
    )

    def _as_scalar_or_array(a):
        return float(a) if np.isscalar(a) or np.asarray(a).shape == () else np.asarray(a, dtype=float)

    return {
        'C_FN': _as_scalar_or_array(c_fn),
        'C_FP': _as_scalar_or_array(c_fp),
        'ratio_CFN_CFP': _as_scalar_or_array(ratio),
    }


# 2. EXPECTED COST -- SCALAR MODE (constant portfolio cost ratio)
def expected_cost_curve_scalar(y_true, y_proba, c_fn, c_fp, n_thresholds=1000):
    """
    Compute E[C(tau)] over all candidate thresholds, with C_FN and C_FP
    as fixed scalars applied uniformly across the portfolio.

    This is the most common simplification in practice: C_FN and C_FP
    estimated as portfolio averages (e.g. mean EAD x mean LGD) rather
    than per observation.

    Returns
    -------
    dict: thresholds, expected_costs, optimal_threshold, optimal_cost,
          plus fpr/fnr at the optimal point for interpretation.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    p1 = np.mean(y_true == 1)   # P(y=1), prevalensi BAD
    p0 = 1 - p1                 # P(y=0)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    y_bad = (y_true == 1)
    y_good = (y_true == 0)
    n_bad = int(y_bad.sum())
    n_good = int(y_good.sum())

    bad_scores = np.sort(y_proba[y_bad])
    good_scores = np.sort(y_proba[y_good])

    fn_arr = np.searchsorted(bad_scores, thresholds, side='left')
    fp_arr = n_good - np.searchsorted(good_scores, thresholds, side='left')

    fnr_arr = fn_arr / n_bad if n_bad > 0 else np.zeros_like(thresholds, dtype=float)
    fpr_arr = fp_arr / n_good if n_good > 0 else np.zeros_like(thresholds, dtype=float)

    expected_costs = c_fn * p1 * fnr_arr + c_fp * p0 * fpr_arr
    optimal_idx = np.argmin(expected_costs)

    return {
        'thresholds': thresholds,
        'expected_costs': expected_costs,
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_cost': expected_costs[optimal_idx],
        'optimal_fnr': fnr_arr[optimal_idx],
        'optimal_fpr': fpr_arr[optimal_idx],
        'optimal_recall_bad': 1 - fnr_arr[optimal_idx],
        'optimal_specificity': 1 - fpr_arr[optimal_idx],
        'cost_ratio_used': c_fn / c_fp,
    }


# 3. EXPECTED COST -- PER-OBSERVATION MODE (heterogeneous EAD)

def expected_cost_curve_per_observation(
    y_true, y_proba, c_fn_array, c_fp_array, n_thresholds=1000
):
    """
    A more realistic version for heterogeneous consumer credit portfolios: 
    each observation carries its own `C_FN_i` and `C_FP_i` (i.e. `EAD_i` 
    differs because loan ceilings vary across borrowers), instead of 
    a single portfolio average.

        E[C(tau)] = (1/n) * sum_i [ C_FN_i * 1{y_i=1, yhat_i=0}
                                   + C_FP_i * 1{y_i=0, yhat_i=1} ]

    This is the nominal expected financial loss amount (not a 
    proportion/rate as in scalar mode), so the `C(tau)` scale here is 
    directly interpretable in currency units per average observation -- 
    more informative for reporting to business units.

    Parameters
    ----------
    c_fn_array, c_fp_array : array-like, with n_observations in length
        Individual costs per observation, e.g., the result of
        `estimate_cost_components()` with `EAD` as an array.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    c_fn_array = np.asarray(c_fn_array, dtype=float)
    c_fp_array = np.asarray(c_fp_array, dtype=float)

    n = len(y_true)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    y_bad = (y_true == 1)
    y_good = (y_true == 0)

    # Sort bad and good loans by probability to compute threshold costs via searchsorted
    order_bad = np.argsort(y_proba[y_bad])
    bad_scores = y_proba[y_bad][order_bad]
    bad_costs_cum = np.concatenate([[0.0], np.cumsum(c_fn_array[y_bad][order_bad])])

    order_good = np.argsort(y_proba[y_good])
    good_scores = y_proba[y_good][order_good]
    good_costs = c_fp_array[y_good][order_good]
    good_costs_cum = np.concatenate([[0.0], np.cumsum(good_costs)])
    total_good_cost = good_costs_cum[-1] if len(good_costs_cum) > 0 else 0.0

    idx_bad = np.searchsorted(bad_scores, thresholds, side='left')
    idx_good = np.searchsorted(good_scores, thresholds, side='left')

    fn_costs = bad_costs_cum[idx_bad]
    fp_costs = total_good_cost - good_costs_cum[idx_good]

    expected_costs = (fn_costs + fp_costs) / n if n > 0 else np.zeros_like(thresholds)
    optimal_idx = np.argmin(expected_costs)

    return {
        'thresholds': thresholds,
        'expected_costs': expected_costs,
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_cost_per_observation': expected_costs[optimal_idx],
    }


# 4. COMPARING TAU* ACROSS CRITERIA (Cost vs Youden vs F1)

def compare_threshold_criteria(y_true, y_proba, c_fn, c_fp):
    """
    Comparing the tau* generated by three different criteria on the same data, 
    to diagnose how far the statistical threshold (Youden/F1) that has been 
    used deviates from the economic-cost-optimal threshold.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Criterion 1: Expected Cost
    cost_result = expected_cost_curve_scalar(y_true, y_proba, c_fn, c_fp)
    tau_cost = cost_result['optimal_threshold']

    # Criterion 2: Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    tau_youden = roc_thresholds[np.argmax(j_scores)]

    # Criterion 3: F1-maximal (PR curve)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    precisions_t, recalls_t = precisions[:-1], recalls[:-1]
    denom = precisions_t + recalls_t
    f1_scores = np.zeros_like(denom, dtype=float)
    np.divide(2 * precisions_t * recalls_t, denom, out=f1_scores,
              where=denom > 0)
    tau_f1 = pr_thresholds[np.argmax(f1_scores)]

    return {
        'tau_expected_cost': round(float(tau_cost), 4),
        'tau_youden_j': round(float(tau_youden), 4),
        'tau_f1_maximal': round(float(tau_f1), 4),
        'cost_ratio_CFN_CFP': round(c_fn / c_fp, 2),
        'interpretation': (
            f"With a C_FN/C_FP ratio of {c_fn/c_fp:.4f}, the cost-optimal "
            f"threshold is {'below' if tau_cost < tau_youden else 'above'} "
            f"the Youden threshold -- difference {abs(tau_cost - tau_youden):.4f} "
            f"on the probability scale."
        ),
    }


# 5. EXPECTED-COST CURVE VISUALIZATION

def plot_expected_cost_curve(cost_result, dataset_name='train_cal', ax=None):
    """Plot E[C(tau)] against tau, marking the optimal point."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        cost_result['thresholds'], cost_result['expected_costs'],
        color='#8e44ad', linewidth=2, label='E[C(τ)]'
    )
    ax.axvline(
        cost_result['optimal_threshold'], color='black', linestyle="--",
        linewidth=1.2,
        label=f"τ* = {cost_result['optimal_threshold']:.4f}"
    )
    ax.scatter(
        [cost_result['optimal_threshold']],
        [cost_result['expected_costs'][np.argmin(cost_result['expected_costs'])]],
        color='black', zorder=5, s=40
    )

    ax.set_xlabel('Decision Threshold (τ)')
    ax.set_ylabel('Expected Cost  E[C(τ)]')
    ax.set_title(f"Kurva Expected Cost — {dataset_name}")
    ax.set_xlim(0, 1)
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(alpha=0.25)
    return ax
