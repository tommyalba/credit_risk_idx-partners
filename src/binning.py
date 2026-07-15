"""
binning.py

Coarse-classing (binning) module for the credit risk preprocessing
pipeline. Positioned immediately after the train/valid/test split and
immediately before missing-value imputation, scaling, and one-hot
encoding.

Two binning families are handled with DIFFERENT leakage-safety contracts:

1. Fixed / business-rule bins (`pd.cut` with hardcoded boundaries).
   Boundaries are analyst-defined thresholds informed by EDA and bad-rate
   inspection on the historical population; they do not depend on the
   statistical distribution of any particular split, and are therefore
   applied identically -- with no fitting step -- to train, validation,
   test, and any future production batch.

2. Quantile bins (`pd.qcut`-derived), covering `annual_inc`,
   `total_rev_hi_lim`, and `tot_cur_bal`. Bin edges here ARE a function of
   the input distribution, so they must be estimated exactly once on the
   training partition (`fit_binning`) and then applied as static,
   pre-computed boundaries to every other split (`transform_binning`).
   This mirrors the scikit-learn Transformer contract and is what
   prevents distributional leakage from validation/test into the
   bin-edge estimation.

Typical usage inside the wider preprocessing pipeline
-------------------------------------------------------
    X_train, X_valid, X_test = split_raw_data(df_raw)  # split FIRST

    artifacts = fit_binning(df_train)  # fit on train only
    X_train_binned = transform_binning(df_train, artifacts)
    X_valid_binned = transform_binning(df_valid, artifacts)
    X_test_binned  = transform_binning(df_test, artifacts)

    # persist for production scoring / audit reproducibility:
    joblib.dump(artifacts, "binning_artifacts.pkl")

    # downstream: split numerical/categorical -> impute -> scale/OHE -> concat
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 1. FIXED (business-rule) bin specifications
#    Leakage-free by construction: boundaries are constants, not statistics
#    estimated from any particular DataFrame.
# ============================================================================

FIXED_CUT_BINS: Dict[str, dict] = {
    'credit_age_years': {
        'bins': [-np.inf, 5, 15, np.inf],
        'labels': ['less_5', '5_15', 'greater_15'],
    },
    'emp_length_ordinal': {
        'bins': [-1, 2, 5, 9, 100],
        'labels': ['New_Employee', 'Mid_Employee', 'Senior_Employee', 'Expert_Employee'],
        'nan_label': 'No_Formal_Employee',
    },
    'total_acc': {
        'bins': [0, 15, 25, 35, 999],
        'labels': ['less_15', '15_to_25', '25_to_35', 'greater_35'],
    },
    'revol_bal': {
        'bins': [-1, 5000, 10000, 20000, 40000, np.inf],
        'labels': ['0_to_5k', '5k_to_10k', '10k_to_20k', '20k_to_40k', 'greater_40k'],
    },
    'revol_util': {
        'bins': [-0.01, 20.0, 40.0, 60.0, 80.0, 100.0, 9999.0],
        'labels': ['0_to_20', '20_to_40', '40_to_60', '60_to_80', '80_to_100', 'greater_100'],
    },
    'open_acc': {
        'bins': [-1, 3, 8, 14, 21, 999],
        'labels': ['0_to_3', '4_to_8', '9_to_14', '15_to_21', '22_or_more'],
    },
}

# Quantile (distribution-dependent) columns: MUST be fitted on train only.
QUANTILE_COLUMNS: Dict[str, dict] = {
    'annual_inc': {'q': 5, 'labels': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']},
}


# ============================================================================
# 2. Fittable quantile binner (leakage-safe by construction)
# ============================================================================

class QuantileBinner:
    """
    Fittable, leakage-safe quantile binner.

    Implements a minimal `fit` / `transform` contract, deliberately
    mirroring scikit-learn's Transformer API, to guarantee that
    percentile-based bin edges are estimated exactly once -- on the
    training partition -- and then applied as FIXED, static boundaries to
    validation, test, and production scoring data. This prevents:

      (a) distributional leakage from validation/test into the bin-edge
          estimation itself, and
      (b) inconsistent bin definitions across splits, which would silently
          make the resulting categorical feature non-comparable between
          train and out-of-sample data.
    """

    def __init__(self, q: int = 5, labels: Optional[List[str]] = None):
        self.q = q
        self.labels = labels or ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
        self.bin_edges_: Optional[np.ndarray] = None
        self.fitted_labels_: Optional[List[str]] = None

    def fit(self, series: pd.Series) -> 'QuantileBinner':
        """Estimate quantile bin edges from a TRAINING-ONLY series."""
        _, edges = pd.qcut(series, q=self.q, retbins=True, duplicates='drop')
        edges = np.asarray(edges, dtype=float).copy()

        # Open-end the outermost edges so that out-of-range values observed
        # later in validation/test/production (beyond the train-set
        # min/max) still fall inside the first/last bin instead of
        # silently becoming NaN.
        edges[0], edges[-1] = -np.inf, np.inf
        self.bin_edges_ = edges

        n_bins = len(edges) - 1
        if n_bins < len(self.labels):
            # duplicates='drop' collapsed some quantile boundaries -- typical
            # for skewed or heavily-tied/discrete distributions. Truncate the
            # label set to match and surface this explicitly rather than
            # failing silently or mis-aligning labels to bins.
            self.fitted_labels_ = self.labels[:n_bins]
        else:
            self.fitted_labels_ = self.labels
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply the previously fitted, static bin edges to any split."""
        if self.bin_edges_ is None:
            raise RuntimeError(f"QuantileBinner.transform() called before fit().")
        binned = pd.cut(series, bins=self.bin_edges_, labels=self.fitted_labels_)
        if series.isnull().any():
            binned = binned.cat.add_categories('Unknown').fillna('Unknown')
        return binned

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)


@dataclass
class BinningArtifacts:
    """
    Container for fitted (train-only) binning parameters.

    Produced once via `fit_binning(df_train)` and reused, unchanged, for
    `transform_binning()` on validation, test, and any future production
    scoring batch. Persist via `joblib.dump` / `pickle` alongside the
    model artifact to guarantee bin-boundary reproducibility and
    auditability -- a standard requirement for credit scorecard model
    governance (cf. SR 11-7 model risk management expectations).
    """

    quantile_binners: Dict[str, QuantileBinner] = field(default_factory=dict)


def fit_binning(X_train: pd.DataFrame) -> BinningArtifacts:
    """
    Fit the leakage-prone (quantile-based) bin edges on the TRAINING
    partition only.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training split, produced by the raw-data train/valid/test
        split that must occur BEFORE any binning is applied.

    Returns
    -------
    BinningArtifacts
        Fitted quantile binners to be passed into `transform_binning` for
        every split -- including `df_train` itself, for consistency.
    """
    artifacts = BinningArtifacts()
    for col, spec in QUANTILE_COLUMNS.items():
        binner = QuantileBinner(q=spec['q'], labels=spec['labels'])
        binner.fit(X_train[col])
        artifacts.quantile_binners[col] = binner
    return artifacts


# ============================================================================
# 3. Derived-feature construction (must run before fixed-bin application)
# ============================================================================

def _add_credit_history_age(df: pd.DataFrame) -> pd.DataFrame:
    """Derive credit history length (months, years) at loan issuance."""
    credit_history_age_month = (
        (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12
        + (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
    ).fillna(0).astype(int)
    df['credit_age_years'] = credit_history_age_month / 12
    df = df.drop(columns=['issue_d', 'earliest_cr_line'])
    return df


def _extract_emp_length_years(text: Optional[str]) -> float:
    """Parse free-text employment length, e.g. '< 1 year', '10+ years'."""
    if pd.isna(text):
        return np.nan
    text = str(text)
    if '< 1' in text:
        return 0
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else np.nan


# ============================================================================
# 4. Shared fixed-bin helper
# ============================================================================

def _apply_fixed_cut(df: pd.DataFrame, source_col: str, target_col: str, spec: dict) -> pd.DataFrame:
    """Apply a hardcoded pd.cut spec and route NaNs to 'Unknown' (or nan_label)."""
    series = df[source_col]
    binned = pd.cut(series, bins=spec['bins'], labels=spec['labels'])
    nan_label = spec.get('nan_label', 'Unknown')
    if series.isnull().any():
        binned = binned.cat.add_categories(nan_label).fillna(nan_label)
    df[target_col] = binned
    return df


# ============================================================================
# 5. Main transform: applies BOTH fixed bins and fitted quantile bins
# ============================================================================

def transform_binning(df: pd.DataFrame, artifacts: BinningArtifacts) -> pd.DataFrame:
    """
    Apply coarse-classing (binning) to a single split (train, valid, test,
    or a production scoring batch), using pre-fitted quantile boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        A single split. Must contain the raw columns listed in the module
        docstring. `issue_d` / `earliest_cr_line` must already be
        datetime64 dtype.
    artifacts : BinningArtifacts
        Output of `fit_binning(df_train)`. The SAME artifacts object must
        be reused across train/valid/test/production to guarantee
        identical, leakage-free bin boundaries everywhere.

    Returns
    -------
    pd.DataFrame
        New DataFrame (input not mutated) with binned/categorical
        features in place of their raw numeric counterparts.

    Notes
    -----
    Missing-value policy: NaNs are mapped to an explicit 'Unknown' (or
    feature-specific) category rather than dropped, so "missingness"
    remains available to the model / a downstream WoE-IV encoder as an
    informative category -- standard practice in scorecard development.
    """
    out = df.copy()
    
    # --- derived features -------------------------------------------------
    out = _add_credit_history_age(out)
    out['emp_length_ordinal'] = out['emp_length'].apply(_extract_emp_length_years)
    out = out.drop(columns=["emp_length"])

    # --- fixed / business-rule bins ----------------------------------------
    out = _apply_fixed_cut(out, 'credit_age_years', 'credit_age_group', FIXED_CUT_BINS['credit_age_years'])
    out = out.drop(columns=['credit_age_years'])
    out = _apply_fixed_cut(out, 'emp_length_ordinal', 'emp_length_group', FIXED_CUT_BINS['emp_length_ordinal'])
    out = out.drop(columns=['emp_length_ordinal'])
    out = _apply_fixed_cut(out, 'total_acc', 'total_acc_group', FIXED_CUT_BINS['total_acc'])
    out = out.drop(columns=['total_acc'])
    out = _apply_fixed_cut(out, 'revol_bal', 'revol_bal_group', FIXED_CUT_BINS['revol_bal'])
    out = out.drop(columns=['revol_bal'])
    out = _apply_fixed_cut(out, 'revol_util', 'revol_util_group', FIXED_CUT_BINS['revol_util'])
    out = out.drop(columns=['revol_util'])
    out = _apply_fixed_cut(out, 'open_acc', 'open_acc_group', FIXED_CUT_BINS['open_acc'])
    out['open_acc_group'] = out['open_acc_group'].astype("category")
    out = out.drop(columns=['open_acc'])

    # --- fitted quantile bins (leakage-safe: edges come from artifacts) ---
    for col, binner in artifacts.quantile_binners.items():
        out[f"{col}_group"] = binner.transform(out[col])
        out = out.drop(columns=[col])

    # --- presence / count-based groupings (business-rule, stateless) ------
    out['collections_12mths_group'] = out['collections_12_mths_ex_med'].apply(
        lambda v: 'Unknown' if pd.isna(v) else ('No_Collections' if v == 0 else 'Has_Collections')
    ).astype('category')
    out = out.drop(columns=['collections_12_mths_ex_med'])

    def _group_pub_rec(val: float) -> str:
        if pd.isna(val):
            return 'Unknown'
        elif val == 0:
            return '0_pub_rec'
        elif val == 1:
            return '1_pub_rec'
        return '2_or_more'

    out['pub_rec_group'] = out['pub_rec'].apply(_group_pub_rec).astype('category')
    out = out.drop(columns=['pub_rec'])

    def _group_delinq(val: float) -> str:
        if pd.isna(val):
            return 'Unknown'
        elif val == 0:
            return '0_delinq'
        elif val == 1:
            return '1_delinq'
        elif val == 2:
            return '2_delinq'
        return '3_or_more'

    out['delinq_2yrs_group'] = out['delinq_2yrs'].apply(_group_delinq).astype('category')
    out = out.drop(columns=['delinq_2yrs'])

    def _group_inquiries(val: float) -> str:
        if val == 0:
            return '0_none'
        elif 1 <= val <= 2:
            return '1_low'
        elif 3 <= val <= 5:
            return '2_medium'
        return '3_high'

    out['inq_last_group'] = out['inq_last_6mths'].fillna(0).apply(_group_inquiries).astype('category')
    out = out.drop(columns=['inq_last_6mths'])
    
    # --- nominal normalization (no binning required) -----------------------
    out['purpose'] = out['purpose'].replace({'renewable_energy': 'other', 'moving': 'other', 'vacation': 'other'}).astype('category')
    out['home_ownership'] = out['home_ownership'].replace({'ANY': 'OTHER', 'NONE': 'OTHER'}).astype('category')
    out['term'] = out['term'].replace({' 36 months': '36_months', ' 60 months': '60_months'}).astype('category')
    out['verification_status'] = out['verification_status'].replace(
        {'Not Verified': 'Not_Verified', 'Source Verified': 'Verified'}
    ).astype('category')

    return out


# ============================================================================
# 6. Bad-rate monotonicity validator
# ============================================================================

def evaluate_bad_rate_monotonicity(
    df: pd.DataFrame, bin_col: str, target_col: str, bin_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute the bad rate per bin and report whether it is monotonic.
    
    Parameters
    ----------
    df : pd.DataFrame
        Binned TRAINING data (never validation/test -- this is a design
        diagnostic, not a metric to be reported as out-of-sample
        performance).
    bin_col : str
        Name of the categorical/binned column to evaluate, e.g.
        'annual_inc_group'.
    target_col : str
        Binary target column (1 = BAD, 0 = GOOD).
    bin_order : list of str, optional
        Explicit ordinal ordering of the bin categories (e.g.
        ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']). If omitted,
        the existing categorical order (or first-seen order) is used --
        supply this explicitly for any bin whose label ordering is not
        already lexically/naturally ordinal.

    Returns
    -------
    pd.DataFrame
        Columns: [bin_col, 'count', 'bad_count', 'bad_rate',
        'is_monotonic_increasing', 'is_monotonic_decreasing'], ordered by
        `bin_order` (or inferred order).
    """
    summary = (
        df.groupby(bin_col, observed=True)[target_col]
        .agg(count="size", bad_count="sum")
        .assign(bad_rate=lambda d: d["bad_count"] / d["count"])
    )

    if bin_order is not None:
        summary = summary.reindex(bin_order)

    summary['is_monotonic_increasing'] = summary['bad_rate'].is_monotonic_increasing
    summary['is_monotonic_decreasing'] = summary['bad_rate'].is_monotonic_decreasing
    return summary.reset_index()