"""
binning_v3.py

Hybrid coarse-classing module.

    ALGORITHMIC (optbinning.BinningProcess, monotonic_trend set EXPLICITLY):
        annual_inc, revol_util, purpose, home_ownership

    MANUAL / BUSINESS-RULE (pd.cut with hardcoded, human-defined boundaries):
        credit_age_years, emp_length_ordinal, collections_12_mths_ex_med,
        pub_rec, delinq_2yrs, total_acc, revol_bal, open_acc, inq_last_6mths
    
    EXCLUDED (dropped upstream, before this module receives the data):
        tot_coll_amt, tot_cur_bal, total_rev_hi_lim
        -> 100% missing for issue_d in 2007-2011, populated from 2012+.
           A reporting-schema artifact tied to loan vintage, not a risk
           signal; confirmed via characteristic-level PSI (train vs. OOT
           test ~4.0) and direct missing-rate inspection by vintage.

Rationale for the algorithmic / manual split
----------------------------------------------
`annual_inc` and `revol_util` are continuous with a genuine, data-driven
optimal cutpoint search worth running. `purpose` (11 categories) and
`home_ownership` (4 categories) benefit from optimal category-group
merging, which is not a decision a human should hand-pick by inspection.

The remaining features are low-cardinality integer counts (or already-
simple continuous scales) whose natural grouping is unambiguous from
domain logic alone (e.g. `delinq_2yrs`: 0 / 1 / 2 / 3+) -- searching for
a "statistically optimal" split here adds solver-dependent complexity
without a corresponding gain, and keeping the boundaries as literal
constants in code keeps the audit trail direct: a reviewer can verify
the boundary without re-deriving it from a fitted object.

On `status == "OPTIMAL"` and `monotonic_trend`
-------------------------------------------------
`BinningProcess.summary()["status"] == "OPTIMAL"` reports SOLVER
CONVERGENCE only -- the constrained-optimization problem was solved to
global optimality under the requested constraints. It does NOT certify
that the resulting bins are monotonic in the simple ascending/descending
sense.

`monotonic_trend="auto"` does not mean "find a monotonic trend
automatically" -- it means "search across ALL supported trend shapes,
including 'peak' and 'valley' (which are, by definition, NOT monotonic),
and pick whichever maximizes IV". This is why a manual bad-rate audit can
find violations even on features whose solver status reads "OPTIMAL".
To guarantee genuine monotonicity, `monotonic_trend` is set EXPLICITLY
below ("descending" for annual_inc, "ascending" for revol_util) rather
than left as "auto".

Usage
-----
    X_train, X_valid, X_test_oot = split_raw_data(df_raw)

    binning_artifacts = fit_binning(X_train, target_col='target')

    X_train = transform_binning(X_train, binning_artifacts)
    X_valid = transform_binning(X_valid, binning_artifacts)
    X_test  = transform_binning(X_test_oot, binning_artifacts)

    # Bad-rate monotonicity audit on any binned column (train ONLY --
    this is a design diagnostic, not an out-of-sample metric):
    evaluate_bad_rate_monotonicity(
        X_train_binned.assign(y=y_train),
        bin_col='annual_inc_bin', target_col='y',
    )
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from optbinning import BinningProcess


# ============================================================================
# 1. Feature routing
# ============================================================================

ALGORITHMIC_NUMERICAL: List[str] = [
    'annual_inc', 'dti', 'loan_to_income', 'revol_util', 'revol_bal_to_income',
    'open_acc_ratio', 'inq_last_6mths',
]
ALGORITHMIC_CATEGORICAL: List[str] = ['purpose', 'home_ownership']
ALGORITHMIC_FEATURES: List[str] = ALGORITHMIC_NUMERICAL + ALGORITHMIC_CATEGORICAL

MANUAL_FEATURES: List[str] = [
    'credit_age_years', 'emp_length_ordinal', 'total_acc'
]

PASSTHROUGH_CATEGORICAL_FEATURES: List[str] = ['term', 'verification_status']

# Defensive only: these should already be absent from the input DataFrame
# (dropped upstream). Referenced here solely so `transform_binning` can
# drop them with `errors="ignore"` if they slip through unexpectedly.
EXCLUDED_FEATURES: List[str] = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

ALGORITHMIC_FIT_PARAMS: Dict[str, dict] = {
    'annual_inc': dict(max_n_bins=10, min_bin_size=0.05, monotonic_trend='descending'),
    'dti': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'inq_last_6mths': dict(max_n_bins=4, min_bin_size=0.05, monotonic_trend='ascending'),
    'loan_to_income': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'revol_bal_to_income': dict(max_n_bins=8, min_bin_size=0.05, monotonic_trend='ascending'),
    'revol_util': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'open_acc_ratio': dict(max_n_bins=7, min_bin_size=0.05, monotonic_trend='ascending'),
    # Nominal categories have no inherent numeric order, so 'auto' is
    # appropriate here: categories are internally sorted by event rate
    # before optimal merging, so the resulting groups are monotonic along
    # THAT induced ordering by construction -- unlike the numerical if
    # 'peak'/'valley' ambiguity described above.
    'purpose': dict(max_n_bins=5, min_bin_size=0.02, monotonic_trend='auto', cat_cutoff=0.02),
    'home_ownership': dict(max_n_bins=3, min_bin_size=0.03, monotonic_trend='auto'),
}

# Manual fixed-boundary specs (pd.cut). Business-rule, leakage-free by
# construction: boundaries are constants, not statistics of any split.
FIXED_CUT_BINS: Dict[str, dict] = {
    'credit_age_years': {
        'bins': [-np.inf, 5, 15, np.inf],
        'labels': ['less_5', '5_15', 'greater_15'],
    },
    'emp_length_ordinal': {
        'bins': [-1, 2, 5, 9, 100],
        'labels': ['New_Employee', 'Mid_Employee', 'Senior_Employee', 'Expert_Employee'],
        'nan_label': 'No_Formal_Employee',  # NaN = no formal employment reported
    },
    'total_acc': {
        'bins': [0, 10, 20, 30, 40, 999],
        'labels': ['less_10', '10_to_20', '20_to_30', '30_to_40', 'greater_40'],
    },
    # 'revol_bal': {
    #     'bins': [-1, 6000, 12000, np.inf],
    #     'labels': ['0_to_6k', '6k_to_12k', 'greater_12k'],
    # }, 
    # 'open_acc': {
    #     'bins': [-1, 6, 12, 21, 999],
    #     'labels': ['0_to_6', '7_to_12', '13_to_21', '22_or_more'],
    # },
}


# ============================================================================
# 2. Derived-feature construction
# ============================================================================

def _extract_emp_length_years(text: Optional[str]) -> float:
    """Parse free-text employment length, e.g. '< 1 year', '10+ years'."""
    if pd.isna(text):
        return np.nan
    text = str(text)
    if '< 1' in text:
        return 0
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else np.nan


def _add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Construct credit_age_years and emp_length_ordinal from raw columns."""
    credit_history_age_month = (
        (X['issue_d'].dt.year - X['earliest_cr_line'].dt.year) * 12
        + (X['issue_d'].dt.month - X['earliest_cr_line'].dt.month)
    ).fillna(0).astype(int)
    X['credit_age_years'] = credit_history_age_month / 12

    X = X.drop(columns=['issue_d', 'earliest_cr_line',])
    
    X['emp_length_ordinal'] = X['emp_length'].apply(_extract_emp_length_years)
    return X


# ============================================================================
# 3. Manual binning helpers
# ============================================================================

def _apply_fixed_cut(series: pd.Series, spec: dict) -> pd.Series:
    """Apply a hardcoded pd.cut spec and route NaNs to 'Unknown' (or nan_label)."""
    binned = pd.cut(series, bins=spec['bins'], labels=spec['labels'])
    nan_label = spec.get('nan_label', 'Unknown')
    if series.isnull().any():
        binned = binned.cat.add_categories(nan_label).fillna(nan_label)
    return binned


# def _group_pub_rec(val: float) -> str:
#     if pd.isna(val):
#         return 'Unknown'
#     elif val == 0:
#         return '0_pub_rec'
#     elif val == 1:
#         return '1_pub_rec'
#     return '2_or_more'


# def _group_delinq(val: float) -> str:
#     if pd.isna(val):
#         return 'Unknown'
#     elif val == 0:
#         return '0_delinq'
#     elif val == 1:
#         return '1_delinq'
#     elif val == 2:
#         return '2_delinq'
#     return '3_or_more'


# def _group_inquiries(val: float) -> str:
#     if val == 0:
#         return '0_none'
#     elif 1 <= val <= 2:
#         return '1_low'
#     elif 3 <= val <= 5:
#         return '2_medium'
#     return '3_high'


def _apply_manual_binning(X: pd.DataFrame) -> pd.DataFrame:
    """Apply every MANUAL_FEATURES transform; drops consumed raw columns."""
    out = X.copy()

    out['credit_age_group'] = _apply_fixed_cut(X['credit_age_years'], FIXED_CUT_BINS['credit_age_years'])
    out = out.drop(columns=['credit_age_years'])

    out['emp_length_group'] = _apply_fixed_cut(X['emp_length_ordinal'], FIXED_CUT_BINS['emp_length_ordinal'])
    out = out.drop(columns=['emp_length_ordinal', 'emp_length'])

    out['total_acc_group'] = _apply_fixed_cut(X['total_acc'], FIXED_CUT_BINS['total_acc'])
    out = out.drop(columns=['total_acc'])
    
    # out['inq_last_group'] = X['inq_last_6mths'].fillna(0).apply(_group_inquiries).astype('category')
    # out = out.drop(columns=['inq_last_6mths'])
    
    condition = (
        (X['delinq_2yrs'] > 0) | 
        (X['pub_rec'] > 0) | 
        (X['collections_12_mths_ex_med'] > 0)
    )

    X['has_derogatory'] = np.where(condition, 1, 0)
    out['has_derogatory'] = X['has_derogatory'].apply(
        lambda v: 'Unknown' if pd.isna(v) else ('No_Derogatory' if v == 0 else 'Has_Derogatory')
    ).astype('category')
    out = out.drop(columns=['delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med'])
    
    # out['revol_bal_group'] = _apply_fixed_cut(out['revol_bal'], FIXED_CUT_BINS['revol_bal'])
    # out = out.drop(columns=['revol_bal'])
    
    # out['open_acc_group'] = _apply_fixed_cut(out['open_acc'], FIXED_CUT_BINS['open_acc']).astype('category')
    # out = out.drop(columns=['open_acc'])

    # out['collections_12mths_group'] = out['collections_12_mths_ex_med'].apply(
    #     lambda v: 'Unknown' if pd.isna(v) else ('No_Collections' if v == 0 else 'Has_Collections')
    # ).astype('category')
    # out = out.drop(columns=['collections_12_mths_ex_med'])

    # out['pub_rec_group'] = out['pub_rec'].apply(_group_pub_rec).astype('category')
    # out = out.drop(columns=['pub_rec'])
    
    # out['delinq_2yrs_group'] = out['delinq_2yrs'].apply(_group_delinq).astype('category')
    # out = out.drop(columns=['delinq_2yrs'])
    
    return out


# ============================================================================
# 4. Fit / transform contract
# ============================================================================

@dataclass
class BinningArtifacts:
    """
    Container for the fitted `BinningProcess` (algorithmic features only).

    Manual features need no fitted state -- their boundaries are
    constants baked into `FIXED_CUT_BINS` / the grouping functions above,
    so they are leakage-free and identical across train/valid/test by
    construction. Persist this object (e.g. `joblib.dump`) alongside the
    model artifact for reproducibility.
    """

    process: BinningProcess


def fit_binning(X_train: pd.DataFrame, target_col: str = 'target') -> BinningArtifacts:
    """Fit BinningProcess on ALGORITHMIC_FEATURES using the TRAINING partition only."""
    X_train = _add_derived_features(X_train)
    process = BinningProcess(
        variable_names=ALGORITHMIC_FEATURES,
        categorical_variables=ALGORITHMIC_CATEGORICAL,
        binning_fit_params=ALGORITHMIC_FIT_PARAMS,
    )
    process.fit(X_train[ALGORITHMIC_FEATURES], X_train[target_col])
    return BinningArtifacts(process=process)


def _build_categorical_label_map(binning_table_df: pd.DataFrame) -> Dict[str, str]:
    """Map each original category -> human-readable merged-bin label."""
    mapping: Dict[str, str] = {}
    body = binning_table_df.iloc[:-1]  # drop 'Totals' row
    for _, row in body.iterrows():
        bin_val = row['Bin']
        if isinstance(bin_val, str):
            continue  # 'Special' / 'Missing' rows -- no raw categories to map
        cats = sorted(str(c) for c in bin_val)
        if not cats:
            continue
        label = "_".join(cats)
        for cat in cats:
            mapping[cat] = label
    return mapping


def _format_numeric_bin(bin_str: str) -> str:
    """Reformat an optbinning interval string into a compact, safe label."""
    if bin_str in ('Special', 'Missing'):
        return bin_str.lower()
        
    nums = re.findall(r"-?\d+\.?\d*", bin_str)
    if not nums:
        return bin_str

    # Fungsi pembantu untuk memformat float menjadi string aman tanpa titik (.)
    # Contoh: 0.19 -> "0p19" atau "0_19" agar aman untuk penamaan kolom/fitur
    def fmt(val_str: float) -> str:
        val = float(val_str)
        # Jika aslinya integer (misal 1.0 atau 2), hilangkan .0 nya
        if val.is_integer():
            return str(int(val))
        # Jika desimal, ganti tanda titik dengan huruf 'p' (point) atau '_'
        return str(val).replace('.', 'p')
    
    if bin_str.strip().startswith('(-inf'):
        return f"less_{fmt(nums[-1])}"
    
    if bin_str.strip().endswith('inf)'):
        return f"greater_{fmt(nums[0])}"
    
    if len(nums) >= 2:
        return f"{fmt(nums[0])}_to_{fmt(nums[1])}"
    
    return bin_str


def transform_binning(X: pd.DataFrame, binning_artifacts: BinningArtifacts) -> pd.DataFrame:
    """
    Apply the hybrid binning (algorithmic + manual) to a single split.

    Parameters
    ----------
    X : pd.DataFrame
        A single split. `issue_d` / `earliest_cr_line` must be datetime64.
    binning_artifacts : BinningArtifacts
        Output of `fit_binning(X_train)`. Reused, unchanged, across every
        split for the 4 algorithmic features.

    Returns
    -------
    pd.DataFrame
        New DataFrame (input not mutated). Every feature -- algorithmic
        and manual alike -- ends up as a single categorical `_group` /
        `_bin` column, ready for downstream one-hot encoding, matching
        the existing scorecard convention (one dummy per bin, e.g.
        `revol_util_bin_ascending_bucket`).

    Notes
    -----
    Unseen categorical values (a `purpose` category never observed
    during `fit`) map to `'Unknown'`; true missing values map to
    `'Missing'`. A rising rate of either in production is itself a
    population-drift signal -- monitor via characteristic-level PSI
    (stability.py), not silently absorbed here.
    """
    X = _add_derived_features(X)
    X = _apply_manual_binning(X)

    # --- algorithmic (numerical): metric='bins' gives clean interval strings.
    # BinningProcess.transform requires ALL fitted variable_names to be
    # present in X, so we pass the full algorithmic feature set and then
    # select only the numerical columns from the result.
    all_bins = binning_artifacts.process.transform(X[ALGORITHMIC_FEATURES], metric='bins')
    numeric_bins = all_bins[ALGORITHMIC_NUMERICAL]
    for col in ALGORITHMIC_NUMERICAL:
        X[f"{col}_bin"] = numeric_bins[col].astype(str).apply(_format_numeric_bin).astype('category')
    
    # # --- algorithmic (numerical - WoE Transformation) ---
    # for col in ALGORITHMIC_NUMERICAL:
    #     binned_var = binning_artifacts.process.get_binned_variable(col)
        
    #     # metric="woe" otomatis mengubah angka kontinu menjadi nilai float WoE-nya
    #     X[f"{col}_woe"] = pd.Series(
    #         binned_var.transform(X[col], metric="woe"), 
    #         index=X.index
    #     ).astype(float)
    
    # --- algorithmic (categorical): explicit category-> label map (robust
    # to unseen categories / missing values, unlike raw bin-index lookup)
    for col in ALGORITHMIC_CATEGORICAL:
        binned_var = binning_artifacts.process.get_binned_variable(col)
        label_map = _build_categorical_label_map(binned_var.binning_table.build())
        mapped = X[col].map(label_map)
        mapped = np.where(X[col].isna(), 'Missing', mapped)
        mapped = pd.Series(mapped, index=X.index).fillna('Unknown')
        X[f"{col}_bin"] = mapped.astype('category')
 
    X = X.drop(columns=ALGORITHMIC_FEATURES + EXCLUDED_FEATURES, errors='ignore')
    
    X['term'] = X['term'].replace({' 36 months': '36_months', ' 60 months': '60_months'}).astype('category')
    X['verification_status'] = X['verification_status'].replace(
        {'Not Verified': 'Not_Verified', 'Source Verified': 'Verified'}
    ).astype('category')
     
    return X

# ============================================================================
# 5. Audit utilities
# ============================================================================

def binning_summary(binning_artifacts: BinningArtifacts) -> pd.DataFrame:
    """IV / bin-count / optimizer-status audit table for the 4 algorithmic features."""
    return binning_artifacts.process.summary()


def binning_table(binning_artifacts: BinningArtifacts, variable_name: str) -> pd.DataFrame:
    """Full per-bin table (Bin, Count, Event rate, WoE, IV) for one algorithmic feature."""
    return binning_artifacts.process.get_binned_variable(variable_name).binning_table.build()


def evaluate_bad_rate_monotonicity(
    X: pd.DataFrame, bin_col: str, target_col: str, bin_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute the bad rate per bin and report whether it is monotonic.

    Works identically on MANUAL bins (e.g. 'delinq_2yrs_group') and
    ALGORITHMIC bins (e.g. 'annual_inc_bin') -- the check operates purely
    on the resulting categorical column, independent of how it was
    produced.

    Parameters
    ----------
    X : pd.DataFrame
        Binned TRAINING data (never validation/test -- this is a design
        diagnostic, not an out-of-sample performance metric).
    bin_col : str
        Name of the binned/categorical column to evaluate.
    target_col : str
        Binary target column (1 = BAD, 0 = GOOD).
    bin_order : list of str, optional
        Explicit ordinal ordering of the bin categories. If omitted, bins
        are ordered by their own empirical bad rate (ascending) purely
        for display -- supply this explicitly whenever the bin has a
        real domain ordering you want to verify against (e.g.
        ['less_5', '5_15', 'greater_15']).

    Returns
    -------
    pd.DataFrame
        ['bin', 'count', 'bad_count', 'bad_rate',
         'is_monotonic_increasing', 'is_monotonic_decreasing'].
    """
    summary = (
        X.groupby(bin_col, observed=True)[target_col]
        .agg(count='size', bad_count='sum')
        .assign(bad_rate=lambda d: d['bad_count'] / d['count'])
    )

    if bin_order is not None:
        summary = summary.reindex(bin_order)
    else:
        summary = summary.sort_values('bad_rate')

    summary['is_monotonic_increasing'] = summary['bad_rate'].is_monotonic_increasing
    summary['is_monotonic_decreasing'] = summary['bad_rate'].is_monotonic_decreasing
    return summary.reset_index()