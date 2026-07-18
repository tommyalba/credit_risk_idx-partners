"""
binning_woe.py

Monotonic Binning module.

    ALGORITHMIC (optbinning.BinningProcess, monotonic_trend set EXPLICITLY):
    'annual_inc', 'credit_age_years', 'dti', 'emp_length_ordinal', 'inq_last_6mths', 
    'loan_to_income', 'open_acc_ratio', 'revol_bal_to_income', 'revol_util', 'total_acc',
    
    EXCLUDED (dropped upstream, before this module receives the data):
        tot_coll_amt, tot_cur_bal, total_rev_hi_lim
        -> 100% missing for issue_d in 2007-2011, populated from 2012+.
           A reporting-schema artifact tied to loan vintage, not a risk
           signal; confirmed via characteristic-level PSI (train vs. OOT
           test ~4.0) and direct missing-rate inspection by vintage.

Rationale for the algorithmic split
----------------------------------------------
`annual_inc` and `revol_util` are continuous with a genuine, data-driven
optimal cutpoint search worth running. `purpose` (11 categories) and
`home_ownership` (4 categories) benefit from optimal category-group
merging, which is not a decision a human should hand-pick by inspection.

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
find violations even on features whose solver status reads "OPTIMAL" on Train.
The trend also can shift on Validation & OOT (Out-Of-Time) Test set. 

To guarantee genuine monotonicity, `monotonic_trend` is set EXPLICITLY
below ("descending" for annual_inc, "ascending" for revol_util, etc.) 
rather than left as "auto".

Quick usage
-----------
The input to this module is raw LendingClub-style data before derived
features are created. `issue_d` and `earliest_cr_line` must already be
datetime64 columns.

1. Fit binning only on Train.

    from binning_woe import fit_binning

    binning_artifacts = fit_binning(X_train_raw, target_col="target")

2. Transform every split with the fitted artifacts.

    from binning_woe import transform_binning

    X_train_woe = transform_binning(X_train_raw, binning_artifacts, metric="woe")
    X_valid_woe = transform_binning(X_valid_raw, binning_artifacts, metric="woe")
    X_test_woe = transform_binning(X_test_raw, binning_artifacts, metric="woe")

3. Use alternate metrics when needed.

    X_train_bins = transform_binning(X_train_raw, binning_artifacts, metric="bins")
    X_train_idx = transform_binning(X_train_raw, binning_artifacts, metric="indices")

4. Inspect fitted bins and audits.

    from binning_woe import (
        binning_summary,
        binning_table,
        check_special_missing_consistency,
        transform_bin_labels,
    )

    summary = binning_summary(binning_artifacts)
    annual_inc_table = binning_table(binning_artifacts, "annual_inc")
    labels = transform_bin_labels(X_valid_raw, binning_artifacts)
    audit = check_special_missing_consistency(
        binning_artifacts,
        X_train_raw,
        feature="annual_inc",
    )

Production notes
----------------
- Do not refit on Validation, Test, or production data.
- Use `metric="woe"` for logistic-regression model training/inference.
- Use `transform_bin_labels()` when another module needs canonical bin
  labels. It is the shared source of truth for scorecard lookup labels.
- Missing/Special WoE transforms use empirical values in this module; keep
  that convention aligned with scorecard validation.
    
"""

USAGE = """
binning_woe.py quick usage
--------------------------
from binning_woe import (
    fit_binning,
    transform_binning,
    transform_bin_labels,
    binning_summary,
    binning_table,
)

# Fit on Train only.
binning_artifacts = fit_binning(X_train_raw, target_col="target")

# Produce model-ready WoE columns.
X_train_woe = transform_binning(X_train_raw, binning_artifacts, metric="woe")
X_valid_woe = transform_binning(X_valid_raw, binning_artifacts, metric="woe")
X_test_woe = transform_binning(X_test_raw, binning_artifacts, metric="woe")

# Produce canonical labels for scorecard lookup/reporting.
bin_labels = transform_bin_labels(X_valid_raw, binning_artifacts)

# Inspect fitted bins.
summary = binning_summary(binning_artifacts)
annual_inc_table = binning_table(binning_artifacts, "annual_inc")
"""

from collections.abc import Iterable
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
    'annual_inc', 'credit_age_years', 'dti', 'emp_length_ordinal', 'inq_last_6mths', 
    'loan_to_income', 'open_acc_ratio', 'revol_util',
]
ALGORITHMIC_CATEGORICAL: List[str] = [
    'has_derogatory', 'home_ownership', 'purpose', 'term'
]
ALGORITHMIC_FEATURES: List[str] = ALGORITHMIC_NUMERICAL + ALGORITHMIC_CATEGORICAL

# Defensive only: these should already be absent from the input DataFrame
# (dropped upstream). Referenced here solely so `transform_binning` can
# drop them with `errors="ignore"` if they slip through unexpectedly.
EXCLUDED_FEATURES: List[str] = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

# Features that need to be dropped due to wrong-sign coefficients and 
# undetected multicollinearity after training and evaluation
DROPPED_FEATURES: List[str] = ['revol_bal_to_income', 'total_acc', 'verification_status']

_NUMERIC_INTERVAL_RE = re.compile(
    r"^[\(\[]\s*-?(?:\d+\.?\d*|\.\d+|inf)\s*,\s*-?(?:\d+\.?\d*|\.\d+|inf)\s*[\)\]]$"
)

ALGORITHMIC_FIT_PARAMS: Dict[str, dict] = {
    # --- CONTINUOUS/ORDINAL NUMERIC FEATURES ---
    'annual_inc': dict(max_n_bins=10, min_bin_size=0.05, monotonic_trend='descending'),
    'credit_age_years': dict(max_n_bins=4, min_bin_size=0.1, monotonic_trend='descending'),
    'dti': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'emp_length_ordinal': dict(max_n_bins=10, min_bin_size=0.05, monotonic_trend='ascending'),
    'inq_last_6mths': dict(max_n_bins=4, min_bin_size=0.02, monotonic_trend='ascending'),
    'loan_to_income': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'revol_bal_to_income': dict(max_n_bins=8, min_bin_size=0.05, monotonic_trend='ascending'),
    'revol_util': dict(max_n_bins=11, min_bin_size=0.05, monotonic_trend='ascending'),
    'total_acc': dict(max_n_bins=6, min_bin_size=0.05, monotonic_trend='descending'),
    'open_acc_ratio': dict(max_n_bins=7, min_bin_size=0.05, monotonic_trend='ascending'),
    # --- NOMINAL CATEGORY FEATURES ---
    # Nominal categories have no inherent numeric order, so 'auto' is
    # appropriate here: categories are internally sorted by event rate
    # before optimal merging, so the resulting groups are monotonic along
    # THAT induced ordering by construction -- unlike the numerical if
    # 'peak'/'valley' ambiguity described above.
    'purpose': dict(max_n_bins=5, min_bin_size=0.02, monotonic_trend='ascending', cat_cutoff=0.02),
    'home_ownership': dict(max_n_bins=3, min_bin_size=0.03, monotonic_trend='ascending'),
    # --- DISCRETE/ORDINAL/BINARY CATEGORY (Tenor 36/60 & Has/No) ---
    # Locked to 2 bins, e.g. there are only the numbers 36 and 60 with an Ascending trend.
    'term': dict(max_n_bins=2, min_bin_size=0.05, monotonic_trend='ascending'),
    'verification_status': dict(max_n_bins=2, min_bin_size=0.05, monotonic_trend='ascending'),
    'has_derogatory': dict(max_n_bins=2, min_bin_size=0.05, monotonic_trend='ascending'),
}


# ============================================================================
# 2. Derived-feature construction
# ============================================================================
def extract_emp_length_years(text: Optional[str]) -> float:
    """Parse free-text employment length, e.g. '< 1 year', '10+ years'."""
    if pd.isna(text):
        return np.nan
    text = str(text)
    if '< 1' in text:
        return 0
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else np.nan


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Construct credit_age_years and emp_length_ordinal from raw columns."""
    out = X.copy()
    
    credit_history_age_month = (
        (out['issue_d'].dt.year - out['earliest_cr_line'].dt.year) * 12
        + (out['issue_d'].dt.month - out['earliest_cr_line'].dt.month)
    ).fillna(0).astype(int)
    out['credit_age_years'] = credit_history_age_month / 12
    out['emp_length_ordinal'] = out['emp_length'].apply(extract_emp_length_years)

    condition = (
        (out['collections_12_mths_ex_med'] > 0) | 
        (out['delinq_2yrs'] > 0) | 
        (out['pub_rec'] > 0)
    )
    
    out['has_derogatory'] = np.where(condition, 1, 0)
    out['has_derogatory'] = out['has_derogatory'].apply(
        lambda v: 'Unknown' if pd.isna(v) else ('No_Derogatory' if v == 0 else 'Has_Derogatory')
    ).astype('category')

    out['loan_to_income'] = out['loan_amnt'] / out['annual_inc']
    out['revol_bal_to_income'] = out['revol_bal'] / out['annual_inc']
    out['open_acc_ratio'] = out['open_acc'] / out['total_acc'].replace(0, np.nan)  # avoid div/0

    out['term'] = out['term'].replace(
        {' 36 months': '36_months', ' 60 months': '60_months'}
    ).astype('category')

    out['home_ownership'] = out['home_ownership'].astype('category')
    out['purpose'] = out['purpose'].astype('category')
    
    # List of dropped features
    base_cols = [
        'collections_12_mths_ex_med', 'delinq_2yrs', 'earliest_cr_line', 
        'emp_length', 'issue_d', 'loan_amnt', 'revol_bal', 'open_acc', 'pub_rec'
    ]
    out = out.drop(columns=base_cols + DROPPED_FEATURES)
    
    return out


# ============================================================================
# 3. Fit / transform contract
# ============================================================================

@dataclass
class BinningArtifacts:
    """
    Container for the fitted `BinningProcess` (algorithmic features).
    """
    process: BinningProcess


def fit_binning(X_train: pd.DataFrame, target_col: str = 'target') -> BinningArtifacts:
    """Fit BinningProcess on ALGORITHMIC_FEATURES using the TRAINING partition only."""
    X_train = add_derived_features(X_train)
    process = BinningProcess(
        variable_names=ALGORITHMIC_FEATURES,
        categorical_variables=ALGORITHMIC_CATEGORICAL,
        binning_fit_params=ALGORITHMIC_FIT_PARAMS,
    )
    process.fit(X_train[ALGORITHMIC_FEATURES], X_train[target_col])

    binning_artifacts = BinningArtifacts(process=process)

    print("\n" + "="*50)
    print("RUNNING SPECIAL/MISSING CONSISTENCY AUDIT")
    print("="*50)
    
    for feature in ALGORITHMIC_FEATURES:
        audit = check_special_missing_consistency(
            binning_artifacts=binning_artifacts,
            X=X_train,
            feature=feature
        )
        
        # Cetak log untuk visibilitas
        display(audit.set_index('Feature'))
        
        # Evaluasi asersi (jika tabel tidak kosong)
        if not audit.empty:
            assert audit["Consistent"].all(), (
                f"FATAL INCONSISTENCY DETECTED in feature '{feature}'! "
                "The WoE value in the transformation table does not match the binning_table. "
                "Check for data type or value leaks."
            )
            
    print("\nAUDIT PASSED: All Special & Missing mappings are proven to be consistent.")
    print("="*62 + "\n")
    
    return binning_artifacts


def build_categorical_label_map(binning_table_df: pd.DataFrame) -> Dict[str, str]:
    """Map each original category -> human-readable merged-bin label."""
    mapping: Dict[str, str] = {}
    body = binning_table_df.iloc[:-1]  # drop 'Totals' row
    for _, row in body.iterrows():
        bin_val = row['Bin']
        if isinstance(bin_val, str) and bin_val in ('Special', 'Missing'):
            continue
        if isinstance(bin_val, str):
            cats = [bin_val]
        else:
            cats = sorted(str(c) for c in bin_val)
        if not cats:
            continue
        label = canonical_bin_label(cats)
        for cat in cats:
            mapping[cat] = label
    return mapping


def format_numeric_bin(bin_str: str) -> str:
    """Reformat an optbinning interval string into a compact, safe label."""
    if bin_str in ('Special', 'Missing'):
        return bin_str.lower()
        
    nums = re.findall(r"-?\d+\.?\d*", bin_str)
    if not nums:
        return bin_str
    
    # Helper function to format float to safe string without dot (.)
    # Example: 0.19 -> "0p19" or "0_19" to be safe for column/feature naming
    def fmt(val_str: float) -> str:
        val = float(val_str)
        # If the original is an integer (e.g. 1.0 or 2), remove the .0
        if val.is_integer():
            return str(int(val))
        # If decimal, replace the dot with the letter 'p' (dot) or '_'
        return str(val).replace('.', '.')
    
    if bin_str.strip().startswith('(-inf'):
        return f"less_{fmt(nums[-1])}"
    
    if bin_str.strip().endswith('inf)'):
        return f"greater_{fmt(nums[0])}"
    
    if len(nums) >= 2:
        return f"{fmt(nums[0])}_to_{fmt(nums[1])}"
    
    return bin_str


def is_numeric_interval_label(value) -> bool:
    """
    Identify optbinning numeric interval labels at the binning boundary.

    This check lives here so scorecard code does not duplicate interval
    grammar or accidentally confuse categorical labels containing numbers
    with numeric bins.
    """
    if not isinstance(value, str):
        return False
    value = value.strip()
    if value in ('Special', 'Missing'):
        return False
    return bool(_NUMERIC_INTERVAL_RE.match(value))


def _canonical_category_label(value) -> str:
    """
    Normalize category groups from optbinning tables without depending on
    pandas' object reprs.

    The scorecard and any other runtime consumer need a stable join key,
    but parsing stringified extension-array internals would couple the
    project to pandas display behavior. This helper only accepts real
    Python values from the binning table or already-clean labels produced
    by this module.
    """
    if value is None:
        return 'Missing'

    if np.isscalar(value):
        try:
            if pd.isna(value):
                return 'Missing'
        except Exception:
            pass

    if isinstance(value, str):
        value = value.strip()
        if value in ('Special', 'Missing'):
            return value
        return value

    if isinstance(value, Iterable):
        cats = []
        for item in value:
            if item is None:
                continue
            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass
            cats.append(str(item))
        return "_".join(sorted(set(cats)))

    return str(value)


def canonical_bin_label(value) -> str:
    """
    Return the canonical bin label used by downstream scorecard runtime code.

    Numeric interval formatting lives here because `binning_woe.py` owns the
    optbinning contract. Scorecard code should receive a ready-to-join label
    and should not need to know whether a value came from a numeric interval,
    a categorical group, or a structural Special/Missing row.
    """
    if isinstance(value, str):
        value = value.strip()
        if value in ('Special', 'Missing'):
            return value
        if is_numeric_interval_label(value):
            return format_numeric_bin(value)
        return value

    return _canonical_category_label(value)


def transform_bin_labels(
    X: pd.DataFrame,
    binning_artifacts: BinningArtifacts,
) -> pd.DataFrame:
    """
    Transform raw rows into canonical bin labels for every algorithmic feature.

    This is the shared label source for reporting and scorecard scoring. It
    deliberately avoids categorical `metric='bins'` output because some
    optbinning/pandas combinations expose extension-array reprs there; mapping
    raw categories through the fitted binning table is both clearer and more
    stable.
    """
    if set(ALGORITHMIC_FEATURES).issubset(X.columns):
        X_prepared = X.copy()
    else:
        X_prepared = add_derived_features(X)
    process = binning_artifacts.process
    labels = pd.DataFrame(index=X.index)

    all_bins = process.transform(X_prepared[ALGORITHMIC_FEATURES], metric='bins')
    for col in ALGORITHMIC_NUMERICAL:
        labels[col] = all_bins[col].map(canonical_bin_label)

    for col in ALGORITHMIC_CATEGORICAL:
        binned_var = process.get_binned_variable(col)
        label_map = build_categorical_label_map(binned_var.binning_table.build())
        mapped = X_prepared[col].map(lambda value: label_map.get(str(value), np.nan))
        mapped = mapped.mask(X_prepared[col].isna(), 'Missing')
        labels[col] = mapped.fillna('Unknown')

    return labels


def transform_binning(
    X: pd.DataFrame, 
    binning_artifacts: BinningArtifacts,
    metric: str = 'woe'
    ) -> pd.DataFrame:
    """
    Apply the binning (algorithmic features) to a single split.
    
    Parameters
    ----------
    X : pd.DataFrame
        A single split. `issue_d` / `earliest_cr_line` must be datetime64.
    binning_artifacts : BinningArtifacts
        Output of `fit_binning(X_train)`. Reused, unchanged, across every
        split for all algorithmic features.

    Returns
    -------
    pd.DataFrame
        New DataFrame (input not mutated).

    Notes
    -----
    Unseen categorical values (a `purpose` category never observed
    during `fit`) map to `'Unknown'`; true missing values map to
    `'Missing'`. A rising rate of either in production is itself a
    population-drift signal -- monitor via characteristic-level PSI
    (population_stability_index.py), not silently absorbed here.
    """
    X = add_derived_features(X)

    if metric == 'bins':
        # =========================================================
        # 1. 'BINS' STRATEGY: Outputs interval category strings
        # =========================================================
        canonical_labels = transform_bin_labels(X, binning_artifacts)
        for col in ALGORITHMIC_FEATURES:
            X[f"{col}_bin"] = canonical_labels[col].astype('category')

    elif metric in ['woe', 'indices']:
        # =========================================================
        # 2. 'WOE' / 'INDICES' STRATEGY: numeric representation
        # =========================================================
        suffix = "woe" if metric == "woe" else "idx"
        target_dtype = float if metric == "woe" else int

        # metric_missing='empirical' / metric_special='empirical' -- see
        # the FIX explained in the docstring above. Without these,
        # missing/special rows silently get 0.0 / index 0 instead of
        # their actual fitted bin's metric.
        transformed = binning_artifacts.process.transform(
            X[ALGORITHMIC_FEATURES],
            metric=metric,
            metric_missing='empirical',
            metric_special='empirical',
        )
    
        for col in ALGORITHMIC_FEATURES:
            X[f"{col}_{suffix}"] = transformed[col].astype(target_dtype)
    
    else:
        raise ValueError(f"Metric '{metric}' is not supported. Please choose 'bins', 'woe', or 'indices'.")
        
    X = X.drop(columns=ALGORITHMIC_FEATURES + EXCLUDED_FEATURES, errors='ignore')
     
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
    X: pd.DataFrame, bin_col: str, target_col: str
) -> pd.DataFrame:
    """
    Compute the bad rate per bin and report whether it is monotonic.

    Works for all metric="woe" as well as "bins" -- the check applies 
    to both numeric and categorical WoE columns generated.

    Parameters
    ----------
    X : pd.DataFrame
        Binned TRAINING data (never validation/test -- this is a design
        diagnostic, not an out-of-sample performance metric).
    bin_col : str
        Name of the binned/categorical column to evaluate.
    target_col : str
        Binary target column (0 = GOOD, 1 = BAD).

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
    
    summary['is_monotonic_increasing'] = summary['bad_rate'].is_monotonic_increasing
    summary['is_monotonic_decreasing'] = summary['bad_rate'].is_monotonic_decreasing
    return summary.reset_index()


def check_special_missing_consistency(
    binning_artifacts: BinningArtifacts,
    X: pd.DataFrame,
    feature: str,
    atol: float = 1e-10,
) -> pd.DataFrame:
    """
    Audit consistency between Special/Missing WoE shown in the binning table
    and the WoE actually produced by transform(metric='woe').

    Parameters
    ----------
    binning_artifacts : BinningArtifacts
        Output of fit_binning().
    X : pd.DataFrame
        Dataset containing the feature to audit.
    feature : str
        Feature name.
    atol : float
        Numerical tolerance.

    Returns
    -------
    pd.DataFrame
        One row for each of {"Special", "Missing"}.

        Columns
        -------
        Bin
            "Special" or "Missing"
        Table WoE
            WoE reported by binning_table.build()
        Transform WoE
            Actual WoE produced by transform().
        Count
            Number of observations falling into that bin.
        Consistent
            True if both values match within tolerance.
    """
    # 1. Pipeline prep
    X_eval = X.copy()
    
    # [LOCKED]: Force numeric to float so that C-engine optbinning doesn't mistake pure NaNs for "Special" (WoE 0.0)
    for col in ALGORITHMIC_NUMERICAL:
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype(float)
            
    process = binning_artifacts.process

    # 2. Transform simultaneously
    transformed = process.transform(
        X_eval[ALGORITHMIC_FEATURES],
        metric="woe",
        metric_missing="empirical",
        metric_special="empirical",
    )

    # obtain bin labels
    bins = process.transform(
        X_eval[ALGORITHMIC_FEATURES],
        metric="bins",
    )

    table = binning_table(binning_artifacts, feature)
    rows = []

    for label in ("Special", "Missing"):
        table_row = table.loc[table["Bin"].astype(str) == label]
        # Safe from IndexError if bin does not exist during fitting
        if table_row.empty:
            continue

        table_woe = float(table_row["WoE"].iloc[0])
        mask = bins[feature].astype(str) == label
        count = int(mask.sum())

        if count == 0:
            transform_woe = np.nan
            consistent = True
        else:
            observed = transformed.loc[mask, feature].unique()
    
            if len(observed) != 1:
                consistent = False
                # Convert numpy array to string to keep it informative in DataFrame 
                # and not hide traces of value leaks during audit 
                transform_woe = f"MULTIPLE_VALUES: {observed.tolist()}"
            else:
                transform_woe = float(observed[0])
                consistent = np.isclose(
                    table_woe,
                    transform_woe,
                    atol=atol,
                    equal_nan=True,
                )

        rows.append(
            {
                "Feature": feature,
                "Bin": label,
                "Table WoE": table_woe,
                "Transform WoE": transform_woe,
                "Count": count,
                "Consistent": consistent,
            }
        )
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(USAGE)
