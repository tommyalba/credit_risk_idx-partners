"""
performance_window.py

Fixed performance-window population eligibility filter for the
Application-Time credit risk model. Replaces the earlier "wait for final
resolution" target construction, which was shown to introduce vintage-
dependent selection bias: dropping 'Current'/'In Grace Period'
unconditionally removes most surviving-good loans from young vintages
while retaining nearly all already-surfaced bad loans from those same
vintages (because 'Charged Off' can occur within months, while 'Fully
Paid' requires near-full term completion) -- producing an artificially
rising bad rate purely as a function of vintage recency (13.95% at the
2010Q4 vintage vs. 28.68% at 2014Q4 in the shared analysis), not a
genuine economic/behavioral shift.

Methodology
-----------
1. `compute_time_to_bad_distribution` -- empirically estimate how many
   months after issuance a loan surfaces as BAD, using
   (last_pymnt_d - issue_d) for loans already in a bad status. This is
   the DATA-DRIVEN basis for choosing M; it is not a fixed textbook
   number.
2. `recommend_performance_window` -- pick M as the `coverage`-th
   percentile of that distribution (default 90th): M months is enough
   time for at least `coverage` proportion of eventual bad loans to have
   already surfaced.
3. `apply_performance_window_filter` -- apply M as an ELIGIBILITY rule
   (loan_age_months >= M), not a relabeling rule. Within the eligible
   population, the loan's status AT SNAPSHOT is used directly --
   'Current' loans that have survived >= M months are legitimately
   labeled GOOD (they were genuinely at risk for long enough and did not
   go bad). This is the key correction relative to the prior approach of
   unconditionally dropping every 'Current' loan regardless of age.

CRITICAL: this filter must be applied with the SAME M and the SAME
`snapshot_date` to train, valid, AND OOT test. For OOT test in
particular, `snapshot_date` must be the actual dataset extraction date
-- never `issue_d.max()` computed within that split alone -- otherwise
loan age is computed inconsistently across splits and the whole point of
this filter is defeated.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


DEFAULT_BAD_STATUSES: List[str] = [
    'Charged Off', 'Default', 'Late (31-120 days)', 
    'Does not meet the credit policy. Status:Charged Off',
]

# RESOLVED good: the loan reached a terminal, unambiguous good outcome
# (paid in full, whether on schedule or early). Once resolved this way,
# it can never become Bad later -- no waiting period needed, regardless
# of how young the loan is.
DEFAULT_RESOLVED_GOOD_STATUSES: List[str] = [
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid',
]

# OPEN good: the loan is still active and has NOT (yet) gone bad. This is
# the ONLY status group for which age matters -- an open loan can still
# go bad later, so it is only confidently labeled Good once it has
# survived the full M-month risk window.
DEFAULT_OPEN_GOOD_STATUSES: List[str] = ['Current']

# Excluded on a separate rationale from the maturity issue: genuinely
# ambiguous / in-progress outcome, not yet resolved either way.
DEFAULT_INDETERMINATE_STATUSES: List[str] = [
    'In Grace Period',
    'Late (16-30 days)',
]


# ============================================================================
# 1. Data-driven M selection
# ============================================================================

def compute_time_to_bad_distribution(
    df: pd.DataFrame,
    bad_statuses: List[str] = DEFAULT_BAD_STATUSES,
    issue_col: str = "issue_d",
    last_pymnt_col: str = "last_pymnt_d",
    status_col: str = "loan_status",
) -> pd.Series:
    """
    Empirical months-to-bad distribution: for every loan currently in a
    BAD status, the elapsed time between `issue_d` and `last_pymnt_d`
    (proxy for when the loan stopped performing).

    Returns
    -------
    pd.Series
        One value (months, float) per bad loan. Inspect via
        `.describe()` / `.quantile([0.5, 0.75, 0.9, 0.95])` before
        choosing M.
    """
    bad = df.loc[df[status_col].isin(bad_statuses), [issue_col, last_pymnt_col]].dropna()
    months = (
        (bad[last_pymnt_col].dt.year - bad[issue_col].dt.year) * 12
        + (bad[last_pymnt_col].dt.month - bad[issue_col].dt.month)
    ).clip(lower=0)
    return months.rename("months_to_bad")


def recommend_performance_window(months_to_bad: pd.Series, coverage: float = 0.90) -> int:
    """
    Recommend M as the `coverage`-th percentile of the empirical
    months-to-bad distribution.

    Parameters
    ----------
    months_to_bad : pd.Series
        Output of `compute_time_to_bad_distribution`.
    coverage : float
        e.g. 0.90 = "90% of eventual bad loans have surfaced by month M".
        Higher coverage -> larger M -> fewer immature exclusions, at the
        cost of pushing the usable OOT window further back in time.

    Returns
    -------
    int
        Recommended M, rounded up to the nearest whole month.
    """
    return int(np.ceil(months_to_bad.quantile(coverage)))


# ============================================================================
# 2. Eligibility filter
# ============================================================================

@dataclass
class PerformanceWindowResult:
    """Eligible population with target assigned, plus an audit funnel."""

    df: pd.DataFrame
    n_input: int
    n_eligible: int
    n_excluded_immature: int
    n_excluded_indeterminate: int


def apply_performance_window_filter(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    M: int,
    issue_col: str = "issue_d",
    status_col: str = "loan_status",
    bad_statuses: List[str] = DEFAULT_BAD_STATUSES,
    resolved_good_statuses: List[str] = DEFAULT_RESOLVED_GOOD_STATUSES,
    open_good_statuses: List[str] = DEFAULT_OPEN_GOOD_STATUSES,
    indeterminate_statuses: List[str] = DEFAULT_INDETERMINATE_STATUSES,
) -> PerformanceWindowResult:
    """
    Apply the fixed performance-window eligibility filter and assign
    the binary target, using ASYMMETRIC eligibility by status:

      - `bad_statuses`           -> ALWAYS eligible, any age. The outcome
        is already resolved and final; waiting for M months would only
        discard already-known information, not improve it.
      - `resolved_good_statuses` -> ALWAYS eligible, any age. Fully Paid
        (on-schedule or early) is a terminal outcome -- the loan can
        never become Bad afterward.
      - `open_good_statuses` (i.e. 'Current') -> eligible ONLY if
        loan_age_months >= M. This is the ONLY status group where the
        outcome is genuinely still unresolved, so it is the only one
        that needs the maturity guarantee before being labeled Good.
      - `indeterminate_statuses` -> always excluded (ambiguous outcome,
        unrelated rationale to the maturity issue).

    This asymmetry is what actually fixes the vintage-recency bias
    (which came specifically from mishandling 'Current' loans), while
    avoiding the unnecessary loss of already-resolved Bad/Fully-Paid
    loans from young vintages that a blanket age cutoff discards for no
    methodological reason.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loan-level data containing `issue_col` and `status_col`.
    snapshot_date : pd.Timestamp
        Dataset extraction date. loan_age_months = snapshot_date - issue_d.
    M : int
        Minimum required loan age (months), applied ONLY to
        `open_good_statuses`.

    Returns
    -------
    PerformanceWindowResult
    """
    out = df.copy()
    n_input = len(out)

    loan_age_months = (
        (snapshot_date.year - out[issue_col].dt.year) * 12
        + (snapshot_date.month - out[issue_col].dt.month)
    )

    is_bad = out[status_col].isin(bad_statuses)
    is_resolved_good = out[status_col].isin(resolved_good_statuses)
    is_open_good = out[status_col].isin(open_good_statuses)
    is_indeterminate = out[status_col].isin(indeterminate_statuses)

    unclassified = ~(is_bad | is_resolved_good | is_open_good | is_indeterminate)
    if unclassified.any():
        raise ValueError(
            f"{int(unclassified.sum())} rows have a loan_status not covered "
            f"by any status list: "
            f"{sorted(out.loc[unclassified, status_col].unique().tolist())}"
        )

    is_mature_open_good = is_open_good & (loan_age_months >= M)
    is_immature_open_good = is_open_good & (loan_age_months < M)

    is_eligible = is_bad | is_resolved_good | is_mature_open_good
    n_excluded_immature = int(is_immature_open_good.sum())
    n_excluded_indeterminate = int(is_indeterminate.sum())

    out = out.loc[is_eligible].copy()
    out["target"] = out[status_col].isin(bad_statuses).astype(int)

    return PerformanceWindowResult(
        df=out,
        n_input=n_input,
        n_eligible=len(out),
        n_excluded_immature=n_excluded_immature,
        n_excluded_indeterminate=n_excluded_indeterminate,
    )


def eligibility_funnel_report(result: PerformanceWindowResult) -> pd.DataFrame:
    """Human-readable audit table of the eligibility filter's effect."""
    return pd.DataFrame([
        {"stage": "Input population", "count": result.n_input, "pct_of_input": 100.0},
        {"stage": "Excluded: immature (age < M)", "count": result.n_excluded_immature,
         "pct_of_input": round(100 * result.n_excluded_immature / result.n_input, 2)},
        {"stage": "Excluded: indeterminate status", "count": result.n_excluded_indeterminate,
         "pct_of_input": round(100 * result.n_excluded_indeterminate / result.n_input, 2)},
        {"stage": "Eligible (modeling population)", "count": result.n_eligible,
         "pct_of_input": round(100 * result.n_eligible / result.n_input, 2)},
    ])


# ============================================================================
# 3. Diagnostic: bad rate by vintage, before/after the filter
# ============================================================================

def bad_rate_by_vintage(
    df_with_target: pd.DataFrame, issue_col: str = "issue_d", target_col: str = "target"
) -> pd.DataFrame:
    """
    Quarterly bad rate on an already-labeled/eligible population -- use
    this to verify the filter actually removed the vintage-recency trend
    (i.e. bad rate should no longer rise monotonically toward the most
    recent quarters once M is applied correctly).
    """
    vintage = df_with_target[issue_col].dt.to_period("Q").astype(str)
    return (
        df_with_target.groupby(vintage)[target_col]
        .agg(count="size", bad_rate="mean")
        .reset_index()
        .rename(columns={issue_col: "vintage"})
    )


# ============================================================================
# 4. Safe modeling horizon (residual near-boundary resolution skew)
# ============================================================================

def safe_horizon_cutoff(snapshot_date: pd.Timestamp, M: int) -> pd.Timestamp:
    """
    Compute the latest `issue_d` that should be used ANYWHERE in the
    modeling population (train, valid, AND OOT test) -- not just the
    per-loan eligibility cutoff.

    Rationale: `apply_performance_window_filter` correctly labels
    individual loans, but for vintages issued within M months of
    `snapshot_date`, almost no 'Current' loan has yet crossed the
    M-month maturity bar. The tiny sliver of ALREADY-RESOLVED loans left
    eligible in those vintages is structurally skewed toward Bad --
    'Charged Off' can surface within months, while a genuine full-term
    'Fully Paid' cannot occur before the loan's term (36/60 months)
    elapses, so only rare early payoffs count as resolved-Good there.
    This is an observability problem (too few Good outcomes have had
    time to mature), not a labeling error, and no per-loan rule can fix
    it -- the vintage itself must be excluded.

    Empirically, this cutoff should coincide with the point where
    `bad_rate_by_vintage` stabilizes; if it does not, `M` may need
    re-examination.

    Returns
    -------
    pd.Timestamp
        The latest safe `issue_d` (exclusive upper bound):
        `snapshot_date` minus `M` months.
    """
    month = snapshot_date.month - M
    year = snapshot_date.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def truncate_to_safe_horizon(
    df: pd.DataFrame, snapshot_date: pd.Timestamp, M: int, issue_col: str = "issue_d"
) -> pd.DataFrame:
    """
    Drop every row with `issue_d >= safe_horizon_cutoff(snapshot_date, M)`
    from the ELIGIBLE (already `apply_performance_window_filter`-ed)
    population. Apply this ONCE, before any train/valid/OOT split -- the
    resulting DataFrame is what `bad_rate_by_vintage` should be flat on,
    all the way to its final (most recent) included quarter.
    """
    cutoff = safe_horizon_cutoff(snapshot_date, M)
    return df.loc[df[issue_col] < cutoff].copy()