"""
scorecard.py

Production-grade Credit Scorecard construction, scoring, and audit module.

This module consumes the artifacts of `binning_woe.py` (a fitted
`optbinning.BinningProcess`) together with a fitted `sklearn.linear_model.
LogisticRegression` trained on WoE-encoded features, and produces a
functional scorecard: a static per-(characteristic, bin) point lookup table
that can be summed to a final score without runtime multiplication.

Design decisions
----------------
1. Base Points are reported as a separate row.
   `offset - intercept * factor` is kept as one "Base Score" contribution
   instead of being distributed across characteristics.

2. Binning labels come from `binning_woe.py`.
   Binning, label normalization, categorical grouping, and numeric interval
   formatting are owned by `binning_woe.py`. This module only consumes
   canonical labels through `canonical_bin_label` and `transform_bin_labels`.

3. Runtime scoring is dictionary-based.
   `lookup_table` remains the reporting artifact, while scoring and
   explanation use cached dictionaries keyed by characteristic and bin label.

4. Consistency validation preserves the scorecard math.
   `validate_scorecard_consistency` compares rounded lookup-table scoring
   against the continuous closed-form score. Some drift is expected because
   points are rounded per bin.

Quick usage
-----------
This module expects:
- `binning_artifacts` from `binning_woe.fit_binning()`.
- A fitted binary `sklearn.linear_model.LogisticRegression`.
- `feature_names` matching the model coefficient order, usually the WoE
  columns used to fit the model, such as `["annual_inc_woe", ...]`.

1. Train a model on WoE-transformed data.

    from src.binning_woe import fit_binning, transform_binning
    from sklearn.linear_model import LogisticRegression

    binning_artifacts = fit_binning(X_train_raw, target_col="target")
    X_train_woe = transform_binning(X_train_raw, binning_artifacts, metric="woe")

    feature_names = [c for c in X_train_woe.columns if c.endswith("_woe")]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_woe[feature_names], y_train)

2. Build the scorecard.

    from src.scorecard import ScorecardConfig, build_scorecard

    config = ScorecardConfig(target_score=600, target_odds=50, pdo=20)
    scorecard = build_scorecard(
        model=model,
        binning_artifacts=binning_artifacts,
        feature_names=feature_names,
        config=config,
    )

3. Score, explain, validate, audit, and export.

    from src.scorecard import (
        audit_scorecard,
        explain_score,
        export_scorecard_report,
        score_dataframe,
        validate_scorecard_consistency,
    )
    
    scores = score_dataframe(scorecard, X_valid_raw)
    explanation = explain_score(scorecard, X_valid_raw.iloc[[0]])
    consistency = validate_scorecard_consistency(scorecard, X_valid_raw)
    audit = audit_scorecard(scorecard)
    markdown = export_scorecard_report(scorecard)

Production notes
----------------
- Keep `lookup_table` for reporting and audit.
- Runtime scoring uses cached dictionaries on `ScorecardArtifacts`.
- Rebuild the scorecard whenever the binning process or fitted model changes.
- Do not normalize bin labels in this module; use helpers from
  `binning_woe.py`.
"""

USAGE = """
scorecard.py quick usage
---------------------------
from binning_woe import fit_binning, transform_binning
from scorecard_v2 import (
    ScorecardConfig,
    audit_scorecard,
    build_scorecard,
    explain_score,
    export_scorecard_report,
    score_dataframe,
    validate_scorecard_consistency,
)

# 1) Fit binning and train LogisticRegression on WoE columns.
binning_artifacts = fit_binning(X_train_raw, target_col="target")
X_train_woe = transform_binning(X_train_raw, binning_artifacts, metric="woe")
feature_names = [c for c in X_train_woe.columns if c.endswith("_woe")]
model.fit(X_train_woe[feature_names], y_train)

# 2) Build scorecard.
scorecard = build_scorecard(
    model=model,
    binning_artifacts=binning_artifacts,
    feature_names=feature_names,
    config=ScorecardConfig(target_score=600, target_odds=50, pdo=20),
)

# 3) Use and audit scorecard.
scores = score_dataframe(scorecard, X_valid_raw)
explanation = explain_score(scorecard, X_valid_raw.iloc[[0]])
consistency = validate_scorecard_consistency(scorecard, X_valid_raw)
audit = audit_scorecard(scorecard)
markdown = export_scorecard_report(scorecard)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.binning_woe import (
    ALGORITHMIC_FEATURES,
    BinningArtifacts,
    add_derived_features,
    binning_table,
    canonical_bin_label,
    transform_bin_labels,
)


LOOKUP_COLUMNS = [
    "characteristic",
    "bin_label",
    "count",
    "woe",
    "coefficient",
    "point",
]


# ============================================================================
# 1. Configuration and artifact containers
# ============================================================================

@dataclass(frozen=True)
class ScorecardConfig:
    """
    Scorecard scaling parameters, following the standard industry calibration:

        Score = Offset + Factor * ln(odds_GOOD)
        Factor = PDO / ln(2)
        Offset = target_score - Factor * ln(target_odds)
    """

    target_score: int = 600
    target_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.target_score - self.factor * np.log(self.target_odds)


@dataclass
class ScorecardArtifacts:
    """
    Output of `build_scorecard`.

    `lookup_table` is intentionally kept as a plain reporting DataFrame.
    Runtime scoring uses `point_lookup` and `row_lookup` so production paths
    do not repeatedly filter DataFrames for every applicant and feature.
    """

    config: ScorecardConfig
    intercept: float
    base_points: int
    lookup_table: pd.DataFrame
    coefficients: Dict[str, float]
    feature_names: List[str]
    binning_artifacts: BinningArtifacts
    point_lookup: Dict[str, Dict[str, int]] = field(default_factory=dict, repr=False)
    row_lookup: Dict[str, Dict[str, dict]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.point_lookup or not self.row_lookup:
            self.refresh_runtime_lookup()

    def refresh_runtime_lookup(self) -> None:
        """
        Rebuild dictionary caches from `lookup_table`.

        The DataFrame is useful for reporting and audit, but dictionary lookup
        is the better runtime contract: it makes scoring O(1) per
        characteristic and keeps the scoring path independent from pandas row
        filtering behavior.
        """
        point_lookup: Dict[str, Dict[str, int]] = {}
        row_lookup: Dict[str, Dict[str, dict]] = {}

        for row in self.lookup_table.to_dict(orient="records"):
            characteristic = row["characteristic"]
            bin_label = row["bin_label"]
            point_lookup.setdefault(characteristic, {})[bin_label] = int(row["point"])
            row_lookup.setdefault(characteristic, {})[bin_label] = row

        self.point_lookup = point_lookup
        self.row_lookup = row_lookup

    def label_point_map(self, base_name: str) -> Dict[str, int]:
        """
        Return cached point mapping for one characteristic.

        This method preserves the previous public convenience API while
        avoiding the old repeated `lookup_table.loc[...]` filtering.
        """
        if not self.point_lookup:
            self.refresh_runtime_lookup()
        return self.point_lookup.get(base_name, {})

    def lookup_entry(self, base_name: str, bin_label: str) -> Optional[dict]:
        """
        Return cached reporting metadata for one score contribution.

        `explain_score` needs the same row-level facts as `lookup_table`, but
        not enough to justify a DataFrame filter inside every feature loop.
        """
        if not self.row_lookup:
            self.refresh_runtime_lookup()
        return self.row_lookup.get(base_name, {}).get(bin_label)


# ============================================================================
# 2. Internal helpers
# ============================================================================

def _strip_woe_suffix(feature_name: str) -> str:
    """
    Enforce the model/transform naming contract at the scorecard boundary.

    Failing early here is preferable to building a scorecard whose feature
    names cannot be traced back to fitted binning variables.
    """
    if not feature_name.endswith("_woe"):
        raise ValueError(
            f"Feature '{feature_name}' does not end with '_woe'. "
            "build_scorecard expects WoE-encoded model feature names "
            "matching transform_binning(metric='woe')."
        )
    return feature_name[: -len("_woe")]


def _base_feature_names(feature_names: List[str]) -> List[str]:
    """
    Convert model feature names to binning feature names once per workflow.

    Keeping this conversion centralized makes the feature contract explicit
    and prevents score, explain, and validation code from drifting apart.
    """
    return [_strip_woe_suffix(feature) for feature in feature_names]


def _validate_model_feature_contract(model, feature_names: List[str]) -> None:
    """
    Validate shape and feature ownership before any scorecard rows are built.

    A coefficient/order mismatch is silent and damaging in scorecards, so it
    deserves a hard failure before points are calculated.
    """
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("model must expose fitted coef_ and intercept_ attributes.")

    coef = np.asarray(model.coef_)
    if coef.ndim != 2 or coef.shape[0] != 1:
        raise ValueError(f"Expected binary LogisticRegression coef_ shape (1, n), got {coef.shape}.")

    if coef.shape[1] != len(feature_names):
        raise ValueError(
            f"model.coef_ has {coef.shape[1]} coefficient(s), but "
            f"{len(feature_names)} feature name(s) were provided."
        )

    unknown = sorted(set(_base_feature_names(feature_names)) - set(ALGORITHMIC_FEATURES))
    if unknown:
        raise ValueError(
            "feature_names contains variable(s) that are not managed by "
            f"binning_woe.py: {unknown}."
        )


def _assert_no_duplicate_lookup_keys(lookup_table: pd.DataFrame) -> None:
    """
    Guard the runtime dictionary from lossy key collisions.

    If two rows share the same characteristic/bin label, one point would
    overwrite the other in the runtime cache. Treating that as fatal keeps
    reporting and scoring artifacts consistent.
    """
    duplicated = lookup_table.duplicated(subset=["characteristic", "bin_label"], keep=False)
    if duplicated.any():
        offending = lookup_table.loc[duplicated, ["characteristic", "bin_label"]]
        raise ValueError(
            "Duplicate characteristic + bin_label entries would corrupt "
            f"runtime lookup: {offending.to_dict(orient='records')}."
        )


def _ordered_lookup_table(scorecard: ScorecardArtifacts) -> pd.DataFrame:
    """
    Return lookup rows in model feature order for stable report rendering.

    The report should read like the fitted model, while preserving bin order
    inside each characteristic as produced by the binning table.
    """
    lookup = scorecard.lookup_table.copy()
    order = {base: idx for idx, base in enumerate(_base_feature_names(scorecard.feature_names))}
    lookup["_feature_order"] = lookup["characteristic"].map(order)
    return (
        lookup.sort_values(["_feature_order"], kind="stable")
        .drop(columns="_feature_order")
        .reset_index(drop=True)
    )


# ============================================================================
# 3. Scorecard construction
# ============================================================================

def build_scorecard(
    model,
    binning_artifacts: BinningArtifacts,
    feature_names: List[str],
    config: Optional[ScorecardConfig] = None,
) -> ScorecardArtifacts:
    """
    Build a functional, per-bin credit scorecard.

    The lookup table is derived directly from fitted binning tables. No
    reference dataset is required, and no scorecard-side bin parser is used.
    """
    if config is None:
        config = ScorecardConfig()

    _validate_model_feature_contract(model, feature_names)

    coefs = {feature: float(beta) for feature, beta in zip(feature_names, model.coef_[0])}
    intercept = float(model.intercept_[0])
    factor = config.factor
    offset = config.offset
    base_points = int(round(offset - intercept * factor))

    rows = []
    for feature in feature_names:
        base_name = _strip_woe_suffix(feature)
        beta = coefs[feature]
        table = binning_table(binning_artifacts, base_name)
        body = table.iloc[:-1]  # drop Totals

        for _, row in body.iterrows():
            woe = float(row["WoE"])
            point = int(round(-(beta * woe) * factor))
            rows.append(
                {
                    "characteristic": base_name,
                    "bin_label": canonical_bin_label(row["Bin"]),
                    "count": int(row["Count"]),
                    "woe": woe,
                    "coefficient": beta,
                    "point": point,
                }
            )

    lookup_table = pd.DataFrame(rows, columns=LOOKUP_COLUMNS)
    _assert_no_duplicate_lookup_keys(lookup_table)

    return ScorecardArtifacts(
        config=config,
        intercept=intercept,
        base_points=base_points,
        lookup_table=lookup_table,
        coefficients=coefs,
        feature_names=list(feature_names),
        binning_artifacts=binning_artifacts,
    )


# ============================================================================
# 4. Scoring and validation
# ============================================================================

def score_dataframe(scorecard: ScorecardArtifacts, X: pd.DataFrame) -> pd.Series:
    """
    Score a raw dataset using cached per-bin integer points.

    Raises when a canonical bin label is absent from the scorecard. That is
    usually a sign that the scorecard and binning artifacts do not belong to
    the same fitted process, or that production data contains an unseen
    categorical value that should be handled deliberately upstream.
    """
    bin_labels = transform_bin_labels(X, scorecard.binning_artifacts)
    total = pd.Series(scorecard.base_points, index=X.index, dtype=float)

    for feature in scorecard.feature_names:
        base_name = _strip_woe_suffix(feature)
        point_map = scorecard.label_point_map(base_name)
        labels = bin_labels[base_name]
        unmapped = ~labels.isin(set(point_map))

        if unmapped.any():
            examples = labels.loc[unmapped].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"{int(unmapped.sum())} observation(s) resolve to a bin_label "
                f"for '{base_name}' that is absent from the scorecard lookup "
                f"table. Example label(s): {examples}."
            )

        total = total + labels.map(point_map).astype(float)

    return total.round().astype(int)


def _validate_scorecard_runtime_contract(scorecard: ScorecardArtifacts) -> None:
    """
    Validate the minimum structure required by score/validate routines.

    This is intentionally narrower than `audit_scorecard`: validation should
    fail only on conditions that would make the numerical comparison unsafe.
    """
    _assert_no_duplicate_lookup_keys(scorecard.lookup_table)

    expected = set(_base_feature_names(scorecard.feature_names))
    actual = set(scorecard.lookup_table["characteristic"].dropna())
    missing_lookup = sorted(expected - actual)
    if missing_lookup:
        raise ValueError(f"Missing lookup rows for feature(s): {missing_lookup}.")

    missing_coefs = sorted(set(scorecard.feature_names) - set(scorecard.coefficients))
    if missing_coefs:
        raise ValueError(f"Missing coefficient(s) for feature(s): {missing_coefs}.")


def validate_scorecard_consistency(
    scorecard: ScorecardArtifacts,
    X: pd.DataFrame,
    max_expected_drift: Optional[float] = None,
) -> Dict[str, float]:
    """
    Cross-check rounded lookup scoring against the continuous score formula.

    Missing and Special WoE values are transformed with empirical metrics, the
    same source-of-truth convention used in `binning_woe.py`.
    """
    _validate_scorecard_runtime_contract(scorecard)

    X_prepared = add_derived_features(X.copy())
    process = scorecard.binning_artifacts.process
    woe_transformed = process.transform(
        X_prepared[ALGORITHMIC_FEATURES],
        metric="woe",
        metric_missing="empirical",
        metric_special="empirical",
    )

    used_features = _base_feature_names(scorecard.feature_names)
    if not np.isfinite(woe_transformed[used_features].to_numpy(dtype=float)).all():
        raise ValueError("WoE transform produced NaN or infinite values for scorecard features.")

    continuous_score = pd.Series(
        scorecard.config.offset - scorecard.intercept * scorecard.config.factor,
        index=X.index,
    )
    for feature in scorecard.feature_names:
        base_name = _strip_woe_suffix(feature)
        beta = scorecard.coefficients[feature]
        continuous_score = continuous_score + (
            -(beta * woe_transformed[base_name]) * scorecard.config.factor
        )

    lookup_score = score_dataframe(scorecard, X)
    diff = (lookup_score - continuous_score).abs()

    if max_expected_drift is None:
        max_expected_drift = 0.5 * (len(scorecard.feature_names) + 1)

    flagged = diff > max_expected_drift

    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "n_flagged": int(flagged.sum()),
        "n_total": int(len(diff)),
        "tolerance": float(max_expected_drift),
    }


def explain_score(scorecard: ScorecardArtifacts, x_row: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-characteristic point breakdown for a single applicant.
    """
    if len(x_row) != 1:
        raise ValueError(f"explain_score expects exactly 1 row, got {len(x_row)}.")

    bin_labels = transform_bin_labels(x_row, scorecard.binning_artifacts)
    parts = []

    for feature in scorecard.feature_names:
        base_name = _strip_woe_suffix(feature)
        bin_label = bin_labels[base_name].iloc[0]
        entry = scorecard.lookup_entry(base_name, bin_label)

        if entry is None:
            raise ValueError(
                f"No lookup entry for '{base_name}' at bin_label={bin_label!r}. "
                "This scorecard was likely built from a different "
                "BinningProcess than the one used to transform this data."
            )

        parts.append(
            {
                "characteristic": base_name,
                "bin_label": entry["bin_label"],
                "point": int(entry["point"]),
            }
        )

    breakdown = pd.DataFrame(parts).sort_values(
        by="point", key=lambda s: s.abs(), ascending=False
    )
    base_row = pd.DataFrame(
        [{"characteristic": "Base Score", "bin_label": "-", "point": scorecard.base_points}]
    )
    total_row = pd.DataFrame(
        [
            {
                "characteristic": "Total",
                "bin_label": "-",
                "point": int(breakdown["point"].sum() + scorecard.base_points),
            }
        ]
    )
    return pd.concat([base_row, breakdown, total_row], ignore_index=True)


# ============================================================================
# 5. Reporting
# ============================================================================

def _markdown_table(rows: List[dict], columns: List[tuple]) -> List[str]:
    """
    Render a compact markdown table from explicit column definitions.

    Keeping table formatting isolated prevents report layout changes from
    bleeding into scorecard construction or scoring logic.
    """
    headers = [header for header, _, _ in columns]
    alignments = [alignment for _, _, alignment in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]

    for row in rows:
        cells = []
        for _, key, _ in columns:
            value = row[key]
            if key in {"woe", "coefficient"}:
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def export_scorecard_report(scorecard: ScorecardArtifacts) -> str:
    """
    Render the scorecard as Markdown.

    Base Score and scaling metadata are separated from the characteristic
    table so the report mirrors the conceptual score formula.
    """
    cfg = scorecard.config
    lines = [
        "## **Credit Scorecard**",
        "",
        "### **Scaling**",
        "",
        f"- Target score: **{cfg.target_score}** at odds **{cfg.target_odds}:1**",
        f"- PDO (Points to Double the Odds): **{cfg.pdo}**",
        "",
        "### **Base Score**",
        "",
        "| Component | Point |",
        "|---|---:|",
        f"| Intercept + offset | {scorecard.base_points} |",
        "",
        "### Characteristic Points",
        "",
    ]

    rows = _ordered_lookup_table(scorecard).to_dict(orient="records")
    columns = [
        ("Characteristic", "characteristic", "---"),
        ("Bin", "bin_label", "---"),
        ("Point", "point", "---:"),
        ("Count", "count", "---:"),
        ("WoE", "woe", "---:"),
        ("Coefficient", "coefficient", "---:"),
    ]
    lines.extend(_markdown_table(rows, columns))
    return "\n".join(lines)


# ============================================================================
# 6. Audit
# ============================================================================

def _audit_row(section: str, check: str, passed: bool, n_issues: int, details: str = "") -> dict:
    return {
        "section": section,
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "n_issues": int(n_issues),
        "details": details,
    }


def _missing_column_audit(lookup: pd.DataFrame, column: str) -> tuple:
    if column not in lookup.columns:
        return False, len(lookup), f"Column '{column}' is absent."
    missing = lookup[column].isna()
    return not missing.any(), int(missing.sum()), ""


def _finite_numeric_audit(lookup: pd.DataFrame, column: str) -> tuple:
    if column not in lookup.columns:
        return False, len(lookup), f"Column '{column}' is absent."
    numeric = pd.to_numeric(lookup[column], errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric)
    return not invalid.any(), int(invalid.sum()), ""


def audit_scorecard(scorecard: ScorecardArtifacts) -> pd.DataFrame:
    """
    Audit structural integrity of a scorecard artifact.

    This intentionally avoids Special/Missing consistency checks because
    those belong to `binning_woe.py`, where the fitted binning process and
    empirical transform behavior are already audited.
    """
    lookup = scorecard.lookup_table
    rows = []

    duplicated = lookup.duplicated(subset=["characteristic", "bin_label"], keep=False)
    rows.append(
        _audit_row(
            "Lookup",
            "duplicate characteristic + bin_label",
            not duplicated.any(),
            duplicated.sum(),
            str(lookup.loc[duplicated, ["characteristic", "bin_label"]].head(5).to_dict("records")),
        )
    )

    for column in ["point", "woe", "coefficient"]:
        passed, n_issues, details = _missing_column_audit(lookup, column)
        rows.append(_audit_row("Lookup", f"missing {column}", passed, n_issues, details))

    empty_characteristic = (
        lookup["characteristic"].isna()
        | lookup["characteristic"].astype(str).str.strip().eq("")
    )
    rows.append(
        _audit_row(
            "Mapping",
            "no empty characteristic",
            not empty_characteristic.any(),
            empty_characteristic.sum(),
        )
    )

    expected_features = set(_base_feature_names(scorecard.feature_names))
    lookup_features = set(lookup["characteristic"].dropna().astype(str))
    missing_feature_bins = sorted(expected_features - lookup_features)
    empty_feature_bins = lookup.groupby("characteristic", dropna=False).size().loc[lambda s: s < 1]
    rows.append(
        _audit_row(
            "Mapping",
            "all characteristics have at least one bin",
            not missing_feature_bins and empty_feature_bins.empty,
            len(missing_feature_bins) + len(empty_feature_bins),
            f"missing={missing_feature_bins}",
        )
    )

    point_type_ok = False
    if "point" in lookup.columns:
        point_values = lookup["point"].dropna()
        point_type_ok = point_values.map(lambda v: isinstance(v, (int, np.integer))).all()
    rows.append(
        _audit_row(
            "Point",
            "point type is integer",
            bool(point_type_ok),
            0 if point_type_ok else len(lookup),
        )
    )

    passed, n_issues, details = _finite_numeric_audit(lookup, "point")
    rows.append(_audit_row("Point", "point is finite and non-null", passed, n_issues, details))

    for column, section in [("woe", "WoE"), ("coefficient", "Coefficient")]:
        type_ok = False
        if column in lookup.columns:
            values = lookup[column].dropna()
            type_ok = values.map(lambda v: isinstance(v, (float, np.floating))).all()
        rows.append(
            _audit_row(
                section,
                f"{column} type is float",
                bool(type_ok),
                0 if type_ok else len(lookup),
            )
        )

        passed, n_issues, details = _finite_numeric_audit(lookup, column)
        rows.append(_audit_row(section, f"{column} is finite and non-null", passed, n_issues, details))

    rows.append(
        _audit_row(
            "Metadata",
            "lookup feature count equals feature_names",
            len(lookup_features) == len(expected_features),
            abs(len(lookup_features) - len(expected_features)),
            f"lookup={len(lookup_features)}, feature_names={len(expected_features)}",
        )
    )

    missing_lookup = sorted(expected_features - lookup_features)
    rows.append(
        _audit_row(
            "Metadata",
            "all feature_names have lookup",
            not missing_lookup,
            len(missing_lookup),
            f"missing={missing_lookup}",
        )
    )

    extra_lookup = sorted(lookup_features - expected_features)
    rows.append(
        _audit_row(
            "Metadata",
            "lookup has no extra feature",
            not extra_lookup,
            len(extra_lookup),
            f"extra={extra_lookup}",
        )
    )

    missing_coefficients = sorted(set(scorecard.feature_names) - set(scorecard.coefficients))
    rows.append(
        _audit_row(
            "Metadata",
            "all feature_names have coefficient",
            not missing_coefficients,
            len(missing_coefficients),
            f"missing={missing_coefficients}",
        )
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(USAGE)
