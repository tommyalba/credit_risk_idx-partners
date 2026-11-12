# Libraries
import pandas as pd
import numpy as np
import shap
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

"""
SHAP (SHapley Additive exPlanations) interpretability for binary
credit-risk XGBClassifier models.

References:
    Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
        Interpreting Model Predictions. NeurIPS.
    Lundberg, S. M. et al. (2020). From Local Explanations to Global
        Understanding with Explainable AI for Trees. Nature Machine
        Intelligence.

Module coverage:
    1. Shapley value computation via TreeSHAP (exact, O(T*L*D^2))
    2. Global diagnostics   -- summary plot (feature importance & direction)
    3. Relational diagnostics -- dependence plot (monotonicity vs domain)
    4. Local explanations    -- per-individual waterfall/force plots,
                                for "adverse action notice" use cases
    5. Axiomatic validation  -- numerical check of the efficiency property
    6. Consistency audit     -- SHAP importance vs XGBoost's built-in
                                gain-based feature_importances

Important note on output scale:
    For objective="binary:logistic", TreeSHAP returns contributions on
    the LOG-ODDS scale (pre-sigmoid) by default, NOT the probability
    scale directly -- a consequence of the axiomatic additivity
    ("efficiency" property) which only holds linearly on the model's
    margin scale, not on the non-linear probability scale.
    Interpretation on the probability scale requires an additional
    non-linear transform (see shap.Explanation with link="logit" on
    some plots, or manual interpretation).
"""

def compute_shap_values(model, X, feature_names=None):
    """
    Compute exact Shapley values via the TreeSHAP algorithm for
    tree-based models (XGBoost, LightGBM, CatBoost, RandomForest, etc.).

    Parameters
    ----------
    model : XGBClassifier (or another tree-based model supported by shap.TreeExplainer)
        A fitted model.
    X : pd.DataFrame or array-like
        Data on which SHAP is computed -- use a representative subset
        (e.g. X_valid_clean or a sample of X_test_clean) when n is large,
        because even though TreeSHAP is polynomial, cost still grows with n.
    feature_names : list, optional
        Required when X is not a DataFrame (e.g. a numpy array).

    Returns
    -------
    shap.Explanation object -- the unified modern shap structure (>=0.40)
    carrying .values (per-feature contribution per observation),
    .base_values (E[f(X)], the base expectation), and .data (raw feature values).
    """
    if not isinstance(X, pd.DataFrame) and feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    return explanation

def validate_efficiency_axiom(explanation, model, X, atol=1e-3):
    """
    Numerically validate the 'efficiency' axiom: the sum of all phi_i
    plus base_value must equal the model's raw output (log-odds/margin
    scale) for every observation.

        f(x) = base_value + sum_i phi_i

    This is not merely a unit test -- it verifies that the SHAP
    decomposition is genuinely 'additive' as guaranteed axiomatically
    by Shapley value theory, not a diverging approximation.
    """
    
    if hasattr(model, 'predict'):
        model_classname = type(model).__name__
        if 'XGB' in model_classname:
            raw_margin = model.predict(X, output_margin=True)
        elif 'CatBoost' in model_classname:
            # CatBoost uses prediction_type='RawFormulaVal' for margin/log-odds scale
            raw_margin = model.predict(X, prediction_type='RawFormulaVal')
        elif 'LGBM' in model_classname:
            raw_margin = model.predict(X, raw_score=True)
        else:
            raise TypeError(f"Model type {model_classname} is not supported for margin validation.")
    else:
        raise TypeError("The provided object is not a valid estimator model.")
        
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)

    max_diff = np.max(np.abs(raw_margin - reconstructed))
    is_valid = max_diff < atol

    return {
        'max_absolute_difference': float(max_diff),
        'axiom_satisfied': bool(is_valid),
        'interpretation': (
            "Efficiency axiom verified -- the SHAP decomposition is "
            "exactly additive with respect to the model's margin output."
            if is_valid else
            "WARNING: difference exceeds tolerance. Check for possible "
            "prediction non-determinism (e.g. GPU vs CPU) or a shap version "
            "incompatible with the xgboost version."
        ),
    }

def plot_global_importance(explanation, max_display=None, plot_type="dot"):
    """
    Summary plot: feature ranking by mean |SHAP value|, with a
    color-coded scatter (red=high feature value, blue=low) showing the
    DIRECTION of influence -- unlike XGBoost's built-in gain-based
    feature_importance which only ranks without direction.

    Parameters
    ----------
    plot_type : str
        "dot"  -- per-observation spread (most informative, default)
        "bar"  -- mean |SHAP| only (more compact for executive reports)
    """
    display_count = max_display if max_display is not None else 20
    fig = plt.figure(figsize=(9, max(5, display_count * 0.35)))
    shap.summary_plot(
        explanation, max_display=max_display, plot_type=plot_type, show=False
    )
    plt.tight_layout()
    return fig

def get_global_importance_table(explanation, feature_names):
    """
    Feature importance table based on mean |SHAP value| -- the numeric
    counterpart of the summary plot, for tabular reporting and for
    explicit comparison against XGBoost's gain-based importance.
    """
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df

def compare_shap_vs_gain_importance(explanation, model, feature_names):
    """
    Compare SHAP-importance ranking against XGBoost's built-in
    gain-based feature_importances, with the Spearman rank correlation
    between them -- confirming or diagnosing the divergence noted in
    earlier analysis: gain-based importance has no axiomatic guarantee
    and is prone to bias toward high-cardinality features, so a
    significant ranking divergence deserves investigation rather than
    being ignored.
    """

    shap_df = get_global_importance_table(explanation, feature_names)
    gain_importance = model.feature_importances_

    gain_df = pd.DataFrame({
        'feature': feature_names,
        'gain_importance': gain_importance,
    }).sort_values('gain_importance', ascending=False).reset_index(drop=True)
    gain_df['rank_gain'] = np.arange(1, len(gain_df) + 1)

    merged = shap_df.merge(gain_df, on='feature')
    merged = merged.rename(columns={'rank': 'rank_shap'})

    rho, pvalue = spearmanr(merged['rank_shap'], merged['rank_gain'])

    return {
        'comparison_table': merged.sort_values('rank_shap'),
        'spearman_rho': rho,
        'interpretation': (
            f"Spearman rank correlation = {rho:.3f}. "
            + (
                "High concordance between the two methods -- findings reinforce each other."
                if rho > 0.7 else
                "\nSignificant divergence -- investigate features with large "
                "rank differences; possible cardinality bias in gain-based importance."
            )
        ),
    }

def plot_dependence(explanation, feature_name, interaction_feature="auto"):
    """
    Dependence plot: visualize a feature's SHAP value (y-axis, log-odds
    contribution) against that feature's value (x-axis), colored by the
    strongest automatically detected interaction feature.

    Main diagnostic goal: verify whether the feature-risk relationship
    is MONOTONIC-CONSISTENT-WITH-DOMAIN-INTUITION (e.g. a higher
    debt-to-income ratio -> consistently rising risk contribution) or
    whether the model learned suspicious non-monotonic patterns that
    warrant further investigation (possible proxy bias, leakage, or
    data artifacts).
    """
    fig = plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feature_name, explanation.values, explanation.data,
        feature_names=explanation.feature_names,
        interaction_index=interaction_feature, show=False
    )
    plt.tight_layout()
    return fig

def check_monotonicity(explanation, feature_name, expected_direction="increasing"):
    """
    Quantitative monotonicity check: compute the Spearman correlation
    between a feature's value and its SHAP value. Spearman (not Pearson)
    is chosen because it tests pure rank-based monotonicity without
    assuming a linear relationship -- consistent with the non-parametric
    nature of tree-based models themselves.

    Parameters
    ----------
    expected_direction : str
        "increasing" -- high feature values should raise risk (positive SHAP)
        "decreasing" -- high feature values should lower risk (negative SHAP)
    """
    feature_idx = list(explanation.feature_names).index(feature_name)
    raw_feature_values = explanation.data[:, feature_idx]
    shap_values_clean = np.asarray(explanation.values[:, feature_idx], dtype=float).flatten()

    # Clean feature_values: anticipate data stored as object/string
    feature_ser = pd.Series(raw_feature_values)

    # Use a broad check: if not numeric
    if not pd.api.types.is_numeric_dtype(feature_ser):
        try:
            # Try direct conversion to float if originally numeric but string/object typed
            feature_values_clean = feature_ser.astype(float).to_numpy()
        except ValueError:
            # If that fails (e.g. category text like '20p16_to_21p91'),
            # convert to an ordinal ranking.
            # sort=True is CRITICAL for binning so the order isn't scrambled.
            if isinstance(feature_ser.dtype, pd.CategoricalDtype):
                # Already a pandas ordinal category -- use cat.codes
                feature_values_clean = feature_ser.cat.codes.to_numpy()
            else:
                # Fallback for plain strings
                feature_values_clean = pd.factorize(feature_ser, sort=True)[0]
    else:
        feature_values_clean = np.asarray(raw_feature_values, dtype=float).flatten()

    # Run spearmanr on arrays guaranteed to be pure 1D numeric
    rho, pvalue = spearmanr(feature_values_clean, shap_values_clean)

    direction_observed = 'increasing' if rho > 0 else 'decreasing'
    consistent = direction_observed == expected_direction

    return {
        'feature': feature_name,
        'spearman_rho': round(rho, 4),
        'p_value': pvalue,
        'direction_observed': direction_observed,
        'direction_expected': expected_direction,
        'consistent_with_domain_prior': consistent,
        'interpretation': (
            f"A {direction_observed} monotonic relationship is confirmed "
            f"(rho={rho:.3f}), {'consistent' if consistent else 'CONTRADICTING'} "
            f"domain expectations. "
            + ("" if consistent else
               "Further investigation required -- check the dependence plot "
               "for local non-monotonic patterns or feature interactions reversing direction.")
        ),
    }

def explain_individual_prediction(explanation, index, feature_names=None, top_n=10):
    """
    Additive decomposition for ONE individual observation -- a structure
    directly translatable into a quantitatively auditable 'reason for
    decision' statement (relevant to transparency obligations such as
    the ECOA Adverse Action Notice or OJK consumer-finance protection
    rules).

    Returns
    -------
    DataFrame with the top_n largest contributors (by |SHAP value|)
    to that observation's prediction, sorted by contribution, with
    direction (raising/lowering the risk score).
    """
    values = explanation.values[index]
    data_row = explanation.data[index]
    names = feature_names or explanation.feature_names

    df = pd.DataFrame({
        "feature": names,
        "feature_value": data_row,
        "shap_value": values,
        "abs_shap_value": np.abs(values),
    }).sort_values("abs_shap_value", ascending=False).head(top_n)

    df['direction'] = np.where(
        df['shap_value'] > 0, "raises risk (pushes toward BAD)",
        "lowers risk (pushes toward GOOD)"
    )

    base_value = explanation.base_values[index]
    final_margin = base_value + values.sum()
    final_proba = 1 / (1 + np.exp(-final_margin))

    return {
        'table': df.reset_index(drop=True),
        'base_value_log_odds': round(float(base_value), 4),
        'final_margin_log_odds': round(float(final_margin), 4),
        'final_probability': round(float(final_proba), 4),
    }

def plot_individual_waterfall(explanation, index, max_display=12):
    """
    Waterfall plot for a single observation -- the standard visual
    representation of a local explanation, showing sequentially how
    each feature shifts the prediction from base_value (E[f(X)] over
    the training population) toward f(x) for this specific observation.
    """
    fig = plt.figure(figsize=(8, max(5, max_display * 0.4)))
    shap.plots.waterfall(explanation[index], max_display=max_display, show=False)
    plt.tight_layout()
    return fig

def plot_individual_force(explanation, index):
    """
    Force plot -- an alternative local-explanation representation in
    horizontal format, often easier for non-technical stakeholders to
    read than the waterfall because it resembles a two-way 'tug-of-war'.
    """
    return shap.plots.force(
        explanation.base_values[index],
        explanation.values[index],
        explanation.data[index],
        feature_names=explanation.feature_names,
    )
