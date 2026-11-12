"""Probability calibration pipeline (raw / sigmoid / isotonic)."""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Callable, Optional, Union, List

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.evaluation import extract_positive_proba
from src.metrics import calculate_probability_metrics

logger = logging.getLogger(__name__)


def calibration_curve_table(
    y_true, y_proba, 
    n_bins=10, 
    strategy='quantile'
):
    """
    Generate a manual calibration curve table:
    - mean_predicted_proba
    - observed_event_rate
    - n_obs
    """
    df = pd.DataFrame({
        'y': np.asarray(y_true),
        'p': np.asarray(y_proba)
    }).copy()

    if strategy == 'quantile':
        # qcut can fail with many identical probabilities -> duplicates='drop'
        df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    elif strategy == 'uniform':
        df['bin'] = pd.cut(df['p'], bins=n_bins)
    else:
        raise ValueError("Error: strategy must be 'quantile' or 'uniform'")

    out = (
        df.groupby('bin', observed=False)
          .agg(
              mean_predicted_proba=('p', 'mean'),
              observed_event_rate=('y', 'mean'),
              n_obs=('y', 'size')
          )
          .reset_index(drop=True)
    )
    return out


# CALIBRATION STRATEGIES (Open/Closed)
# Type alias for calibration strategies
# Accepts (p_train_cal, y_train_cal, p_target) -> returns p_calibrated
CalibrateStrategy = Callable[[np.ndarray, pd.Series, np.ndarray], np.ndarray]

def _calibrate_sigmoid(p_train_cal: np.ndarray, y_train_cal: pd.Series, p_target: np.ndarray) -> np.ndarray:
    """Platt Scaling / Sigmoid Calibration using Logistic Regression."""
    calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, random_state=42)
    # Scikit-Learn LogisticRegression requires 2D input
    calibrator.fit(p_train_cal.reshape(-1, 1), y_train_cal)
    return calibrator.predict_proba(p_target.reshape(-1, 1))[:, 1]

def _calibrate_isotonic(p_train_cal: np.ndarray, y_train_cal: pd.Series, p_target: np.ndarray) -> np.ndarray:
    """Isotonic Regression Calibration (Non-parametric)."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    # IsotonicRegression accepts 1D arrays
    calibrator.fit(p_train_cal, y_train_cal)
    return calibrator.predict(p_target)

# Registry of available calibration methods
CALIBRATION_STRATEGIES: Dict[str, CalibrateStrategy] = {
    'sigmoid': _calibrate_sigmoid,
    'isotonic': _calibrate_isotonic
}


# CORE PIPELINE FUNCTION
def run_probability_pipeline(
    fitted_models: Dict[str, Any],
    data_splits: Dict[str, pd.DataFrame],
    y_train_cal: pd.Series,
    cal_strategies: Dict[str, CalibrateStrategy] = CALIBRATION_STRATEGIES
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract raw probabilities and perform automatic calibration for all models.

    Args:
        fitted_models (Dict[str, Any]): Dict contains `model_name` -> model object that has been fitted.
        data_splits (Dict[str, pd.DataFrame]): Dict containing 'train_cal', 'valid', and 'test' dataframes.
        y_train_cal (pd.Series): Actual target labels for the calibration subset.
        cal_strategies (Dict[str, CalibrateStrategy]): Calibration strategy registry.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Structured probability registry.
    """
    # Validate input data splits
    required_splits = {'train_cal', 'valid', 'test'}
    if not required_splits.issubset(data_splits.keys()):
        raise KeyError(f"data_splits must contain the following keys: {required_splits}")

    proba_store: Dict[str, Dict[str, np.ndarray]] = {}
    
    # --- STAGE 1: Raw probability extraction ---
    for model_name, model_obj in fitted_models.items():
        logger.info(f"Extracting raw probabilities for model: %s", model_name)
        
        # Create the key for this raw model version
        raw_key = f"{model_name}"
        proba_store[raw_key] = {}
        
        for split_name, X_data in data_splits.items():
            proba_store[raw_key][split_name] = extract_positive_proba(model_obj, X_data)
            
        # --- STAGE 2: Automatic calibration (Sigmoid & Isotonic) ---
        p_train_cal_raw = proba_store[raw_key]['train_cal']
        
        for strategy_name, strategy_func in cal_strategies.items():
            cal_key = f"{model_name}_cal_{strategy_name}"
            proba_store[cal_key] = {}
            
            logger.info(f"Applying %s calibration for model: %s", strategy_name, model_name)
            
            for split_name, _ in data_splits.items():
                p_target_raw = proba_store[raw_key][split_name]
                
                # Execute calibration via the corresponding strategy function
                proba_store[cal_key][split_name] = strategy_func(
                    p_train_cal_raw, 
                    y_train_cal, 
                    p_target_raw
                )
                
    logger.info(f"Probability extraction and calibration pipeline complete for all models.")
    return proba_store




def evaluate_calibration_candidates(
    proba_store: Dict[str, Dict[str, np.ndarray]],
    y_valid: pd.Series
) -> pd.DataFrame:
    """
    Computes probability metrics for calibration candidates on validation data.
    
    Args:
        proba_store (Dict[str, Dict[str, np.ndarray]]): Output from run_probability_pipeline.
        y_valid (pd.Series): Target label of subset validation.
        
    Returns:
        pd.DataFrame: Leaderboard of calibration results on data validation.
    """
    cal_metrics = []
    
    # Looping through all model versions (raw, cal_sigmoid, cal_isotonic) in proba_store
    for model_version, splits_proba in proba_store.items():
        if 'valid' not in splits_proba:
            continue
            
        p_valid = splits_proba['valid']
        metrics = calculate_probability_metrics(y_valid, p_valid)
        
        row = {'model_version': model_version}
        row.update(metrics)
        cal_metrics.append(row)
        
    return pd.DataFrame(cal_metrics)


def should_calibrate(cal_leaderboard: pd.DataFrame, base_model_name: str) -> Tuple[bool, str, pd.Series]:
    """
    Determines whether calibration provides performance improvement based on the lowest Brier Score.
    
    Args:
        cal_leaderboard (pd.DataFrame): Results dataframe from `evaluate_calibration_candidates`.
        base_model_name (str): Base model name (example: 'xgb_raw').
        
    Returns:
        Tuple[bool, str, pd.Series]:
            - Status whether calibration is needed (True/False).
            - Winning version name (eg: 'xgb_cal_isotonic' or 'xgb_raw').
            - The metric line from the winning version.
    """
    # Sort by lowest Brier Score (main calibration criteria)
    sorted_df = cal_leaderboard.sort_values(by='brier_score', ascending=True)
    best_row = sorted_df.iloc[0]
    best_version = best_row['model_version']
    
    # Check if the winner is the 'raw' version
    raw_version_name = f"{base_model_name}"
    needs_calibration = best_version != raw_version_name
    
    return needs_calibration, best_version, best_row



# CALIBRATION DIAGNOSTICS & LINEAR (LOGIT) RECALIBRATION
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def calibration_slope_intercept(y_true, p):
    """Fit logit(y) ~ a + b * logit(p) on the given data.

    Diagnostic, not a production calibrator: slope b < 1 means scores
    are overconfident (too extreme), b > 1 underconfident; intercept a
    captures level shift. Fit on a held-out split, interpret on it.
    """
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000,
                            random_state=42)
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y_true))
    return {'intercept': float(lr.intercept_[0]),
            'slope': float(lr.coef_[0][0])}


def fit_logit_recalibration(p_cal, y_cal):
    """Fit a + b * logit(p) -> probability map on a calibration set."""
    d = calibration_slope_intercept(y_cal, p_cal)
    return d['intercept'], d['slope']


def apply_logit_recalibration(p, intercept, slope):
    return _expit(intercept + slope * _logit(p))


def expected_calibration_error(y_true, p, n_bins=10):
    """ECE: sum over bins of (n_bin/N) * |empirical - predicted|."""
    y = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    bins = pd.qcut(p, q=n_bins, duplicates='drop')
    df = pd.DataFrame({'y': y, 'p': p, 'bin': bins})
    g = df.groupby('bin', observed=True).agg(
        emp=('y', 'mean'), pred=('p', 'mean'), n=('y', 'size'))
    return float(((g['emp'] - g['pred']).abs() * g['n']).sum() / len(y))


# CALIBRATION STRATEGY SELECTION & DIAGNOSTICS


@dataclass
class CalibrationEvaluation:
    """Per-(model, strategy, split) calibration quality summary."""

    table: pd.DataFrame       # long-form: model, strategy, split, brier, logloss
    winners: pd.DataFrame      # one row per base model: winning strategy on decision_split


def evaluate_calibration_strategies(
    proba_store: Dict[str, Dict[str, np.ndarray]],
    y_by_split: Dict[str, pd.Series],
    base_model_names: List[str],
    decision_split: str = "valid",
) -> CalibrationEvaluation:
    """
    Score every (model, calibration strategy) combination in `proba_store`
    on Brier Score and LogLoss across all splits, and pick the winner per
    base model using ONLY the `decision_split` (the validation set -- never
    'train_cal', since that split was used to FIT the calibrators and
    would give an optimistically biased comparison; OOT test is held out).

    Parameters
    ----------
    proba_store : dict
        Output of `run_probability_pipeline`. Keys are either the raw
        model name (e.g. "logreg_scorecard") or
        "{model_name}_cal_{strategy}" (e.g. "logreg_scorecard_cal_isotonic").
        Each value is a dict of {split_name: np.ndarray of probabilities}.
    y_by_split : dict
        {split_name: true binary labels}, covering every split present
        in `proba_store` (typically 'train_cal', 'valid', 'test').
    base_model_names : list of str
        The raw model name(s) as passed to `fitted_models` in
        `run_probability_pipeline` (without any "_cal_*" suffix).
    decision_split : str
        Split used to decide the winning strategy. Defaults to "valid"
        (the temporal validation partition) -- deliberately never
        "train_cal" or "test".

    Returns
    -------
    CalibrationEvaluation
        `.table`: full long-form Brier/LogLoss breakdown for every
        (model, strategy, split) combination -- useful for plotting
        calibration degradation across splits.
        `.winners`: one row per base model with the strategy
        ('raw' / 'sigmoid' / 'isotonic') that minimizes Brier Score on
        `decision_split`, plus that strategy's Brier/LogLoss for
        transparency.
    """
    rows = []
    for base_name in base_model_names:
        strategy_keys = {"raw": base_name}
        for key in proba_store:
            if key.startswith(f"{base_name}_cal_"):
                strategy_name = key[len(f"{base_name}_cal_"):]
                strategy_keys[strategy_name] = key

        for strategy_name, store_key in strategy_keys.items():
            for split_name, p_pred in proba_store[store_key].items():
                y_true = y_by_split[split_name]
                rows.append(
                    {
                        "model": base_name,
                        "strategy": strategy_name,
                        "split": split_name,
                        "brier": brier_score_loss(y_true, p_pred),
                        "logloss": log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6)),
                    }
                )

    table = pd.DataFrame(rows)

    decision_rows = table[table["split"] == decision_split].copy()
    winners = (
        decision_rows.sort_values("brier")
        .groupby("model", as_index=False)
        .first()[["model", "strategy", "brier", "logloss"]]
        .rename(columns={"strategy": "winning_strategy",
                          "brier": f"brier_{decision_split}",
                          "logloss": f"logloss_{decision_split}"})
    )

    return CalibrationEvaluation(table=table, winners=winners)


def brier_skill_score(y_true: pd.Series, p_pred: np.ndarray) -> dict:
    """
    Brier Skill Score: how much better the model's Brier Score is than a
    TRIVIAL baseline that predicts the population base rate for every
    row (`p_baseline = y_true.mean()` everywhere).

    Rationale: raw Brier Score / LogLoss are dominated by class
    prevalence -- a low base rate mechanically produces a low (good-
    looking) Brier Score even for a model with near-zero discriminative
    skill, since `Brier_trivial = p*(1-p)` is small whenever `p` is far
    from 0.5. Comparing splits by raw Brier Score alone can therefore
    show "improvement" that is actually just a lower base rate in that
    split, not better probabilistic accuracy. This function makes the
    comparison base-rate-invariant.

    Returns
    -------
    dict
        - 'base_rate': observed y_true.mean()
        - 'brier_model': the model's actual Brier Score
        - 'brier_trivial': Brier Score of the constant-base-rate predictor
        - 'skill_score': 1 - brier_model/brier_trivial
          (0 = no better than guessing the base rate; 1 = perfect;
          negative = worse than the trivial baseline)
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred)
    base_rate = y.mean()
    brier_model = brier_score_loss(y, p)
    brier_trivial = base_rate * (1 - base_rate)
    skill = 1 - (brier_model / brier_trivial) if brier_trivial > 0 else np.nan
    return {
        "base_rate": base_rate,
        "brier_model": brier_model,
        "brier_trivial": brier_trivial,
        "skill_score": skill,
    }


def reliability_table(y_true: pd.Series, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Decile-based reliability (calibration curve) table: predicted vs.
    observed bad rate per probability decile, for visual inspection
    alongside the scalar Brier/LogLoss summary above.

    Returns
    -------
    pd.DataFrame
        ['decile', 'count', 'mean_predicted', 'observed_rate'].
        A well-calibrated model has `mean_predicted` ~= `observed_rate`
        in every row.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_pred)})
    df["decile"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (
        df.groupby("decile", observed=True)
        .agg(count=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
