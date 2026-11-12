"""Estimator construction, family-specific fitting, candidate registry,
winner selection and model serialization."""
import os
import copy
import logging
from typing import Any, Dict, Tuple, Callable, Optional, Union, List

import joblib
import pandas as pd

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


# ============================================================
# Safe extraction of best iteration
# ============================================================
def safe_get_best_iteration(estimator: Any) -> Optional[int]:
    """Safely extract the best iteration from a fitted estimator."""
    if hasattr(estimator, 'best_iteration'):
        return estimator.best_iteration
    elif hasattr(estimator, 'best_iteration_'):
        return estimator.best_iteration_
    return None

def safe_get_best_score(estimator: Any) -> Optional[float]:
    """Safely extract the best score from a fitted estimator."""
    if hasattr(estimator, 'best_score'):
        return estimator.best_score
    elif hasattr(estimator, 'best_score_'):
        return estimator.best_score_
    return None


# ============================================================
# ESTIMATOR BUILDER (Consolidated)
# ============================================================
def build_estimator(estimator_class: type, params: Dict[str, Any]) -> Any:
    """
    Creates an instance of an estimator dynamically with safe deep-copied parameters.

    Args:
        estimator_class (type): The class of the ML model (e.g., XGBClassifier).
        params (Dict[str, Any]): Dictionary of hyperparameters.

    Returns:
        Any: Unfitted estimator instance.
        
    Raises:
        TypeError: If params is not a dictionary.
    """
    if not isinstance(params, dict):
        raise TypeError(f"Parameters must be a dictionary, got {type(params)} instead.")
    return estimator_class(**copy.deepcopy(params))

# ============================================================
# FITTING STRATEGIES (Open/Closed Principle)
# ============================================================
# Type alias to make it easier to read the fit function data type
FitStrategy = Callable[[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], Any]

def _fit_xgb_strategy(estimator: Any, X_fit: pd.DataFrame, y_fit: pd.Series, X_es: pd.DataFrame, y_es: pd.Series) -> Any:
    """Fit strategy for XGBoost models."""
    estimator.fit(
        X_fit, y_fit, 
        eval_set=[(X_es, y_es)], 
        verbose=False
    )
    return estimator

def _fit_cat_strategy(estimator: Any, X_fit: pd.DataFrame, y_fit: pd.Series, X_es: pd.DataFrame, y_es: pd.Series) -> Any:
    """Fit strategy for CatBoost models."""
    estimator.fit(
        X_fit, y_fit, 
        eval_set=(X_es, y_es), 
        verbose=False, 
        use_best_model=True
    )
    return estimator

def _fit_lgbm_strategy(estimator, X_fit: pd.DataFrame, y_fit: pd.Series, X_es: pd.DataFrame, y_es: pd.Series) -> Any:
    """LightGBM specific fit strategy with early stopping."""
    # LightGBM uses callbacks for early stopping in modern versions
    from lightgbm import early_stopping, log_evaluation
    estimator.fit(
        X_fit, y_fit, 
        eval_set=[(X_es, y_es)], 
        callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)]
    )
    return estimator

# Registry Strategy: Jika ingin menambah model (misal LightGBM), cukup tambahkan fungsi dan daftarkan di sini.
FIT_STRATEGIES: Dict[str, FitStrategy] = {
    'xgb': _fit_xgb_strategy,
    'cat': _fit_cat_strategy,
    'lgbm': _fit_lgbm_strategy,
}

# ============================================================
# CORE FIT MODULE
# ============================================================
def fit_candidate_model(
    model_name: str, 
    family: str,
    estimator: Any, 
    X_fit: pd.DataFrame, 
    y_fit: pd.Series, 
    X_es: pd.DataFrame, 
    y_es: pd.Series,
    strategies: Dict[str, FitStrategy] = FIT_STRATEGIES
) -> Any:
    """
    Fits a single model utilizing the appropriate family-specific strategy.

    Args:
        model_name (str): Identifier for the model.
        family (str): The family of the model ('xgb', 'cat', etc.) to determine the fit strategy.
        estimator (Any): The instantiated, unfitted ML model.
        X_fit (pd.DataFrame): Training features.
        y_fit (pd.Series): Training target.
        X_es (pd.DataFrame): Early stopping validation features.
        y_es (pd.Series): Early stopping validation target.
        strategies (Dict[str, FitStrategy]): Mapping of model families to their specific fit functions.

    Returns:
        Any: Fitted estimator.

    Raises:
        ValueError: If the defined model family is not supported in the strategies dictionary.
    """
    logger.info(
        f"Fitting %-15s | train shape=%s | early-stop valid shape=%s",
        model_name, X_fit.shape, X_es.shape
    )

    if family not in strategies:
        raise ValueError(
            f"Unsupported model family '{family}' for {model_name}. "
            f"Available families: {list(strategies.keys())}"
        )

    # Delegate fitting execution to the specific strategy
    fit_function = strategies[family]
    fitted_estimator = fit_function(estimator, X_fit, y_fit, X_es, y_es)

    best_iter = safe_get_best_iteration(fitted_estimator)
    best_score = safe_get_best_score(fitted_estimator)

    logger.info(
        f"Done fitting %-15s | best_iteration=%s | best_score=%s",
        model_name, str(best_iter), str(best_score)
    )
    logger.info("-" * 54)
    
    return fitted_estimator

# ============================================================
# PIPELINE ORCHESTRATION
# ============================================================
def build_candidate_registry(
    xgb_params: Dict[str, Any], 
    cat_params: Dict[str, Any],
    lgbm_params: Dict[str, Any],
    enable_blsmote: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Builds the model candidate registry defining families and SMOTE usage.

    Args:
        xgb_params (Dict[str, Any]): Dictionary of parameters for XGBoost.
        cat_params (Dict[str, Any]): Dictionary of parameters for CatBoost.
        lgbm_params (Dict[str, Any]): Dictionary of parameters for LGBM.

    Returns:
        Dict[str, Dict[str, Any]]: The configuration registry.
    """
    registry = {
        'xgb_raw': {
            'family': 'xgb',
            'use_blsmote': False,
            'estimator': build_estimator(XGBClassifier, xgb_params),
        },
        'cat_raw': {
            'family': 'cat',
            "use_blsmote": False,
            'estimator': build_estimator(CatBoostClassifier, cat_params),
        },
        'lgbm_raw': {
            'family': 'lgbm',         # Must match the key in FIT STRATEGIES
            'use_blsmote': False,
            'estimator': build_estimator(LGBMClassifier, lgbm_params)
        },
    }
    if enable_blsmote:
        registry.update({
            'xgb_blsmte': {
                'family': 'xgb',
                'use_blsmote': True,
                'estimator': build_estimator(XGBClassifier, xgb_params),
            },
            'cat_blsmte': {
                'family': 'cat',
                'use_blsmote': True,
                'estimator': build_estimator(CatBoostClassifier, cat_params),
            },
            'lgbm_blsmte': {
                'family': 'lgbm',
                'use_blsmote': True,
                'estimator': build_estimator(LGBMClassifier, lgbm_params)
            },
        })

    return registry


def fit_all_candidate_models(
    registry: Dict[str, Dict[str, Any]],
    X_train_fit: pd.DataFrame, 
    y_train_fit: pd.Series,
    X_train_fit_blsmte: Optional[pd.DataFrame] = None,
    y_train_fit_blsmte: Optional[pd.Series] = None,
    X_train_es: Optional[pd.DataFrame] = None, 
    y_train_es: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Iterates through the model registry and fits all candidates using appropriate datasets.

    Args:
        registry (Dict[str, Dict[str, Any]]): Defined model configuration registry.
        X_train_fit (pd.DataFrame): Original training features.
        y_train_fit (pd.Series): Original training target.
        X_train_fit_blsmte (pd.DataFrame): SMOTE-resampled training features.
        y_train_fit_blsmte (pd.Series): SMOTE-resampled training target.
        X_train_es (pd.DataFrame): Early stopping validation features.
        y_train_es (pd.Series): Early stopping validation target.

    Returns:
        Dict[str, Any]: A dictionary mapping model names to their fitted estimators.
    """
    fitted_models: Dict[str, Any] = {}

    for model_name, spec in registry.items():
        estimator = spec['estimator']
        use_blsmote = spec.get('use_blsmote', False)  # default False
        family = spec['family']

        # Determine which dataset to use based on configuration (Pure Function logic)
        if use_blsmote and X_train_fit_blsmte is not None and y_train_fit_blsmte is not None:
            X_fit_use = X_train_fit_blsmte
            y_fit_use = y_train_fit_blsmte
        else:
            X_fit_use = X_train_fit
            y_fit_use = y_train_fit

        logger.info(
            f"Stage 0 fit | model=%s | family=%s | use_blsmote=%s",
            model_name, family, use_blsmote
        )

        fitted_estimator = fit_candidate_model(
            model_name=model_name,
            family=family,
            estimator=estimator,
            X_fit=X_fit_use,
            y_fit=y_fit_use,
            X_es=X_train_es,
            y_es=y_train_es
        )
        
        fitted_models[model_name] = fitted_estimator

    return fitted_models


# ============================================================
# MODEL SERIALIZATION UTILITIES
# ============================================================
def save_best_model(model_obj: Any, filepath: str) -> None:
    """Save ML model objects to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)


# ============================================================
# STAGE A WINNER SELECTOR
# ============================================================
def select_stage_a_winner(
    leaderboard_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
    manual_winner: Optional[str] = None
) -> Tuple[str, Any, pd.DataFrame]:
    """
    Select the best model from Stage A based on performance on validation data, 
    or use a manually selected model.

    Automatic Criteria:
    1. Highest ROC-AUC (Descending)
    2. Lowest Log Loss (Ascending)

    Args:
        leaderboard_df (pd.DataFrame): DataFrame resulting from `evaluate_base_models_pipeline`.
        fitted_models (Dict[str, Any]): The dictionary contains the model objects that have been fitted.
        manual_winner (Optional[str]): Model name if you want to override automatic selection.

    Returns:
        Tuple[str, Any, pd.DataFrame]: 
            - Name of the winning model.
            - Winning model object.
            - Performance metrics DataFrame for the model across splits.
    """
    if manual_winner is not None:
        if manual_winner not in fitted_models:
            raise ValueError(f"Manual winner '{manual_winner}' not found in model dictionary.")
        winner_name = manual_winner
    else:
        # Filter only data validation
        valid_df = leaderboard_df[leaderboard_df['split'] == '2_valid'].copy()
        
        if valid_df.empty:
            raise ValueError("There is no data with split='2_valid' on the leaderboard.")
            
        # Sort by highest ROC-AUC, then lowest LogLoss
        sorted_valid = valid_df.sort_values(
            by=['roc_auc', 'log_loss'],
            ascending=[False, True]
        )
        
        # Take the top line (winner)
        winner_name = sorted_valid.iloc[0]['model_name']
        
    winner_model = fitted_models[winner_name]

    # Extract ALL split rows (train, valid, test) for the winning model
    winner_metrics = leaderboard_df[
        leaderboard_df['model_name'] == winner_name
    ].copy()
    
    # Sort by split so that the order is always train -> valid -> test
    winner_metrics = winner_metrics.sort_values(by='split').reset_index(drop=True)
    
    return winner_name, winner_model, winner_metrics
