"""Multi-split evaluation of fitted candidate models."""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from src.metrics import calculate_probability_metrics

logger = logging.getLogger(__name__)

# ==========================================
# 2. MULTI-MODEL MULTI-SPLIT EVALUATOR
# ==========================================

def evaluate_base_models_pipeline(
    fitted_models: Dict[str, Any],
    X_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all fitted base models on all split data sets (Train, Valid, Test).

    Architecture Note:
    Models trained with SMOTE (_smote,) are intentionally evaluated against the original 
    X_train_fit (not the SMOTE version). This is necessary so that performance on the 
    training subset reflects the model's ability to handle the real population distribution,
    not synthetic data.

    Args:
        fitted_models (Dict[str, Any]): Dictionary contains `model_name` -> fitted estimator object.
        X_train_fit (pd.DataFrame): Original subset training feature.
        y_train_fit (pd.Series): Target training original subset.
        X_valid (pd.DataFrame): Subset validation feature.
        y_valid (pd.Series): Target validation subset.
        X_test (pd.DataFrame): Subset testing feature.
        y_test (pd.Series): Target testing subset.

    Returns:
        pd.DataFrame: Long-Format structured DataFrame for leaderboard analysis.
    """
    logger.info(f"Starting base model probability metrics evaluation pipeline...")
    
    # Wrap split into a regular data structure for automatic looping.
    splits_config = [
        ('1_train_fit', X_train_fit, y_train_fit),
        ('2_valid', X_valid, y_valid),
        ('3_test', X_test, y_test)
    ]
    
    evaluation_rows: List[Dict[str, Any]] = []
    
    for model_name, model_obj in fitted_models.items():
        if not hasattr(model_obj, "predict_proba"):
            logger.error(f"Model '%s' does not have 'predict_proba' method. Skipping.", model_name)
            continue
            
        logger.info(f"Evaluating probability metrics for model: %s", model_name)
        
        for split_name, X_data, y_true in splits_config:
            # Safely extract the probability of the positive class (index column 1)
            y_proba = model_obj.predict_proba(X_data)[:, 1]
            
            # Calculate metrics
            metrics = calculate_probability_metrics(y_true, y_proba)
            
            # Merge model metadata and split into result rows
            row_entry = {
                'model_name': model_name,
                'split': split_name
            }
            row_entry.update(metrics)
            
            evaluation_rows.append(row_entry)
            
    logger.info(f"Base model evaluation complete.")
    
    # Convert the resulting list into a final DataFrame
    result_df = pd.DataFrame(evaluation_rows)
    return result_df


# CORE UTILITY FUNCTIONS
# ============================================================
def extract_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Safely extract the probability for the positive class (class 1).

    Args:
        model (Any): Fitted estimator model.
        X (pd.DataFrame): Dataset features.

    Returns:
        np.ndarray: 1D Array of positive class probabilities.
    """
    if not hasattr(model, 'predict_proba'):
        raise AttributeError(f"Model {type(model).__name__} tidak memiliki method 'predict_proba'.")
    return model.predict_proba(X)[:, 1]
