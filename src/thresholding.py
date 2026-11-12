"""Decision-threshold optimization (F1 / Youden / expected cost)."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from sklearn.metrics import precision_recall_curve, roc_curve

from src.metrics import threshold_classification_metrics

# FIND BEST COST-BASED THRESHOLD
def find_best_cost_threshold(
    y_true, y_proba,
    c_fn=None, c_fp=None):
    from src.expected_cost_threshold import expected_cost_curve_scalar
    """
    Threshold that minimizes: 
    
    E[C(tau)] = c_fn * P(y=1) * FNR(tau) + c_fp * P(y=0) * FPR(tau)
    
    The only criterion that is genuinely independent of F1 and 
    Youden is that it is based on economic objective functions, 
    not discrimination or precision-recall statistics.
    """
    # Economic parameters:    
    if c_fn is None:
        c_fn = 8_250_000.0  # EAD * LGD = 15jt * 0.55
    
    if c_fp is None:
        c_fp = 2_700_000.0  # EAD * spread * durasi = 15jt * 0.09 * 2

    result = expected_cost_curve_scalar(
        y_true, y_proba, c_fn=c_fn, c_fp=c_fp
    )
    # objective_value = E[C] minimum (dalam skala biaya relatif)
    return result['optimal_threshold'], result['optimal_cost']

# FIND BEST F1 THRESHOLD
def find_best_f1_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        raise ValueError("Error: Thresholds are 0")

    # Align arrays
    precision = precision[:-1]
    recall = recall[:-1]
    
    f1_scores = np.where(
        (precision + recall) > 0, 
        2 * precision * recall / (precision + recall), 
        0
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# FIND BEST YOUDEN J THRESHOLD
def find_best_youden_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if len(thresholds) == 0:
        raise ValueError("Error: Thresholds are 0")

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]

# FIND OPTIMAL THRESHOLD
def find_optimal_threshold(y_true, y_proba, objective='cost'):
    if objective == 'f1':
        return find_best_f1_threshold(y_true, y_proba)
    elif objective == 'youden':
        return find_best_youden_threshold(y_true, y_proba)
    elif objective == 'cost':
        return find_best_cost_threshold(y_true, y_proba)
    else:
        raise ValueError(f"Unsupported threshold objective: {objective}")

# BUILD THRESHOLD ROW
def build_threshold_row(
    model_name,
    split_name,
    threshold_objective,
    y_true,
    y_proba,
    threshold_opt,
    threshold_score
):
    """
    One row = one model + one split + one threshold objective
    All metrics in this row are calculated at the `threshold_opt`.
    """
    row = {
        'model': model_name,
        'split': split_name,
        'threshold_objective': threshold_objective,
        'threshold_opt': round(threshold_opt, 6),
        'threshold_objective_value': threshold_score,
    }

    row.update(
        threshold_classification_metrics(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold_opt,
        )
    )
    return row



def evaluate_thresholds_pipeline(
    model_version: str,
    y_valid: pd.Series,
    p_valid: np.ndarray,
    y_test: pd.Series,
    p_test: np.ndarray,
    objectives: list = ['f1', 'youden', 'cost']
) -> pd.DataFrame:
    """
    Run the threshold search pipeline on the validation data 
    and apply the threshold to the test data.
    
    Args:
        model_version (str): Final model version name (example: 'lgbm_raw').
        y_valid (pd.Series): Actual target subset validation.
        p_valid (np.ndarray): Subset validation prediction probability.
        y_test (pd.Series): Actual target subset testing.
        p_test (np.ndarray): Subset testing prediction probability.
        objectives (list): List of optimization criteria ('f1', 'youden', 'cost').
        
    Returns:
        pd.DataFrame: Classification metrics performance comparison table.
    """
    evaluation_rows = []
    
    for obj in objectives:
        # 1. Find the optimal threshold using ONLY Validation data
        opt_thresh, opt_score = find_optimal_threshold(y_valid, p_valid, objective=obj)
        
        # 2. Build evaluation rows for Validation data
        row_valid = build_threshold_row(
            model_name=model_version,
            split_name='2_valid',
            threshold_objective=obj,
            y_true=y_valid,
            y_proba=p_valid,
            threshold_opt=opt_thresh,
            threshold_score=opt_score
        )
        evaluation_rows.append(row_valid)
        
        # 3. Apply the same threshold (Locked) to the Test data
        row_test = build_threshold_row(
            model_name=model_version,
            split_name='3_test',
            threshold_objective=obj,
            y_true=y_test,
            y_proba=p_test,
            threshold_opt=opt_thresh,
            # Use np.nan because the original optimization score comes from validation
            threshold_score=np.nan 
        )
        evaluation_rows.append(row_test)
        
    return pd.DataFrame(evaluation_rows)

