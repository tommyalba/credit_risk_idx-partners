"""Probability & threshold classification metrics shared across workflows."""
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix, balanced_accuracy_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
)

# ==========================================
# 1. CORE METRIC CALCULATOR (Pure Function)
# ==========================================

def calculate_probability_metrics(y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Computes all probability-based metrics and KS statistics without binary thresholds.

    Args:
        y_true (pd.Series): Original actual target label (0 or 1).
        y_proba (np.ndarray): 1D array of positive class probabilities of model prediction results.

    Returns:
        Dict[str, float]: The dictionary contains the metric names and their score values.
    """
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    ll = float(log_loss(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    gini = 2.0 * roc_auc - 1.0
    
    # Extract KS statistics using internal helper
    from src.ks_statistic_evaluation import compute_ks_statistic
    ks_result = compute_ks_statistic(y_true, y_proba)
    ks_stat = float(ks_result['ks_statistic_roc'])
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'log_loss': ll,
        'brier_score': brier,
        'gini': gini,
        'ks_statistic': ks_stat,
        # KS statistic (magnitude) — a discrimination metric 
        # independent of AUC/Gini; different interpretations 
        # (CDF separation at a single optimal point vs. 
        # integral area under the ROC curve)

        # NOTE: ks_threshold is NOT included here.
        # tau*_KS = tau*_Youden mathematically (exact identity).
        # Threshold is fully managed in Stage 2 via find_optimal_threshold().
    }

# ============================================================
# THRESHOLD CLASSIFICATION METRICS
# ============================================================
def threshold_classification_metrics(y_true, y_proba, threshold):
    """
    Business-ready threshold metrics for binary classification.
    Negative class = GOOD (0)
    Positive class = BAD (1)
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    predicted_bad_rate = y_pred.mean()

    return {
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision_Bad': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Recall_Bad': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'F1_Bad': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'Specificity': specificity,
        'NPV': npv,
        'FPR': fpr,
        'FNR': fnr,
        'Predicted_Bad_Rate': predicted_bad_rate,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
    }


# ============================================================
# CLASSIFICATION REPORT & CONFUSION MATRIX
# ============================================================
def print_report_and_cm(y_true, y_pred, title):
    print(f"\n{title}")
    print(classification_report(
        y_true, y_pred, 
        target_names=['GOOD (0)', 'BAD (1)']))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual GOOD (0)', 'Actual BAD (1)'],
        columns=['Pred GOOD (0)', 'Pred BAD (1)']
    )
    return cm_df
