import os
import joblib
import pandas as pd
import numpy as np
import copy
import logging
from typing import Any, Dict, Tuple, Callable, Optional, Union, List
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, log_loss, 
    matthews_corrcoef, classification_report, confusion_matrix, balanced_accuracy_score, 
    average_precision_score, brier_score_loss, precision_recall_curve, accuracy_score
)

logger = logging.getLogger(__name__)

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
# FIND BEST COST-BASED THRESHOLD
# ============================================================
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

# ============================================================
# FIND BEST F1 THRESHOLD
# ============================================================
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

# ============================================================
# FIND BEST YOUDEN J THRESHOLD
# ============================================================
def find_best_youden_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if len(thresholds) == 0:
        raise ValueError("Error: Thresholds are 0")

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]

# ============================================================
# FIND OPTIMAL THRESHOLD
# ============================================================
def find_optimal_threshold(y_true, y_proba, objective='cost'):
    if objective == 'f1':
        return find_best_f1_threshold(y_true, y_proba)
    elif objective == 'youden':
        return find_best_youden_threshold(y_true, y_proba)
    elif objective == 'cost':
        return find_best_cost_threshold(y_true, y_proba)
    else:
        raise ValueError(f"Unsupported threshold objective: {objective}")

# ============================================================
# BUILD THRESHOLD ROW
# ============================================================
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

# ============================================================
# CALIBRATION CURVE TABLE
# ============================================================
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
        # qcut bisa gagal kalau banyak probability identik -> duplicates='drop'
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
# SMOTE MODULE
# ============================================================
from imblearn.over_sampling import BorderlineSMOTE
def build_borderline_smote(
    sampling_strategy: Union[float, str] = 'auto',
    kind: str = 'borderline-1',
    k_neighbors: int = 5,
    m_neighbors: int = 10,
    random_state: int = 42
) -> BorderlineSMOTE:
    """
    Builds a BorderlineSMOTE object without relying on global variables.

    Args:
        sampling_strategy (Union[float, str]): Sampling strategy to use.
        kind (str): The type of BorderlineSMOTE ('borderline-1' or 'borderline-2').
        k_neighbors (int): Number of nearest neighbours to used to construct synthetic samples.
        m_neighbors (int): Number of nearest neighbours to use to determine if a minority sample is in danger.
        random_state (int): Seed for reproducibility.

    Returns:
        BorderlineSMOTE: Configured SMOTE instance.
    """
    return BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        kind=kind,
        k_neighbors=k_neighbors,
        m_neighbors=m_neighbors,
        random_state=random_state,
    )

def apply_blsmote_on_training_set(
    X_train_fit: pd.DataFrame, 
    y_train_fit: pd.Series, 
    smote_estimator: BorderlineSMOTE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies BorderlineSMOTE exclusively on the training subset to prevent data leakage.

    Args:
        X_train_fit (pd.DataFrame): Training features.
        y_train_fit (pd.Series): Training target.
        smote_estimator (BorderlineSMOTE): Pre-configured SMOTE object.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled X and y arrays/dataframes.
    """
    logger.info(f"Applying BorderlineSMOTE on train_fit subset...")
    
    X_res, y_res = smote_estimator.fit_resample(X_train_fit, y_train_fit)

    logger.info(
        f"BorderlineSMOTE complete | before=%s / %s | after=%s / %s",
        X_train_fit.shape, y_train_fit.shape,
        X_res.shape, y_res.shape
    )

    dist_before = pd.Series(y_train_fit).value_counts(dropna=False).sort_index()
    dist_after = pd.Series(y_res).value_counts(dropna=False).sort_index()

    logger.info(
        f"\n=== BORDERLINESMOTE CLASS DISTRIBUTION ===\n"
        f"Before:\n{dist_before.to_string()}\n\n"
        f"After:\n{dist_after.to_string()}"
    )
    return X_res, y_res

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
    X_train_es: pd.DataFrame = None, 
    y_train_es: pd.Series = None
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
# CALIBRATION STRATEGIES (Open/Closed)
# ============================================================
# Type alias untuk strategi kalibrasi
# Menerima (p_train_cal, y_train_cal, p_target) -> mengembalikan p_calibrated
CalibrateStrategy = Callable[[np.ndarray, pd.Series, np.ndarray], np.ndarray]

def _calibrate_sigmoid(p_train_cal: np.ndarray, y_train_cal: pd.Series, p_target: np.ndarray) -> np.ndarray:
    """Platt Scaling / Sigmoid Calibration menggunakan Logistic Regression."""
    calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, random_state=42)
    # Scikit-Learn LogisticRegression mewajibkan input 2D array
    calibrator.fit(p_train_cal.reshape(-1, 1), y_train_cal)
    return calibrator.predict_proba(p_target.reshape(-1, 1))[:, 1]

def _calibrate_isotonic(p_train_cal: np.ndarray, y_train_cal: pd.Series, p_target: np.ndarray) -> np.ndarray:
    """Isotonic Regression Calibration (Non-parametric)."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    # IsotonicRegression menerima 1D array
    calibrator.fit(p_train_cal, y_train_cal)
    return calibrator.predict(p_target)

# Registry untuk metode kalibrasi yang tersedia
CALIBRATION_STRATEGIES: Dict[str, CalibrateStrategy] = {
    'sigmoid': _calibrate_sigmoid,
    'isotonic': _calibrate_isotonic
}

# ============================================================
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

# ============================================================
# CORE PIPELINE FUNCTION
# ============================================================
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
        data_splits (Dict[str, pd.DataFrame]): Dict berisi 'train_cal', 'valid', dan 'test' dataframes.
        y_train_cal (pd.Series): Aktual target label untuk subset calibration.
        cal_strategies (Dict[str, CalibrateStrategy]): Registry strategi kalibrasi.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Registry probabilitas terstruktur.
    """
    # Validasi input data split
    required_splits = {'train_cal', 'valid', 'test'}
    if not required_splits.issubset(data_splits.keys()):
        raise KeyError(f"data_splits wajib memiliki key berikut: {required_splits}")

    proba_store: Dict[str, Dict[str, np.ndarray]] = {}

    # --- TAHAP 1: Ekstraksi Probabilitas Mentah (Raw) ---
    for model_name, model_obj in fitted_models.items():
        logger.info(f"Extracting raw probabilities for model: %s", model_name)
        
        # Buat key untuk versi raw model tersebut
        raw_key = f"{model_name}"
        proba_store[raw_key] = {}
        
        for split_name, X_data in data_splits.items():
            proba_store[raw_key][split_name] = extract_positive_proba(model_obj, X_data)
            
        # --- TAHAP 2: Kalibrasi Otomatis (Sigmoid & Isotonic) ---
        p_train_cal_raw = proba_store[raw_key]['train_cal']
        
        for strategy_name, strategy_func in cal_strategies.items():
            cal_key = f"{model_name}_cal_{strategy_name}"
            proba_store[cal_key] = {}
            
            logger.info(f"Applying %s calibration for model: %s", strategy_name, model_name)
            
            for split_name, _ in data_splits.items():
                p_target_raw = proba_store[raw_key][split_name]
                
                # Eksekusi kalibrasi menggunakan fungsi strategi terkait
                proba_store[cal_key][split_name] = strategy_func(
                    p_train_cal_raw, 
                    y_train_cal, 
                    p_target_raw
                )
                
    logger.info(f"Probability extraction and calibration pipeline complete for all models.")
    return proba_store


# ============================================================
# MODEL SERIALIZATION UTILITIES
# ============================================================
def save_best_model(model_obj: Any, filepath: str) -> None:
    """
    Save ML model objects to disk using joblib.
    
    Args:
        model_obj (Any): The fit estimator object.
        filepath (str): Full storage path location (including .pkl or .joblib).
    """
    import joblib
    # Memastikan direktori tujuan tersedia
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model_obj, filepath)


# ============================================================
# STAGE A WINNER SELECTOR
# ============================================================
def select_stage_a_winner(
    leaderboard_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
    manual_winner: Optional[str] = None
) -> Tuple[str, Any, pd.Series]:
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
        Tuple[str, Any, pd.Series]: 
            - Name of the winning model.
            - Winning model object.
            - The performance metrics row of the model in the validation set.
    """
    if manual_winner is not None:
        if manual_winner not in fitted_models:
            raise ValueError(f"Manual winner '{manual_winner}' tidak ditemukan di dictionary model.")
        winner_name = manual_winner
        
        # Extract validation metric rows for manual models
        valid_metrics = leaderboard_df[
            (leaderboard_df['split'] == '2_valid') & 
            (leaderboard_df['model_name'] == winner_name)
        ].iloc[0]
        
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
        valid_metrics = sorted_valid.iloc[0]
        winner_name = valid_metrics['model_name']
        
    winner_model = fitted_models[winner_name]
    
    return winner_name, winner_model, valid_metrics


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
        cal_leaderboard (pd.DataFrame): DataFrame results of `evaluate calibration candidates`.
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


# ======================================