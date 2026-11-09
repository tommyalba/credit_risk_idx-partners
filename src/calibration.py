"""Probability calibration pipeline (raw / sigmoid / isotonic)."""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Callable, Optional, Union, List

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation import extract_positive_proba
from src.metrics import calculate_probability_metrics

logger = logging.getLogger(__name__)


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

