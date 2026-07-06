# Libraries
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve

# Viz libraries
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Display all columns of data
pd.set_option("display.max_columns", None)

# Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')

"""
Implementation of Kolmogorov-Smirnov (KS) Statistics as a Complementary Diagnostic
for the Evaluation of Binary Credit Risk Classification Models.

Theoretical context:
    KS = max_tau | F_BAD(tau) - F_GOOD(tau) |
       = max_tau | TPR(tau) - FPR(tau) |

Mathematically KS is identical to the maximum value of the Youden index (KS = max J),
but is reported as a single scalar that summarizes the overall separation capacity of the model,
accompanied by a visualization of two cumulative CDF curves of the prediction scores for each class (GOOD vs BAD).

Industry interpretation convention (Siddiqi, 2017, Credit Risk Scorecards):
    KS < 20%   : Weak discrimination
    20% - 40%  : Adequate discrimination
    > 40%      : Strong discrimination
"""

# ==========================================
# COMPUTE KS STATISTICS
# ==========================================
def compute_ks_statistic(y_true, y_proba):
    """
    Calculate the KS statistic using two mutually verifying approaches:
    (1) directly from the ROC curve (KS = max(TPR - FPR)),
    (2) via scipy.stats.ks_2samp on two score distributions (BAD vs GOOD).

    Both approaches should produce very close values ​​(ideally identical 
    up to the discretization numerical error); significant differences 
    indicate an error in one of the implementations.
    
    Parameters
    ----------
    y_true : array-like
        Actual binary label (0 = GOOD, 1 = BAD).
    y_proba : array-like
        The predicted probability of the positive class (BAD), 
        e.g. predict_proba(X)[:, 1].

    Returns
    -------
    dict contains:
        ks_statistic_roc   : KS is calculated from the ROC curve
        ks_threshold       : threshold (probability score) at which maximum KS is reached
        ks_statistic_2samp : KS is calculated via scipy.stats.ks_2samp (cross-check)
        ks_pvalue          : p-value of the two-sample KS test.
                            (H0: the distributions of BAD and GOOD scores are identical; 
                            the very small p-value here is trivial and expected -- 
                            not a measure of model power, since large n makes this test 
                            almost always significant; ignore for diagnostic purposes, 
                            use only the KS statistic itself)
        fpr, tpr, thresholds : complete array of `roc_curve` for plotting
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # --- Approach 1: via ROC curve ---
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    ks_statistic_roc = j_scores[optimal_idx]
    ks_threshold = thresholds[optimal_idx]

    # --- Approach 2: via scipy ks_2samp (independent cross-check) ---
    scores_bad = y_proba[y_true == 1]
    scores_good = y_proba[y_true == 0]
    ks_result = ks_2samp(scores_bad, scores_good)

    return {
        'ks_statistic_roc': ks_statistic_roc,
        'ks_threshold': ks_threshold,
        'ks_statistic_2samp': ks_result.statistic,
        'ks_pvalue': ks_result.pvalue,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }

# ==========================================
# INTERPRET KS
# ==========================================
def interpret_ks(ks_value):
    """Conventional interpretation of KS value according 
    to credit scoring industry conventions."""
    ks_pct = ks_value * 100
    if ks_pct < 20:
        return "weak"
    elif ks_pct < 40:
        return "adequate"
    else:
        return "strong"

# ==========================================
# KS REPORT TABLE
# ==========================================
def ks_report_table(datasets: dict):
    """
    Build compact KS tables for multi-partition, 
    consistent format with existing metrics tables 
    (Balanced Accuracy, ROC-AUC, etc.)

    Parameters
    ----------
    datasets : dict
        Format: {'Training': (y_train, proba_train),
                 'Validation': (y_valid, proba_valid),
                 'Test': (y_test, proba_test)}

    Returns
    -------
    list of dict, one line per partition.
    """
    rows = []
    for name, (y_true, y_proba) in datasets.items():
        result = compute_ks_statistic(y_true, y_proba)
        rows.append({
            'Dataset': name,
            'KS Statistic': round(result['ks_statistic_roc'], 4),
            'KS (%)': round(result['ks_statistic_roc'] * 100, 2),
            'KS Threshold': round(result['ks_threshold'], 4),
            'Interpretasi': interpret_ks(result['ks_statistic_roc']),
        })
        # Validasi silang antar dua metode perhitungan
        diff = abs(result['ks_statistic_roc'] - result['ks_statistic_2samp'])
        assert diff < 1e-6, (
            f"Selisih KS antar metode pada {name} melebihi toleransi: {diff}. "
            "Check for possible score duplication or threshold discretization issues."
        )
    return pd.DataFrame(rows)

# ==========================================
# PLOT KS CURVE
# ==========================================
def plot_ks_curve(y_true, y_proba, dataset_name="Validation", ax=None):
    """
    Memvisualisasikan dua kurva CDF kumulatif (GOOD vs BAD) terhadap skor
    probabilitas, dengan penanda eksplisit pada titik pemisahan maksimum (KS).

    Ini adalah representasi konvensional industri credit scoring, lebih
    intuitif bagi pemangku kepentingan non-teknis dibanding kurva ROC,
    karena secara visual langsung menunjukkan skor di mana kedua populasi
    paling terpisah.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    result = compute_ks_statistic(y_true, y_proba)

    # Bangun CDF kumulatif terhadap skor terurut, dipetakan ke thresholds ROC
    # tpr = F_BAD(tau) secara definisi (TPR = proporsi BAD dengan skor >= tau,
    # diukur dari kanan; untuk CDF dari kiri gunakan 1 - tpr dan 1 - fpr)
    fpr, tpr = result['fpr'], result['tpr']
    thresholds = result['thresholds']

    cdf_good = fpr          # F_GOOD(skor <= tau), dari definisi FPR komplementer
    cdf_bad = tpr           # F_BAD(skor <= tau), dari definisi TPR komplementer

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(thresholds, cdf_bad, label="CDF — BAD (1)", color="#c0392b", linewidth=2)
    ax.plot(thresholds, cdf_good, label="CDF — GOOD (0)", color="#2980b9", linewidth=2)

    ks_idx = np.argmax(tpr - fpr)
    ax.vlines(
        thresholds[ks_idx], cdf_good[ks_idx], cdf_bad[ks_idx],
        color="black", linestyle="--", linewidth=1.5,
        label=f"KS = {result['ks_statistic_roc']:.4f} @ τ={thresholds[ks_idx]:.4f}"
    )

    ax.set_xlabel("Probability Score Threshold (τ)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title(f"Kolmogorov–Smirnov Curve— {dataset_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    return ax


# =======================
# Running KS STATISTICS
# =======================
if __name__ == "__main__":
    
    datasets = {
        'Training (train_cal)' : (y_train_cal, proba_train_cal_winner),
        'Validation'           : (y_valid, proba_valid_winner),
        'Test'                 : (y_test,  proba_test_winner),
    }
    print(f"=== KS STATISTICS (WINNER MODEL: FINAL THRESHOLD ALIGNMENT) ===")
    ks_table = ks_report_table(datasets)
    display(ks_table)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, (y_t, p)) in zip(axes, datasets.items()):
        plot_ks_curve(y_t, p, dataset_name=name, ax=ax)
    plt.suptitle(f"KS Curves — {winner_model_name}", fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig("ks_curves_all_partitions.png", dpi=150)
    pass