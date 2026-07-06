# Libraries
import pandas as pd
import numpy as np

# Module
import copy

# Viz libraries
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# roc curve from sci-kit learn
from sklearn.metrics import roc_curve

# Display all columns of data
pd.set_option("display.max_columns", None)

# Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')

"""
Implementasi Fungsi Expected Cost untuk Penentuan Ambang Keputusan Optimal-Biaya
pada Model Klasifikasi Risiko Kredit Biner.

Kerangka teoretis (Elkan, 2001; BCBS, 2006 untuk estimasi komponen LGD/EAD):

    E[C(tau)] = C_FN * P(y=1) * P(yhat=0 | y=1, tau)
              + C_FP * P(y=0) * P(yhat=1 | y=0, tau)

    di mana:
        P(yhat=0 | y=1, tau) = 1 - Recall_BAD(tau)      [False Negative Rate]
        P(yhat=1 | y=0, tau) = FPR(tau)                  [False Positive Rate]

    C_FN (biaya false negative, memberi pinjaman pada peminjam yang gagal bayar)
        didekomposisi sebagai:  C_FN ≈ EAD * LGD

    C_FP (biaya false positive, menolak peminjam yang sesungguhnya baik)
        didekomposisi sebagai opportunity cost margin bunga marjinal:
        C_FP ≈ EAD * spread_bunga * durasi_pinjaman

    tau* = argmin_tau E[C(tau)]

Catatan epistemik penting:
    C_FN dan C_FP di sini adalah ESTIMASI PORTOFOLIO-LEVEL, bukan kepastian
    per-individu. Modul ini menyediakan dua mode: (a) biaya skalar tetap
    (rasio C_FN/C_FP konstan di seluruh observasi), dan (b) biaya per-observasi
    bila EAD bervariasi antar peminjam (mis. plafon pinjaman berbeda) --
    mode (b) lebih realistis untuk portofolio kredit konsumen heterogen.
"""

# ===========================================================================
# 1. ESTIMASI C_FN DAN C_FP DARI KOMPONEN EAD / LGD / SPREAD
# ===========================================================================
def estimate_cost_components(
    ead,
    lgd,
    interest_spread=None,
    loan_duration_years=None,
    fixed_cfp=None,
):
    """
    Menurunkan C_FN dan C_FP dari komponen ekonomi primitif.

    Parameters
    ----------
    ead : float atau array-like
        Exposure at Default -- saldo terutang saat gagal bayar. Skalar untuk
        estimasi portofolio-level, atau array sepanjang n_observasi bila
        plafon pinjaman bervariasi per individu.
    lgd : float atau array-like
        Loss Given Default, proporsi EAD yang tidak terpulihkan (0-1).
        Estimasi historis: LGD = 1 - (nilai_sekarang_pemulihan / EAD).
    interest_spread : float, optional
        Margin bunga tahunan (mis. 0.08 untuk spread 8%), digunakan untuk
        menghitung C_FP sebagai opportunity cost. Wajib diisi bila fixed_cfp
        tidak diberikan.
    loan_duration_years : float, optional
        Estimasi durasi rata-rata pinjaman dalam tahun.
    fixed_cfp : float, optional
        Alternatif: berikan C_FP secara langsung (mis. dari estimasi internal
        unit bisnis) tanpa menurunkannya dari spread*durasi.

    Returns
    -------
    dict berisi C_FN, C_FP (skalar atau array, mengikuti tipe ead), dan rasio
    C_FN/C_FP -- rasio inilah yang sesungguhnya menentukan posisi tau*,
    bukan magnitudo absolut C_FN dan C_FP.
    """
    ead = np.asarray(ead, dtype=float)
    lgd = np.asarray(lgd, dtype=float)

    if np.any((lgd < 0) | (lgd > 1)):
        raise ValueError("LGD harus berada pada rentang [0, 1].")

    c_fn = ead * lgd

    if fixed_cfp is not None:
        c_fp = np.asarray(fixed_cfp, dtype=float)
    else:
        if interest_spread is None or loan_duration_years is None:
            raise ValueError(
                "Sertakan fixed_cfp, ATAU interest_spread + loan_duration_years."
            )
        c_fp = ead * interest_spread * loan_duration_years

    ratio = c_fn / c_fp if np.isscalar(c_fn) or c_fn.shape == () else np.divide(
        c_fn, c_fp, out=np.full_like(c_fn, np.nan), where=(c_fp != 0)
    )

    return {'C_FN': float(c_fn), 'C_FP': float(c_fp), 'ratio_CFN_CFP': float(ratio)}


# ===========================================================================
# 2. FUNGSI EXPECTED COST -- MODE SKALAR (rasio biaya konstan portofolio)
# ===========================================================================
def expected_cost_curve_scalar(y_true, y_proba, c_fn, c_fp, n_thresholds=1000):
    """
    Menghitung E[C(tau)] di seluruh kandidat ambang, dengan C_FN dan C_FP
    sebagai skalar tetap yang berlaku seragam pada seluruh portofolio.

    Ini adalah penyederhanaan paling umum dipakai dalam praktik: estimasi
    C_FN dan C_FP sebagai rata-rata portofolio (mis. EAD rata-rata x LGD
    rata-rata), bukan per-observasi.

    Returns
    -------
    dict: thresholds, expected_costs, optimal_threshold, optimal_cost,
          serta fpr/fnr di titik optimal untuk interpretasi.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    p1 = np.mean(y_true == 1)   # P(y=1), prevalensi BAD
    p0 = 1 - p1                 # P(y=0)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    expected_costs = np.empty_like(thresholds)
    fnr_arr = np.empty_like(thresholds)
    fpr_arr = np.empty_like(thresholds)

    y_bad = (y_true == 1)
    y_good = (y_true == 0)
    n_bad = y_bad.sum()
    n_good = y_good.sum()

    for i, tau in enumerate(thresholds):
        y_pred = (y_proba >= tau).astype(int)

        fn = np.sum((y_pred == 0) & y_bad)
        fp = np.sum((y_pred == 1) & y_good)

        fnr = fn / n_bad if n_bad > 0 else 0.0   # = 1 - Recall_BAD(tau)
        fpr = fp / n_good if n_good > 0 else 0.0  # = FPR(tau)

        expected_costs[i] = c_fn * p1 * fnr + c_fp * p0 * fpr
        fnr_arr[i] = fnr
        fpr_arr[i] = fpr

    optimal_idx = np.argmin(expected_costs)

    return {
        'thresholds': thresholds,
        'expected_costs': expected_costs,
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_cost': expected_costs[optimal_idx],
        'optimal_fnr': fnr_arr[optimal_idx],
        'optimal_fpr': fpr_arr[optimal_idx],
        'optimal_recall_bad': 1 - fnr_arr[optimal_idx],
        'optimal_specificity': 1 - fpr_arr[optimal_idx],
        'cost_ratio_used': c_fn / c_fp,
    }


# ===========================================================================
# 3. FUNGSI EXPECTED COST -- MODE PER-OBSERVASI (EAD heterogen)
# ===========================================================================

def expected_cost_curve_per_observation(
    y_true, y_proba, c_fn_array, c_fp_array, n_thresholds=1000
):
    """
    A more realistic version for heterogeneous consumer credit portfolios: 
    each observation carries its own `C_FN_i` and `C_FP_i` (i.e. `EAD_i` 
    differs because loan ceilings vary across borrowers), instead of 
    a single portfolio average.

        E[C(tau)] = (1/n) * sum_i [ C_FN_i * 1{y_i=1, yhat_i=0}
                                   + C_FP_i * 1{y_i=0, yhat_i=1} ]

    This is the nominal expected financial loss amount (not a 
    proportion/rate as in scalar mode), so the `C(tau)` scale here is 
    directly interpretable in currency units per average observation -- 
    more informative for reporting to business units.

    Parameters
    ----------
    c_fn_array, c_fp_array : array-like, with n_observations in length
        Individual costs per observation, e.g., the result of
        `estimate_cost_components()` with `EAD` as an array.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    c_fn_array = np.asarray(c_fn_array, dtype=float)
    c_fp_array = np.asarray(c_fp_array, dtype=float)

    n = len(y_true)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    expected_costs = np.empty_like(thresholds)

    y_bad = (y_true == 1)
    y_good = (y_true == 0)

    for i, tau in enumerate(thresholds):
        y_pred = (y_proba >= tau).astype(int)

        fn_mask = (y_pred == 0) & y_bad
        fp_mask = (y_pred == 1) & y_good

        total_cost = c_fn_array[fn_mask].sum() + c_fp_array[fp_mask].sum()
        expected_costs[i] = total_cost / n

    optimal_idx = np.argmin(expected_costs)

    return {
        'thresholds': thresholds,
        'expected_costs': expected_costs,
        'optimal_threshold': thresholds[optimal_idx],
        'optimal_cost_per_observation': expected_costs[optimal_idx],
    }


# ===========================================================================
# 4. PERBANDINGAN TAU* ANTAR KRITERIA (Cost vs Youden vs F1)
# ===========================================================================

def compare_threshold_criteria(y_true, y_proba, c_fn, c_fp):
    """
    Comparing the tau* generated by three different criteria on the same data, 
    to diagnose how far the statistical threshold (Youden/F1) that has been 
    used deviates from the economic-cost-optimal threshold.
    """
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Kriteria 1: Expected Cost
    cost_result = expected_cost_curve_scalar(y_true, y_proba, c_fn, c_fp)
    tau_cost = cost_result['optimal_threshold']

    # Kriteria 2: Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    tau_youden = roc_thresholds[np.argmax(j_scores)]

    # Kriteria 3: F1-maximal (PR curve)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    precisions_t, recalls_t = precisions[:-1], recalls[:-1]
    f1_scores = np.where(
        (precisions_t + recalls_t) > 0,
        2 * precisions_t * recalls_t / (precisions_t + recalls_t),
        0,
    )
    tau_f1 = pr_thresholds[np.argmax(f1_scores)]

    return {
        'tau_expected_cost': round(float(tau_cost), 4),
        'tau_youden_j': round(float(tau_youden), 4),
        'tau_f1_maximal': round(float(tau_f1), 4),
        'cost_ratio_CFN_CFP': round(c_fn / c_fp, 2),
        'interpretasi': (
            f"Dengan rasio C_FN/C_FP = {c_fn/c_fp:.4f}, ambang optimal-biaya "
            f"berada {'di bawah' if tau_cost < tau_youden else 'di atas'} "
            f"ambang Youden -- selisih {abs(tau_cost - tau_youden):.4f} pada "
            f"skala probabilitas."
        ),
    }


# ===========================================================================
# 5. VISUALISASI KURVA EXPECTED COST
# ===========================================================================

def plot_expected_cost_curve(cost_result, dataset_name='train_cal', ax=None):
    """Memvisualisasikan E[C(tau)] terhadap tau, dengan penanda titik optimal."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        cost_result['thresholds'], cost_result['expected_costs'],
        color='#8e44ad', linewidth=2, label='E[C(τ)]'
    )
    ax.axvline(
        cost_result['optimal_threshold'], color='black', linestyle="--",
        linewidth=1.2,
        label=f"τ* = {cost_result['optimal_threshold']:.4f}"
    )
    ax.scatter(
        [cost_result['optimal_threshold']],
        [cost_result['expected_costs'][np.argmin(cost_result['expected_costs'])]],
        color='black', zorder=5, s=40
    )

    ax.set_xlabel('Decision Threshold (τ)')
    ax.set_ylabel('Expected Cost  E[C(τ)]')
    ax.set_title(f"Kurva Expected Cost — {dataset_name}")
    ax.set_xlim(0, 1)
    ax.legend(loc='upper center', fontsize=9)
    ax.grid(alpha=0.25)
    return ax


# ===========================================================================
# 6. RUNNING EXPECTED COST FUNCTION
# ===========================================================================
if __name__ == "__main__":
    # --- Scalar mode: portfolio mean estimate ---
    # Misal: EAD rata-rata Rp 15.000.000, LGD historis 55%,
    # For example: Average EAD Rp. 15,000,000, historical LGD 55%,
    # interest spread 9%/year, average loan duration 2 years.
    components = estimate_cost_components(
        ead=15_000_000,
        lgd=0.55,
        interest_spread=0.09,
        loan_duration_years=2.0,
    )
    def print_cost_report(components):
        print("Estimasi Komponen Biaya")
        print("-----------------------")
        print(f"C_FN (kerugian default) : {components['C_FN']:.4f}")
        print(f"C_FP (opportunity cost) : {components['C_FP']:.4f}")
        print(f"Rasio C_FN/C_FP         : {components['ratio_CFN_CFP']:.4f}")

    print_cost_report(components)
    # C_FN = 15jt * 0.55 = 8.25jt
    # C_FP = 15jt * 0.09 * 2 = 2.7jt
    # rasio C_FN/C_FP ≈ 3.06

    cost_result_winner = expected_cost_curve_scalar(
        y_train_cal, proba_train_cal_winner,
        c_fn=components["C_FN"], c_fp=components["C_FP"]
    )
    print(f"\nTau optimal-biaya : {cost_result_winner['optimal_threshold']:.4f}")
    print(f"Recall BAD di tau optimal-biaya : {cost_result_winner['optimal_recall_bad']:.4f}\n")
    
    comparison = compare_threshold_criteria(
        y_train_cal, proba_train_cal_winner, components["C_FN"], components["C_FP"]
    )
    print(f"Perbandingan Ambang Threshold")
    print(f"-----------------------------")
    print(f"Tau optimal-biaya : {comparison['tau_expected_cost']:.4f}")
    print(f"Tau Youden's J    : {comparison['tau_youden_j']:.4f}")
    print(f"Tau F1 maksimal   : {comparison['tau_f1_maximal']:.4f}")
    print(f"Rasio C_FN/C_FP   : {comparison['cost_ratio_CFN_CFP']:.2f}")
    print(f"\nInterpretasi: {comparison['interpretasi']}\n")

    # Ambil threshold_opt untuk objective 'f1'
    f1_row = next(row for row in stage2_rows if row['threshold_objective'] == 'f1')
    youden_row = next(row for row in stage2_rows if row['threshold_objective'] == 'youden')
    f1_threshold_opt = f1_row['threshold_opt']
    ydn_threshold_opt = youden_row['threshold_opt']
    
    ax_cost = plot_expected_cost_curve(cost_result_winner, dataset_name="train_cal")
    # Add vertical markers for F1/Youden (which are identical)
    ax_cost.axvline(
        f1_threshold_opt, 
        color='black', linestyle='-', linewidth=1.5,
        label=f"F1 threshold = {f1_threshold_opt:.6f}"
    )
    ax_cost.axvline(
        ydn_threshold_opt,
        color='red', linestyle='--', linewidth=1.5,
        label=f"Youden threshold = {ydn_threshold_opt:.6f}"
    )
    ax_cost.legend(fontsize=8)
    plt.tight_layout()
    # plt.savefig("expected_cost_curve_valid.png", dpi=150)
    plt.show()
    pass