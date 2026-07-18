# Libraries
import pandas as pd
import numpy as np
import shap
from scipy.stats import spearmanr

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
Implementasi Interpretabilitas SHAP (SHapley Additive exPlanations)
untuk Model XGBClassifier Risiko Kredit Biner.

Referensi:
    Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
        Interpreting Model Predictions. NeurIPS.
    Lundberg, S. M. et al. (2020). From Local Explanations to Global
        Understanding with Explainable AI for Trees. Nature Machine
        Intelligence.

Cakupan modul:
    1. Komputasi nilai Shapley via TreeSHAP (eksak, O(T*L*D^2))
    2. Diagnostik global     -- summary plot (kepentingan & arah fitur)
    3. Diagnostik relasional -- dependence plot (monotonisitas vs domain)
    4. Eksplanasi lokal      -- waterfall/force plot per-individu, 
                                untuk keperluan "adverse action notice"
    5. Validasi aksiomatik   -- pengecekan properti efficiency secara numerik
    6. Audit konsistensi     -- perbandingan SHAP importance vs gain-based
                                feature_importance bawaan XGBoost

Catatan penting mengenai skala output:
    Untuk objective="binary:logistic", TreeSHAP secara default mengembalikan
    kontribusi pada skala LOG-ODDS (sebelum sigmoid), BUKAN skala probabilitas
    langsung -- konsekuensi dari additivity aksiomatik (Bagian "efficiency")
    yang hanya berlaku linear pada skala margin model, bukan pada skala
    probabilitas yang non-linear. Interpretasi pada skala probabilitas
    memerlukan transformasi non-linear tambahan (lihat shap.Explanation
    dengan link="logit" pada beberapa plot, atau interpretasi manual).
"""

# ===========================================================================
# 1. KOMPUTASI NILAI SHAPLEY VIA TreeSHAP
# ===========================================================================

def compute_shap_values(model, X, feature_names=None):
    """
    Menghitung nilai Shapley eksak via algoritma TreeSHAP untuk model
    berbasis pohon (XGBoost, LightGBM, CatBoost, RandomForest, dst.).

    Parameters
    ----------
    model : XGBClassifier (atau model tree-based lain yang didukung shap.TreeExplainer)
        Model yang sudah di-fit.
    X : pd.DataFrame atau array-like
        Data pada mana SHAP dihitung -- gunakan subset representatif (mis.
        X_valid_clean atau sampel dari X_test_clean) bila n observasi besar,
        karena meski TreeSHAP polinomial, biaya tetap meningkat dengan n.
    feature_names : list, optional
        Diperlukan bila X bukan DataFrame (mis. array numpy).

    Returns
    -------
    shap.Explanation object -- struktur terpadu shap modern (>=0.40) yang
    membawa .values (kontribusi per fitur per observasi), .base_values
    (E[f(X)], nilai basis/ekspektasi), dan .data (nilai fitur asli).
    """
    if not isinstance(X, pd.DataFrame) and feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    return explanation

def validate_efficiency_axiom(explanation, model, X, atol=1e-3):
    """
    Validasi numerik aksioma 'efficiency': jumlah seluruh phi_i ditambah
    base_value harus sama dengan output mentah model (skala log-odds/margin)
    untuk setiap observasi.

        f(x) = base_value + sum_i phi_i

    Ini bukan sekadar uji unit -- ini adalah verifikasi bahwa dekomposisi
    SHAP yang dihasilkan benar-benar 'additive' sebagaimana dijamin secara
    aksiomatik oleh teori nilai Shapley, bukan aproksimasi yang menyimpang.
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
            raise TypeError(f"Model tipe {model_classname} belum didukung untuk validasi margin.")
    else:
        raise TypeError(f"Objek yang dimasukkan bukan merupakan model estimator yang valid.")
        
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)

    max_diff = np.max(np.abs(raw_margin - reconstructed))
    is_valid = max_diff < atol

    return {
        'max_absolute_difference': float(max_diff),
        'axiom_satisfied': bool(is_valid),
        'interpretation': (
            "Aksioma efficiency terverifikasi -- dekomposisi SHAP additive "
            "secara eksak terhadap output margin model."
            if is_valid else
            "PERINGATAN: selisih melebihi toleransi. Periksa kemungkinan "
            "non-determinisme prediksi (mis. GPU vs CPU) atau versi shap "
            "yang tidak kompatibel dengan versi xgboost."
        ),
    }

# ===========================================================================
# 2. DIAGNOSTIK GLOBAL -- SUMMARY PLOT
# ===========================================================================

def plot_global_importance(explanation, max_display=None, plot_type="dot"):
    """
    Summary plot: peringkat fitur berdasarkan rata-rata |SHAP value|,
    dengan sebaran titik berwarna (merah=nilai fitur tinggi, biru=rendah)
    yang menunjukkan ARAH pengaruh -- berbeda dari feature_importance gain
    bawaan XGBoost yang hanya memberi peringkat tanpa arah.

    Parameters
    ----------
    plot_type : str
        "dot"  -- sebaran per-observasi (paling informatif, default)
        "bar"  -- rata-rata |SHAP| saja (lebih ringkas untuk laporan eksekutif)
    """
    fig = plt.figure(figsize=(9, max(5, max_display * 0.35)))
    shap.summary_plot(
        explanation, max_display=max_display, plot_type=plot_type, show=False
    )
    plt.tight_layout()
    return fig

def get_global_importance_table(explanation, feature_names):
    """
    Tabel kepentingan fitur berbasis rata-rata |SHAP value| -- versi
    numerik dari summary plot, untuk pelaporan tabular dan untuk
    perbandingan eksplisit dengan gain-based importance bawaan XGBoost.
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
    Menyandingkan peringkat SHAP importance dengan gain-based importance
    bawaan XGBoost (model.feature_importances_), beserta korelasi Spearman
    antar peringkat -- mengonfirmasi atau mendiagnosis ketidaksesuaian
    sebagaimana disinggung pada analisis sebelumnya: gain-based importance
    tidak memiliki jaminan aksiomatik dan rentan bias terhadap fitur
    berkardinalitas tinggi, sehingga divergensi peringkat yang signifikan
    patut diinvestigasi lebih lanjut, bukan diabaikan.
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
            f"Korelasi peringkat Spearman = {rho:.3f}. "
            + (
                "Konkordansi tinggi antar dua metode -- temuan saling menguatkan."
                if rho > 0.7 else
                "\nDivergensi signifikan -- investigasi fitur dengan selisih "
                "peringkat besar, kemungkinan bias kardinalitas pada "
                "gain-based importance."
            )
        ),
    }

# ===========================================================================
# 3. DIAGNOSTIK RELASIONAL -- DEPENDENCE PLOT
# ===========================================================================
def plot_dependence(explanation, feature_name, interaction_feature="auto"):
    """
    Dependence plot: memvisualisasikan SHAP value suatu fitur (sumbu-y,
    kontribusi log-odds) terhadap nilai fitur tersebut (sumbu-x), dengan
    warna menunjukkan fitur interaksi otomatis terkuat.

    Tujuan diagnostik utama: memverifikasi apakah relasi fitur-risiko
    bersifat MONOTON-SESUAI-INTUISI-DOMAIN (mis. rasio utang-pendapatan
    lebih tinggi -> kontribusi risiko konsisten naik) ataukah model
    mempelajari pola non-monoton mencurigakan yang memerlukan investigasi
    lebih lanjut (potensi proxy bias, leakage, atau artefak data).
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
    Pengecekan kuantitatif monotonisitas: menghitung korelasi Spearman
    antara nilai fitur dan SHAP value-nya. Korelasi Spearman dipilih
    (bukan Pearson) karena menguji monotonisitas rank-based murni, tanpa
    mengasumsikan linearitas relasi -- selaras dengan sifat non-parametrik
    model tree-based itu sendiri.

    Parameters
    ----------
    expected_direction : str
        "increasing" -- fitur tinggi seharusnya menaikkan risiko (SHAP positif)
        "decreasing" -- fitur tinggi seharusnya menurunkan risiko (SHAP negatif)
    """
    feature_idx = list(explanation.feature_names).index(feature_name)
    raw_feature_values = explanation.data[:, feature_idx]
    shap_values_clean = np.asarray(explanation.values[:, feature_idx], dtype=float).flatten()

    # Bersihkan feature_values: antisipasi jika data tersimpan sebagai object/string
    feature_ser = pd.Series(raw_feature_values)

    # Gunakan pengecekan yang lebih luas: jika bukan numerik
    if not pd.api.types.is_numeric_dtype(feature_ser):
        try:
            # Coba konversi langsung ke float jika aslinya angka tapi bertipe string/object
            feature_values_clean = feature_ser.astype(float).to_numpy()
        except ValueError:
            # Jika gagal (misal berisi teks kategori seperti '20p16_to_21p91'),
            # Ubah menjadi urutan ordinal angka. 
            # sort=True SANGAT PENTING untuk binning agar urutannya tidak berantakan.
            if isinstance(feature_ser.dtype, pd.CategoricalDtype):
                # Jika sudah berupa category ordinal dari Pandas, manfaatkan cat.codes
                feature_values_clean = feature_ser.cat.codes.to_numpy()
            else:
                # Fallback untuk string biasa
                feature_values_clean = pd.factorize(feature_ser, sort=True)[0]
    else:
        feature_values_clean = np.asarray(raw_feature_values, dtype=float).flatten()

    # Jalankan spearmanr dengan array yang sudah dipastikan bertipe numerik murni 1D
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
            f"Relasi monoton {direction_observed} terkonfirmasi "
            f"(rho={rho:.3f}), {'sesuai' if consistent else 'BERTENTANGAN dengan'} "
            f"ekspektasi domain. "
            + ("" if consistent else
               "Investigasi lebih lanjut diperlukan -- periksa dependence plot "
               "untuk pola non-monoton lokal atau interaksi fitur yang membalik arah.")
        ),
    }

# ===========================================================================
# 4. EKSPLANASI LOKAL -- UNTUK ADVERSE ACTION NOTICE
# ===========================================================================

def explain_individual_prediction(explanation, index, feature_names=None, top_n=10):
    """
    Dekomposisi aditif untuk SATU observasi individual -- struktur yang
    dapat diterjemahkan langsung menjadi pernyataan 'alasan keputusan'
    yang dapat diaudit secara kuantitatif (relevan untuk kewajiban
    transparansi semacam ECOA Adverse Action Notice atau ketentuan
    perlindungan konsumen jasa keuangan OJK).

    Returns
    -------
    DataFrame berisi top_n kontributor terbesar (berdasarkan |SHAP value|)
    terhadap prediksi observasi tersebut, diurutkan dari kontribusi
    terbesar, beserta arah (menaikkan/menurunkan skor risiko).
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
        df['shap_value'] > 0, "menaikkan risiko (mendorong ke BAD)",
        "menurunkan risiko (mendorong ke GOOD)"
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
    Waterfall plot untuk satu observasi -- representasi visual standar
    untuk eksplanasi lokal, menunjukkan secara berurutan bagaimana setiap
    fitur menggeser prediksi dari base_value (E[f(X)] pada populasi
    training) menuju f(x) untuk observasi spesifik ini.
    """
    fig = plt.figure(figsize=(8, max(5, max_display * 0.4)))
    shap.plots.waterfall(explanation[index], max_display=max_display, show=False)
    plt.tight_layout()
    return fig

def plot_individual_force(explanation, index):
    """
    Force plot -- representasi alternatif eksplanasi lokal dalam format
    horizontal, sering lebih mudah dibaca pemangku kepentingan non-teknis
    dibanding waterfall karena menyerupai 'tarik-menarik' dua arah.
    """
    return shap.plots.force(
        explanation.base_values[index],
        explanation.values[index],
        explanation.data[index],
        feature_names=explanation.feature_names,
    )

# ===========================================================================
# Running SHAP Interpretation...
# ===========================================================================
if __name__ == "__main__":

    explanation = compute_shap_values(shap_base_model, X_valid_clean)
    
    # 1. Axiomatic validation (MUST be run once as a sanity check)
    axiom_check = validate_efficiency_axiom(explanation, shap_base_model, X_valid_clean)
    print(f"n=== SHAP AXIOM CHECK ===")
    display(axiom_check)
    
    # 2. Global diagnostics
    fig1 = plot_global_importance(explanation, max_display=15)
    fig1.savefig("shap_summary_global.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    importance_table = get_global_importance_table(
        explanation, X_valid_clean.columns.tolist()
    )
    print(f"\n=== SHAP GLOBAL IMPORTANCE (TOP 12) ===")
    display(importance_table.head(12))
    
    comparison_gain = compare_shap_vs_gain_importance(
        explanation, shap_base_model, X_valid_clean.columns.tolist()
    )
    print(f"\n=== SHAP vs GAIN IMPORTANCE COMPARISON ===")
    print(comparison_gain['interpretation'])
    print(comparison_gain['comparison_table'][['feature','rank_shap','rank_gain']].head(15))
    
    # 3. Relational diagnostics -- example for debt-to-income ratio feature
    fig2 = plot_dependence(explanation, 'dti')  # debt_to_income_ratio
    plt.show()
    # fig2.savefig("shap_dependence_dti.png", dpi=150, bbox_inches="tight")
    mono_check = check_monotonicity(
        explanation, 'dti', expected_direction='increasing'
    )
    print()
    print(mono_check['interpretation'])
    print()
    
    # 4. Local explanation -- example for the observation with the highest risk score
    proba_valid_arr = np.asarray(proba_valid_winner)
    highest_risk_idx = int(np.argmax(proba_valid_arr))
    
    local_explanation = explain_individual_prediction(
        explanation, highest_risk_idx, top_n=12
    )
    print(f"\n=== LOCAL EXPLANATION | Highest risk observation (idx={highest_risk_idx}) ===")
    display(local_explanation["table"])
    print(f"Final probability: {local_explanation['final_probability']:.4f}")
    print(f"Decision (tau=0.2286): {'REJECT (BAD)' if local_explanation['final_probability'] >= winner_threshold else 'APPROVE (GOOD)'}")
    
    fig3 = plot_individual_waterfall(explanation, highest_risk_idx)
    plt.show()
    # fig3.savefig("shap_waterfall_individual.png", dpi=150, bbox_inches="tight")
    pass