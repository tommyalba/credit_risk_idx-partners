"""Reject-inference sensitivity analysis (Scenario A bias quantification).

The model is trained on FUNDED loans only; Lending Club rejects/withdrawals
are unobserved. This script quantifies how portfolio-level PD and the
cost-optimal threshold move under explicit assumptions about the rejected
applicant pool. It does NOT fabricate reject labels.

Method
------
1. Score the temporal validation split (funded loans) with the saved Stage-A
   winner `models/cat_raw_stage_A_winner.joblib` (raw probabilities; nb4
   decided calibration did not help -> cat_raw is the final model).
2. Model the applicant pool as a mixture: funded share `f` (observed valid
distribution) + reject share `1-f` (unobserved). Reject covariates are
unobservable, so reject "model scores" are resampled from the funded score
distribution (same-X assumption) and reject TRUE PDs are shifted:
       PD_true = expit(logit(p_model) + delta)
   delta captures risk the lender's reject decision saw but the model cannot
   (soft information, unobserved covariates). delta=0 -> no bias (sanity).
   Alternative reject profiles: top-decile-score resample (covariate-shift
   proxy) and the worst bound PD_true = 1.
3. For each (f, reject profile): applicant-pool realized PD, approved-book
   realized PD at the operational threshold, and the oracle cost-optimal
   threshold on the mixture (scalar C_FN/C_FP as in nb4 Stage C).

Never loads data/interim/X_test_raw.pkl or y_test.pkl (OOT untouched).

Usage: python experiments/reject_inference.py
Writes: experiments/reports/reject_inference_results.csv and prints a summary table.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.binning_woe import add_derived_features          # noqa: E402
from src.thresholding import find_best_cost_threshold     # noqa: E402

RANDOM_STATE = 42
MODEL_PATH = ROOT / "models" / "cat_raw_stage_A_winner.joblib"

# nb4 Stage C 'cost' objective, moderat scenario (src/thresholding defaults)
C_FN = 8_250_000.0    # EAD * LGD = 15jt * 0.55
C_FP = 2_700_000.0    # EAD * spread * durasi = 15jt * 0.09 * 2

# Funded share of all applications. Lending Club's public data covers funded
# loans only; published approval/funding rates for P2P era are roughly
# 10-30% of completed applications. Treated as a sensitivity parameter.
FUNDED_SHARES = [0.10, 0.20, 0.30]

# Reject profiles: (name, resample_mask_or_None, logit_shift_delta)
# mask=None -> resample all valid rows; mask='top_decile' -> only rows whose
# model score is in the funded top decile (covariate-shift proxy).
REJECT_PROFILES = [
    ("same_X_d0.5",  None,        0.5),
    ("same_X_d1.0",  None,        1.0),
    ("same_X_d1.5",  None,        1.5),
    ("same_X_d0",    None,        0.0),   # sanity: identical to funded
    ("top_decile",   "top_decile", 0.0),
    ("worst_bound",  None,        np.inf),  # PD_true = 1 for all rejects
]

N_TAU = 1001


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def reject_true_pd(p_model, delta):
    """True PD of a rejected applicant given model score p_model."""
    if np.isinf(delta):
        return np.ones_like(p_model)
    return expit(logit(p_model) + delta)


def mixture_cost_curve(pd_true_pool, score_pool, w_pool, taus):
    """Per-capita expected cost E[C(tau)] on a weighted applicant pool.

    Approve iff score < tau. Expected cost per applicant:
        approved:  PD_true * C_FN   (false negative: funded a bad)
        rejected:  (1 - PD_true) * C_FP  (false positive: lost good)
    """
    costs = np.empty_like(taus)
    for i, tau in enumerate(taus):
        approved = score_pool < tau
        costs[i] = np.sum(
            w_pool * np.where(
                approved,
                pd_true_pool * C_FN,
                (1.0 - pd_true_pool) * C_FP,
            )
        )
    return costs


def main():
    X_valid = joblib.load(ROOT / "data" / "interim" / "X_valid_raw.pkl")
    y_valid = joblib.load(ROOT / "data" / "interim" / "y_valid.pkl").to_numpy()

    model = joblib.load(MODEL_PATH)
    feats = list(model.feature_names_)

    Xv = add_derived_features(X_valid)
    Xv = Xv.reindex(columns=feats)
    p = model.predict_proba(Xv)[:, 1]

    # --- sanity: replicate nb4 Stage A / C numbers on funded valid ---
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_valid, p)
    tau_funded, _ = find_best_cost_threshold(y_valid, p)
    print(f"[sanity] valid AUC={auc:.4f} (expect ~0.6468), "
          f"mean PD={p.mean():.4f} vs bad rate={y_valid.mean():.4f}, "
          f"cost-optimal tau={tau_funded:.4f} (nb4 reported 0.2883)")

    taus = np.linspace(0.0, 1.0, N_TAU)
    rows = []

    # Funded subpopulation: empirical labels as realized PD (0/1).
    # Weight per funded obs within pool = f / N_f.
    N_f = len(p)

    for f in FUNDED_SHARES:
        w_funded = np.full(N_f, f / N_f)
        # funded-only realized PD and approved-book PD at operational tau
        for name, mask, delta in REJECT_PROFILES:
            if mask == "top_decile":
                q90 = np.quantile(p, 0.9)
                idx = np.flatnonzero(p >= q90)
            else:
                idx = np.arange(N_f)
            # reject obs = resample of rows idx; each carries equal share
            # (1-f)/len(idx) of the applicant pool.
            w_rej = np.full(len(idx), (1.0 - f) / len(idx))

            pd_true = np.concatenate([y_valid.astype(float),
                                      reject_true_pd(p[idx], delta)])
            score = np.concatenate([p, p[idx]])
            w = np.concatenate([w_funded, w_rej])

            pool_pd = np.sum(w * pd_true)
            approved = score < tau_funded
            appr_mass = np.sum(w[approved])
            appr_pd = (np.sum(w[approved] * pd_true[approved]) / appr_mass
                       if appr_mass > 0 else np.nan)

            costs = mixture_cost_curve(pd_true, score, w, taus)
            tau_star = taus[np.argmin(costs)]
            # funded-only oracle tau for comparison (same weighted machinery,
            # rejects removed -> weights renormalized to funded only)
            costs_f = mixture_cost_curve(
                y_valid.astype(float), p,
                np.full(N_f, 1.0 / N_f), taus)
            tau_star_f = taus[np.argmin(costs_f)]

            rows.append({
                "funded_share": f,
                "reject_profile": name,
                "logit_shift": delta,
                "applicant_pool_PD": pool_pd,
                "funded_PD": y_valid.mean(),
                "approved_mass": appr_mass,
                "approved_book_PD": appr_pd,
                "tau_operational": tau_funded,
                "tau_oracle_mixture": tau_star,
                "tau_oracle_funded_only": tau_star_f,
            })

    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "reports" / "reject_inference_results.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
