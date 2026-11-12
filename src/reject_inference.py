"""Reject-level data ingestion and fuzzy augmentation.

The canonical training population contains funded loans only; rejected
applications are unobserved (see experiments/reports/reject_inference.md for the
sensitivity bounds this creates). This module provides the machinery to
fold a reject sample into Scenario-A training once such data exists.

Two reject label regimes are supported:

1. Real labels (`label_col` present): the reject file carries an observed
   outcome (e.g. bureau follow-up performance). Rejects enter training
   with their true labels at `reject_weight`. This is the high-value
   path: it directly corrects the funded-only selection bias.

2. No labels: fuzzy augmentation. Each reject is scored by the deployed
   Scenario-A bundle and duplicated twice: once as GOOD with weight
   (1 - p) and once as BAD with weight p, so its expected label equals
   the model PD. This is self-distillation: it cannot add outcome
   information the model did not already have, but it does correct the
   covariate distribution the model trains on and reweights the loss
   toward the applicant pool. Treat results as a covariate correction,
   not as reject inference proper.

Reject payloads must satisfy an application-time feature contract:
`feature_columns` present, `forbidden_features` absent. The caller
(e.g. pipeline.scenario_a.features) owns the contract; this module
stays scenario-agnostic.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_rejects(
    path,
    feature_columns: List[str],
    forbidden_features: List[str],
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Read a rejects file (CSV/parquet) and validate the feature contract."""
    path = Path(path)
    df = (pd.read_parquet(path) if path.suffix == '.parquet'
          else pd.read_csv(path))

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f'rejects file missing application columns: {missing}')
    present_forbidden = [c for c in forbidden_features if c in df.columns]
    if present_forbidden:
        raise ValueError(
            f'rejects file contains post-origination columns: '
            f'{present_forbidden}')
    return df


def pseudo_label_frame(X_rej: pd.DataFrame, p_rej: np.ndarray,
                       reject_weight: float = 1.0
                       ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Duplicate each reject as (y=0, w=1-p) and (y=1, w=p).

    Expected label per reject equals p_rej; total weight equals
    reject_weight * len(X_rej).
    """
    p = np.clip(np.asarray(p_rej, dtype=float), 0.0, 1.0)
    X_aug = pd.concat([X_rej, X_rej], ignore_index=True)
    y_aug = pd.Series(np.concatenate([np.zeros(len(p), dtype=int),
                                      np.ones(len(p), dtype=int)]))
    w = reject_weight * np.concatenate([1.0 - p, p])
    return X_aug, y_aug, w


def build_augmented_dataset(
    X_funded_raw: pd.DataFrame,
    y_funded: pd.Series,
    X_rejects_raw: pd.DataFrame,
    label_col: Optional[str] = None,
    reject_pd: Optional[np.ndarray] = None,
    reject_weight: float = 1.0,
    *,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Combine funded rows with rejects into one training frame.

    Parameters
    ----------
    X_funded_raw : pickled interim frame (raw columns, pre-derivation).
    y_funded : aligned target series for the funded frame.
    X_rejects_raw : raw rejects payload (SCENARIO_A_FEATURE_COLUMNS).
    label_col : column in X_rejects_raw holding real reject outcomes.
        When given, rejects enter with those labels at `reject_weight`.
    reject_pd : model PDs for the rejects (required when label_col is
        None; e.g. from `score_records` on the Scenario-A bundle).
    reject_weight : sample-weight multiplier applied to every reject.
        Values < 1 down-weight the unverified reject population relative
        to funded loans with known outcomes.
    feature_fn : feature derivation function applied to both funded and
        reject raw frames (e.g. pipeline.scenario_a.features.
        add_scenario_a_features). Required -- this module does not own
        a feature contract.

    Returns
    -------
    (X_aug, y_aug, sample_weight): derived feature frames ready for
    CatBoost; funded rows carry weight 1.0.
    """
    if (label_col is None) == (reject_pd is None):
        raise ValueError('provide exactly one of label_col or reject_pd')

    X_f = feature_fn(X_funded_raw)
    X_r = feature_fn(X_rejects_raw)

    # Align reject features to the funded frame: same columns, same
    # categorical levels (CatBoost trains on category indices).
    X_r = X_r.reindex(columns=X_f.columns)
    for col in X_f.columns:
        if isinstance(X_f[col].dtype, pd.CategoricalDtype):
            X_r[col] = pd.Categorical(
                X_r[col], categories=X_f[col].cat.categories)

    n_f = len(X_f)
    if label_col is not None:
        if label_col not in X_rejects_raw.columns:
            raise ValueError(f'label_col {label_col!r} not in rejects file')
        y_r = pd.Series(X_rejects_raw[label_col].to_numpy(), dtype=int)
        if not set(y_r.unique()) <= {0, 1}:
            raise ValueError(f'{label_col} must be binary 0/1')
        X_aug = pd.concat([X_f, X_r], ignore_index=True)
        y_aug = pd.concat([y_funded.reset_index(drop=True), y_r],
                          ignore_index=True)
        w = np.concatenate([np.ones(n_f),
                            np.full(len(X_r), reject_weight)])
        return X_aug, y_aug, w

    X_pseudo, y_pseudo, w_pseudo = pseudo_label_frame(
        X_r, reject_pd, reject_weight)
    X_aug = pd.concat([X_f, X_pseudo], ignore_index=True)
    y_aug = pd.concat([y_funded.reset_index(drop=True), y_pseudo],
                      ignore_index=True)
    w = np.concatenate([np.ones(n_f), w_pseudo])
    return X_aug, y_aug, w
