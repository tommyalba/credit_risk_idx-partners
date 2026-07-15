"""
calibration_selection.py

Companion module to the existing `run_probability_pipeline` (Open/Closed
calibration strategy registry). Consumes its `proba_store` output to
answer the question `run_probability_pipeline` deliberately leaves open:
WHICH calibration strategy (raw / sigmoid / isotonic) should actually be
used for a given model, decided empirically on the OOT test split rather
than assumed in advance.

Does not modify or duplicate the existing pipeline -- this module only
reads its output dict.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationEvaluation:
    """Per-(model, strategy, split) calibration quality summary."""

    table: pd.DataFrame       # long-form: model, strategy, split, brier, logloss
    winners: pd.DataFrame      # one row per base model: winning strategy on 'test'


def evaluate_calibration_strategies(
    proba_store: Dict[str, Dict[str, np.ndarray]],
    y_by_split: Dict[str, pd.Series],
    base_model_names: List[str],
    decision_split: str = "test",
) -> CalibrationEvaluation:
    """
    Score every (model, calibration strategy) combination in `proba_store`
    on Brier Score and LogLoss across all splits, and pick the winner per
    base model using ONLY the `decision_split` (the OOT test set -- never
    'train_cal', since that split was used to FIT the calibrators and
    would give an optimistically biased comparison).

    Parameters
    ----------
    proba_store : dict
        Output of `run_probability_pipeline`. Keys are either the raw
        model name (e.g. "logreg_scorecard") or
        "{model_name}_cal_{strategy}" (e.g. "logreg_scorecard_cal_isotonic").
        Each value is a dict of {split_name: np.ndarray of probabilities}.
    y_by_split : dict
        {split_name: true binary labels}, covering every split present
        in `proba_store` (typically 'train_cal', 'valid', 'test').
    base_model_names : list of str
        The raw model name(s) as passed to `fitted_models` in
        `run_probability_pipeline` (without any "_cal_*" suffix).
    decision_split : str
        Split used to decide the winning strategy. Defaults to "test"
        (the OOT partition) -- deliberately never "train_cal".

    Returns
    -------
    CalibrationEvaluation
        `.table`: full long-form Brier/LogLoss breakdown for every
        (model, strategy, split) combination -- useful for plotting
        calibration degradation across splits.
        `.winners`: one row per base model with the strategy
        ('raw' / 'sigmoid' / 'isotonic') that minimizes Brier Score on
        `decision_split`, plus that strategy's Brier/LogLoss for
        transparency.
    """
    rows = []
    for base_name in base_model_names:
        strategy_keys = {"raw": base_name}
        for key in proba_store:
            if key.startswith(f"{base_name}_cal_"):
                strategy_name = key[len(f"{base_name}_cal_"):]
                strategy_keys[strategy_name] = key

        for strategy_name, store_key in strategy_keys.items():
            for split_name, p_pred in proba_store[store_key].items():
                y_true = y_by_split[split_name]
                rows.append(
                    {
                        "model": base_name,
                        "strategy": strategy_name,
                        "split": split_name,
                        "brier": brier_score_loss(y_true, p_pred),
                        "logloss": log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6)),
                    }
                )

    table = pd.DataFrame(rows)

    decision_rows = table[table["split"] == decision_split].copy()
    winners = (
        decision_rows.sort_values("brier")
        .groupby("model", as_index=False)
        .first()[["model", "strategy", "brier", "logloss"]]
        .rename(columns={"strategy": "winning_strategy",
                          "brier": f"brier_{decision_split}",
                          "logloss": f"logloss_{decision_split}"})
    )

    return CalibrationEvaluation(table=table, winners=winners)


def brier_skill_score(y_true: pd.Series, p_pred: np.ndarray) -> dict:
    """
    Brier Skill Score: how much better the model's Brier Score is than a
    TRIVIAL baseline that predicts the population base rate for every
    row (`p_baseline = y_true.mean()` everywhere).

    Rationale: raw Brier Score / LogLoss are dominated by class
    prevalence -- a low base rate mechanically produces a low (good-
    looking) Brier Score even for a model with near-zero discriminative
    skill, since `Brier_trivial = p*(1-p)` is small whenever `p` is far
    from 0.5. Comparing splits by raw Brier Score alone can therefore
    show "improvement" that is actually just a lower base rate in that
    split, not better probabilistic accuracy. This function makes the
    comparison base-rate-invariant.

    Returns
    -------
    dict
        - 'base_rate': observed y_true.mean()
        - 'brier_model': the model's actual Brier Score
        - 'brier_trivial': Brier Score of the constant-base-rate predictor
        - 'skill_score': 1 - brier_model/brier_trivial
          (0 = no better than guessing the base rate; 1 = perfect;
          negative = worse than the trivial baseline)
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred)
    base_rate = y.mean()
    brier_model = brier_score_loss(y, p)
    brier_trivial = base_rate * (1 - base_rate)
    skill = 1 - (brier_model / brier_trivial) if brier_trivial > 0 else np.nan
    return {
        "base_rate": base_rate,
        "brier_model": brier_model,
        "brier_trivial": brier_trivial,
        "skill_score": skill,
    }


def reliability_table(y_true: pd.Series, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Decile-based reliability (calibration curve) table: predicted vs.
    observed bad rate per probability decile, for visual inspection
    alongside the scalar Brier/LogLoss summary above.

    Returns
    -------
    pd.DataFrame
        ['decile', 'count', 'mean_predicted', 'observed_rate'].
        A well-calibrated model has `mean_predicted` ~= `observed_rate`
        in every row.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_pred)})
    df["decile"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    return (
        df.groupby("decile", observed=True)
        .agg(count=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )