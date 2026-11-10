"""autoresearch experiment runner — evaluates ONE candidate config on the
temporal validation split. See autoresearch/program.md for the protocol.

Usage:
    python autoresearch/experiment.py --config autoresearch/configs/<name>.json

Never loads the OOT test pickles. Prints a single `RESULT {...}` JSON line
and appends the trial to autoresearch/results/trials.jsonl.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.binning_woe import add_derived_features          # noqa: E402
from src.training import FIT_STRATEGIES, build_estimator  # noqa: E402
from src.utils import deserialize_data, split_train_test  # noqa: E402

RANDOM_STATE = 42
TRAIN_TEMP_SIZE = 0.15
ES_CAL_SHARE_WITHIN_TEMP = 0.50

# Family baselines = nb4 "best consistent" hyperparameters (verified on
# '2_valid'). Configs deep-merge over these; only changed keys are needed.
BASE_PARAMS = {
    "cat": {
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "iterations": 5000, "early_stopping_rounds": 50,
        "bootstrap_type": "Bernoulli", "cat_features": "__AUTO__",
        "task_type": "CPU", "thread_count": -1, "border_count": 64,
        "learning_rate": 0.034042376046732846, "depth": 6,
        "grow_policy": "SymmetricTree", "l2_leaf_reg": 100.0,
        "subsample": 0.65, "random_strength": 1e-09,
        "min_data_in_leaf": 45, "rsm": 0.6,
        "random_seed": RANDOM_STATE, "verbose": False,
        "allow_writing_files": False,
    },
    "xgb": {
        "objective": "binary:logistic", "booster": "gbtree",
        "eval_metric": "logloss", "tree_method": "hist",
        "enable_categorical": True, "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "learning_rate": 0.01783339959316753, "max_depth": 8,
        "min_child_weight": 20.0, "gamma": 5.0, "subsample": 0.65,
        "colsample_bytree": 0.6286729841272466, "reg_alpha": 1e-08,
        "reg_lambda": 1e-08, "max_delta_step": 0, "max_bin": 507,
        "grow_policy": "depthwise", "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "lgbm": {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "n_estimators": 5000,
        "learning_rate": 0.014752303267203249,
        "early_stopping_rounds": 50, "feature_fraction": 0.6,
        "max_depth": 6, "num_leaves": 16, "min_child_samples": 72,
        "bagging_fraction": 0.65, "bagging_freq": 1,
        "reg_alpha": 0.46210107890864205, "reg_lambda": 1.087620542758583e-08,
        "min_split_gain": 5.524657351707396e-06, "max_bin": 512,
        "verbosity": -1, "n_jobs": -1, "random_state": RANDOM_STATE,
    },
}

ESTIMATOR_CLASSES = {
    "cat": "catboost.CatBoostClassifier",
    "xgb": "xgboost.XGBClassifier",
    "lgbm": "lightgbm.LGBMClassifier",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


OVER_SAMPLERS = {"smote"}
UNDER_SAMPLERS = {"rus", "nearmiss", "clustercentroids"}


def apply_sampling(X, y, spec):
    """Resample the FIT split only. spec: {"method": ..., "minority_frac": f}
    where f is the target BAD fraction of the resampled set.

    "smote" uses SMOTENC (tree features include categorical columns);
    undersamplers keep all bads and shrink the majority class.
    """
    method = spec["method"]
    frac = float(spec["minority_frac"])
    assert method in OVER_SAMPLERS | UNDER_SAMPLERS, f"unknown sampling {method!r}"

    y_arr = np.asarray(y)
    n1 = int((y_arr == 1).sum())
    n0 = len(y_arr) - n1
    n1_target = int(round(frac / (1.0 - frac) * n0))

    if method in OVER_SAMPLERS:
        if n1_target <= n1:
            return X, y  # already at/above target minority fraction
        strategy = {0: n0, 1: n1_target}
    else:
        n0_target = int(n1 * (1.0 - frac) / frac)
        if n0_target >= n0:
            return X, y  # cannot add majority samples
        strategy = {0: n0_target, 1: n1}

    cat_idx = [i for i, c in enumerate(X.columns)
               if str(X[c].dtype) == "category"]
    Xs = X.copy()
    # shortcut: imblearn rejects NaN; emp_length_ordinal has ~3% missing.
    # Fill with -1 sentinel only inside sampled variants (CatBoost treats it
    # as a regular value; effect on ~3% of fit rows is negligible).
    for c in Xs.columns:
        if str(Xs[c].dtype) != "category" and Xs[c].isna().any():
            Xs[c] = Xs[c].fillna(-1)

    if method == "smote":
        from imblearn.over_sampling import SMOTENC
        sampler = SMOTENC(sampling_strategy=strategy,
                          categorical_features=cat_idx,
                          k_neighbors=5, random_state=RANDOM_STATE)
    elif method == "rus":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(sampling_strategy=strategy,
                                     random_state=RANDOM_STATE)
    elif method == "nearmiss":
        from imblearn.under_sampling import NearMiss
        sampler = NearMiss(sampling_strategy=strategy)
    else:
        from imblearn.under_sampling import ClusterCentroids
        sampler = ClusterCentroids(sampling_strategy=strategy,
                                   random_state=RANDOM_STATE)

    X_res, y_res = sampler.fit_resample(Xs, y_arr)
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X.columns)
    for c in cat_idx:
        X_res.iloc[:, c] = X_res.iloc[:, c].astype("category")
    return X_res, np.asarray(y_res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    model = cfg["model"]
    assert model in BASE_PARAMS, f"unknown model {model!r}"

    params = deep_merge(BASE_PARAMS[model], cfg.get("params", {}))

    t0 = time.time()

    # --- replicate nb4 preprocessing exactly ---
    X_train = deserialize_data("data/interim/X_train_raw.pkl")
    y_train = deserialize_data("data/interim/y_train.pkl")
    X_valid = deserialize_data("data/interim/X_valid_raw.pkl")
    y_valid = deserialize_data("data/interim/y_valid.pkl")
    # NOTE: X_test / y_test intentionally never loaded (OOT stays untouched).

    X_fit, X_temp, y_fit, y_temp = split_train_test(
        X_train, y_train, test_size=TRAIN_TEMP_SIZE,
        stratify=y_train, seed=RANDOM_STATE)
    X_es, _X_cal, y_es, _y_cal = split_train_test(
        X_temp, y_temp, test_size=ES_CAL_SHARE_WITHIN_TEMP,
        stratify=y_temp, seed=RANDOM_STATE)

    frames = {k: add_derived_features(v) for k, v in
              {"fit": X_fit, "es": X_es, "valid": X_valid}.items()}
    cols = sorted(frames["fit"].columns)
    frames = {k: df[cols] for k, df in frames.items()}

    # feature subset (applied AFTER derived features + sort, matching nb4)
    feats = cfg.get("features", "all")
    if feats != "all":
        missing = set(feats) - set(cols)
        assert not missing, f"unknown features: {missing}"
        frames = {k: df[list(feats)] for k, df in frames.items()}

    cat_cols = frames["fit"].select_dtypes(include=["category"]).columns.tolist()
    if params.get("cat_features") == "__AUTO__":
        params["cat_features"] = cat_cols

    # optional class-imbalance sampling: fit split ONLY, es/valid untouched
    sampling = cfg.get("sampling")
    if sampling:
        frames["fit"], y_fit = apply_sampling(
            frames["fit"], y_fit, sampling)

    import importlib
    mod_name, cls_name = ESTIMATOR_CLASSES[model].split(".")
    estimator = getattr(importlib.import_module(mod_name), cls_name)(**params)

    fit_fn = FIT_STRATEGIES[model]
    estimator = fit_fn(estimator, frames["fit"], y_fit, frames["es"], y_es)

    proba = estimator.predict_proba(frames["valid"])[:, 1]
    elapsed = time.time() - t0

    # cost-optimal threshold re-tuned on THIS variant's validation probs
    # (never reuse a fixed tau across sampling variants)
    from src.thresholding import find_best_cost_threshold
    cost_tau, _ = find_best_cost_threshold(y_valid, proba)
    approved = proba < cost_tau
    appr_bad_rate = float(np.mean(y_valid[approved])) if approved.any() else None

    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": args.config.name,
        "config_sha": hashlib.sha256(args.config.read_bytes()).hexdigest()[:12],
        "model": model,
        "n_features": len(frames["valid"].columns),
        "features": feats if feats == "all" else "subset",
        "sampling": sampling,
        "auc": round(roc_auc_score(y_valid, proba), 6),
        "logloss": round(log_loss(y_valid, proba), 6),
        "brier": round(brier_score_loss(y_valid, proba), 6),
        "cost_tau": round(float(cost_tau), 4),
        "approved_bad_rate": (round(appr_bad_rate, 6)
                              if appr_bad_rate is not None else None),
        "elapsed_s": round(elapsed, 1),
        "git": git_sha(),
    }

    out_dir = ROOT / "autoresearch" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trials.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
