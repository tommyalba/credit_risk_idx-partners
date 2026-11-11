"""Tests for src/reject_inference.py augmentation machinery."""
import numpy as np
import pandas as pd
import pytest

from src.reject_inference import (
    build_augmented_dataset,
    load_rejects,
    pseudo_label_frame,
)


def _raw_frame(n, seed=0):
    """Minimal valid Scenario-A raw payload."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'annual_inc': rng.uniform(20000, 150000, n),
        'collections_12_mths_ex_med': np.zeros(n, dtype=int),
        'delinq_2yrs': np.zeros(n, dtype=int),
        'dti': rng.uniform(5, 35, n),
        'earliest_cr_line': pd.to_datetime(['2010-01-01'] * n),
        'emp_length': ['10+ years'] * n,
        'home_ownership': ['MORTGAGE'] * n,
        'inq_last_6mths': np.zeros(n, dtype=int),
        'issue_d': pd.to_datetime(['2012-06-01'] * n),
        'loan_amnt': rng.uniform(5000, 35000, n),
        'open_acc': rng.integers(5, 20, n),
        'pub_rec': np.zeros(n, dtype=int),
        'purpose': ['credit_card'] * n,
        'revol_bal': rng.uniform(0, 20000, n),
        'revol_util': rng.uniform(0, 90, n),
        'term': [' 36 months'] * n,
        'total_acc': rng.integers(10, 40, n),
        'verification_status': ['Verified'] * n,
    })


def test_pseudo_label_frame_weights_sum_to_one_per_reject():
    X = _raw_frame(4)
    p = np.array([0.2, 0.5, 0.8, 0.1])
    X2, y, w = pseudo_label_frame(X, p, reject_weight=0.5)

    assert len(X2) == 8 and len(y) == 8 and len(w) == 8
    # each reject contributes total weight reject_weight * ( (1-p) + p )
    assert w.sum() == pytest.approx(4 * 0.5)
    assert y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # per-reject expected label equals p
    assert np.allclose(w[4:] / 0.5, p)


def test_build_augmented_dataset_pseudo_labels():
    Xf_raw, yr = _raw_frame(10, seed=1), pd.Series([0] * 8 + [1] * 2)
    Xr_raw = _raw_frame(3, seed=2)
    p_rej = np.array([0.3, 0.6, 0.9])

    X_aug, y_aug, w = build_augmented_dataset(
        Xf_raw, yr, Xr_raw, reject_pd=p_rej, reject_weight=1.0)

    assert len(X_aug) == 10 + 2 * 3
    assert len(y_aug) == len(w) == len(X_aug)
    assert (w[:10] == 1.0).all()
    assert w[10:].sum() == pytest.approx(3.0)
    # funded labels preserved at the front
    assert y_aug.iloc[:10].tolist() == yr.tolist()


def test_build_augmented_dataset_real_labels():
    Xf_raw = _raw_frame(5, seed=3)
    yr = pd.Series([0, 1, 0, 1, 0])
    Xr_raw = _raw_frame(4, seed=4)
    Xr_raw['bureau_target'] = [1, 0, 1, 0]

    X_aug, y_aug, w = build_augmented_dataset(
        Xf_raw, yr, Xr_raw, label_col='bureau_target',
        reject_weight=0.5)

    assert len(X_aug) == 9
    assert y_aug.iloc[5:].tolist() == [1, 0, 1, 0]
    assert (w[5:] == 0.5).all()


def test_build_augmented_dataset_requires_exactly_one_label_source():
    Xf_raw = _raw_frame(3, seed=5)
    yr = pd.Series([0, 1, 0])
    Xr_raw = _raw_frame(2, seed=6)

    with pytest.raises(ValueError):
        build_augmented_dataset(Xf_raw, yr, Xr_raw)
    with pytest.raises(ValueError):
        build_augmented_dataset(Xf_raw, yr, Xr_raw,
                                label_col='t', reject_pd=np.array([0.5, 0.5]))


def test_load_rejects_rejects_forbidden_columns(tmp_path):
    df = _raw_frame(3)
    df['recoveries'] = 0.0
    p = tmp_path / 'rejects.csv'
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match='post-origination'):
        load_rejects(p)


def test_load_rejects_rejects_missing_columns(tmp_path):
    df = _raw_frame(3).drop(columns=['annual_inc'])
    p = tmp_path / 'rejects.csv'
    df.to_csv(p, index=False)

    with pytest.raises(ValueError, match='missing application columns'):
        load_rejects(p)
