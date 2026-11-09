import numpy as np
import pandas as pd


class AverageProbabilityEnsemble:
    """Soft-voting ensemble: mean of member predict_proba outputs.

    Members must share the same column order for predict_proba
    (all candidates here are binary 0/1 classifiers).
    """

    def __init__(self, members, weights=None):
        self.members = list(members)
        self.weights = np.asarray(weights if weights is not None else [1.0] * len(self.members), dtype=float)
        self.weights = self.weights / self.weights.sum()

    def predict_proba(self, X):
        probas = np.stack([m.predict_proba(X) for m in self.members])  # (n_members, n, 2)
        avg = np.tensordot(self.weights, probas, axes=1)
        return avg

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
