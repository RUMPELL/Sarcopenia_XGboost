"""Per-fold preprocessing: scaler and feature selector fitted on training rows only.

`FoldPreprocessor` is fitted inside each LOOCV fold on the training rows
alone, then applied unchanged to the held-out row. It can be saved next to
the fold's model so that inference can start from a raw feature matrix.
"""

import json
import os

import numpy as np
from sklearn.feature_selection import SelectKBest

from scripts.feature_selection import SCORE_FUNCS
from scripts.preprocessing import make_scaler


class FoldPreprocessor:
    """Min-max scaling followed by top-k univariate feature selection.

    Scaling runs before selection; chi-square selection in particular needs
    non-negative inputs, which scaling provides.
    """

    def __init__(self, method="anova", k=35):
        if method not in SCORE_FUNCS:
            raise ValueError(f"Unknown feature-selection method: {method!r}")
        self.method = method
        self.k = int(k)
        self.scaler = None
        self.selector = None

    # ---- fitting / applying ------------------------------------------------
    def fit(self, X_train, y_train):
        """Fit scaler and selector on training rows only."""
        X_train = np.asarray(X_train, dtype=float)
        self.scaler = make_scaler().fit(X_train)
        X_scaled = self.scaler.transform(X_train)
        k = min(self.k, X_scaled.shape[1])
        self.selector = SelectKBest(score_func=SCORE_FUNCS[self.method], k=k).fit(X_scaled, y_train)
        return self

    def transform(self, X):
        if self.scaler is None or self.selector is None:
            raise RuntimeError("FoldPreprocessor.transform called before fit")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.selector.transform(self.scaler.transform(X))

    def fit_transform(self, X_train, y_train):
        return self.fit(X_train, y_train).transform(X_train)

    @property
    def selected_indices(self):
        return self.selector.get_support(indices=True)

    # ---- persistence -------------------------------------------------------
    def to_dict(self):
        return {
            "method": self.method,
            "k": self.k,
            "n_features_in": int(self.scaler.n_features_in_),
            "scaler_data_min": self.scaler.data_min_.tolist(),
            "scaler_data_range": self.scaler.data_range_.tolist(),
            "selected_indices": self.selected_indices.tolist(),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(method=d["method"], k=d["k"])
        n = d["n_features_in"]
        # Rebuild the fitted scaler from its stored statistics.
        scaler = make_scaler()
        scaler.n_features_in_ = n
        scaler.data_min_ = np.asarray(d["scaler_data_min"], dtype=float)
        scaler.data_range_ = np.asarray(d["scaler_data_range"], dtype=float)
        scaler.data_max_ = scaler.data_min_ + scaler.data_range_
        rng = scaler.data_range_.copy()
        rng[rng == 0] = 1.0
        scaler.scale_ = 1.0 / rng
        scaler.min_ = -scaler.data_min_ * scaler.scale_
        scaler.n_samples_seen_ = 0
        obj.scaler = scaler
        # Rebuild the selector as a fixed mask; the score function is not needed to transform.
        selector = SelectKBest(score_func=SCORE_FUNCS[obj.method], k=len(d["selected_indices"]))
        scores = np.zeros(n, dtype=float)
        scores[d["selected_indices"]] = 1.0
        selector.scores_ = scores
        selector.n_features_in_ = n
        obj.selector = selector
        return obj
