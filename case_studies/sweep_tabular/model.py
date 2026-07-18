"""Creates a mean/logit function for each DGP in datasets.py

This mimics the behavior of an sklearn estimator, treating the output of the
known data generating process as a "model" to explain. We need this file (can't
just save the means while generating data) because many explanations need to
evaluate the "model" at new dataset configurations. This returns the ground
truth mean for the perturbed/intervened data for use in that explanation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from means import MEAN_FNS

SCORES = {}


def register_score(name):
    def decorator(fn):
        SCORES[name] = fn
        return fn
    return decorator


@register_score("linear_additive")
def _score_linear_additive(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["linear_additive"](cols, cfg)


@register_score("xor")
def _score_xor(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["xor"](cols, cfg)


@register_score("product_interaction")
def _score_product_interaction(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["product_interaction"](cols, cfg)


@register_score("dependent_features")
def _score_dependent_features(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    n_groups = n_nonnull // 2
    # odd indices are used, evens are marginally related noise
    cols = [X_df[f"x{2 * j + 1}"] for j in range(n_groups)]
    return MEAN_FNS["dependent_features"](cols, cfg)


@register_score("quadratic")
def _score_quadratic(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["quadratic"](cols, cfg)


@register_score("confounding")
def _score_confounding(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["confounding"](cols, cfg)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class _BaseFunctionModel(BaseEstimator):
    """score_fn(X_df, cfg) -> mean/logit, shared plumbing for classifier/regressor."""

    def __init__(self, score_fn, cfg):
        self.score_fn = score_fn
        self.cfg = cfg

    def fit(self, X, y=None):
        self._feature_names = list(X.columns) if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self

    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self._feature_names)


class FunctionClassifier(ClassifierMixin, _BaseFunctionModel):
    """Defines a .predict_proba(X) method that returns sigmoid(logit)."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return super().fit(X, y)

    def predict_proba(self, X):
        logit = self.score_fn(self._as_df(X), self.cfg)
        p1 = _sigmoid(np.asarray(logit))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FunctionRegressor(RegressorMixin, _BaseFunctionModel):
    """Defines a .predict(X) method that returns the mean."""

    def predict(self, X):
        return np.asarray(self.score_fn(self._as_df(X), self.cfg))
