"""Creates a logit(X) function for each DGP in datasets.py

This mimics the behavior of an sklearn classifier, treating the output of the
known data generating process as a "model" to explain.

We can lookup SCORES[name](X_df, cfg) to apply the DGP's logit to columns X_df;
cfg is the configuration information giving the dataset information that
datasets.py's DATASETS functions take as input (n_nonnull, gamma/beta).
FunctionModel wraps a score into .predict_proba/.predict so it can be used
inside of `model` from case_studies/src/importance.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

SCORES = {}


def register_score(name):
    def decorator(fn):
        SCORES[name] = fn
        return fn
    return decorator


@register_score("linear_additive")
def _score_linear_additive(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    beta = cfg.get("beta", 4.0)
    return beta * sum(X_df[f"x{j + 1}"] for j in range(n_nonnull))


@register_score("xor")
def _score_xor(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    gamma = cfg.get("gamma", 3.0)
    product = np.prod([X_df[f"x{j + 1}"] for j in range(n_nonnull)], axis=0)
    return -gamma * product


@register_score("product_interaction")
def _score_product_interaction(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    gamma = cfg.get("gamma", 3.0)
    n_pairs = n_nonnull // 2
    return gamma * sum(
        X_df[f"x{2 * k + 1}"] * X_df[f"x{2 * k + 2}"] for k in range(n_pairs)
    )


@register_score("dependent_features")
def _score_dependent_features(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    gamma = cfg.get("gamma", 3.0)
    n_groups = n_nonnull // 2
    # anchors x_{2j-1} = z_j exactly (only proxies x_{2j} carry noise).
    return gamma * sum(X_df[f"x{2 * j + 1}"] for j in range(n_groups))


@register_score("confounding")
def _score_confounding(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    gamma = cfg.get("gamma", 3.0)
    return gamma * sum(X_df[f"x{j + 1}"] for j in range(n_nonnull))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class FunctionModel(ClassifierMixin, BaseEstimator):
    """score_fn(X_df, cfg) -> logit, wrapped as .predict_proba(X) = sigmoid(logit)."""

    def __init__(self, score_fn, cfg):
        self.score_fn = score_fn
        self.cfg = cfg

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self._feature_names = list(X.columns) if hasattr(X, "columns") else None
        return self

    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self._feature_names)

    def predict_proba(self, X):
        logit = self.score_fn(self._as_df(X), self.cfg)
        p1 = _sigmoid(np.asarray(logit))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
