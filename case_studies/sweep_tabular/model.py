"""Creates a mean/logit function for each DGP in datasets.py

This mimics the behavior of an sklearn estimator, treating the output of the
known data generating process as a "model" to explain. We need this file (can't
just save the means while generating data) because many explanations need to
evaluate the "model" at new dataset configurations. This returns the ground
truth mean for the perturbed/intervened data for use in that explanation.

SCORES[name] agrees with E[Y | X] for every data generating process, but they
can differ at design locations that are not in the training data. This idea is
used by Example 3.3 (conditional statistical null without functional null).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from means import MEAN_FNS, chain_terminal_indices

SCORES = {}


def fit_explanation_tree(X, y, response_type, params, random_state):
    """Fit the CART model used by methods that require tree structure.

    Most methods in the synthetic sweep explain the declared DGP function.
    MDI and TreeSHAP instead require an actually fitted tree.
    """
    Tree = (
        DecisionTreeClassifier
        if response_type == "classification"
        else DecisionTreeRegressor
    )
    return Tree(random_state=random_state, **params).fit(X, y)


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


@register_score("parity")
def _score_parity(X_df, cfg):
    n_nonnull = cfg["n_nonnull"]
    cols = [X_df[f"x{j + 1}"] for j in range(n_nonnull)]
    return MEAN_FNS["parity"](cols, cfg)


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


@register_score("highly_correlated_dependent")
def _score_highly_correlated_dependent(X_df, cfg):
    return _score_dependent_features(X_df, cfg)


@register_score("mediated_chains")
def _score_mediated_chains(X_df, cfg):
    terminal_indices = chain_terminal_indices(
        cfg["n_nonnull"], cfg.get("chain_length", 3)
    )
    cols = [X_df[f"x{j + 1}"] for j in terminal_indices]
    return MEAN_FNS["mediated_chains"](cols, cfg)


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


@register_score("redundant_pair")
def _score_redundant_pair(X_df, cfg):
    """Average both copies of each latent, the representation of Example 3.3.

    On the data the two copies coincide and this equals E[Y|X]. Off the data
    they do not, which is why gradients and partial dependence see both copies
    while the conditional and risk methods see neither.
    """
    n_groups = cfg["n_nonnull"] // 2
    cols = [
        (X_df[f"x{2 * j + 1}"] + X_df[f"x{2 * j + 2}"]) / 2
        for j in range(n_groups)
    ]
    return MEAN_FNS["redundant_pair"](cols, cfg)


@register_score("bayes_incomplete")
def _score_bayes_incomplete(X_df, cfg):
    # Odd/even refer to anchors and (truly null) squares of the anchors.
    n_groups = cfg["n_nonnull"] // 2
    cols = [X_df[f"x{2 * j + 1}"] for j in range(n_groups)]
    return MEAN_FNS["bayes_incomplete"](cols, cfg)


@register_score("heteroscedastic")
def _score_heteroscedastic(X_df, cfg):
    # Odd is used in the mean; evens influence the noise scale.
    n_groups = cfg["n_nonnull"] // 2
    cols = [X_df[f"x{2 * j + 1}"] for j in range(n_groups)]
    return MEAN_FNS["heteroscedastic"](cols, cfg)


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
