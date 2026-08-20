"""Consistent interface to feature importance methods."""

import logging
import numpy as np
import pandas as pd
import shap
import knockpy
from rf import fit_final
from axiom_interp import presets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, Ridge
from sklearn.metrics import matthews_corrcoef as mcc_score
from sklearn.inspection import permutation_importance as pi
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, pearsonr, norm as _norm
from sklearn.inspection import partial_dependence as pd_func
import xgboost as xgb

log = logging.getLogger(__name__)

METHODS = {}

# Most of our explanations are based off of a known data generating process mean
# function. These two, however, require an actual fitted estimator because they
# depend on the structure of the model.
FITTED_MODEL_METHODS = {"mdi", "treeshap"}


def register(name):
    def decorator(fn):
        METHODS[name] = fn
        return fn
    return decorator


@register("mdi")
def mdi(model, feature_names):
    return pd.Series(model.feature_importances_, index=feature_names, name="mdi")


@register("permutation")
def permutation(model, X, y, feature_names, n_repeats, rng, response_type):
    scoring = "matthews_corrcoef" if response_type == "classification" else "r2"
    result = pi(
        model, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=int(rng.integers(1, 2**31)),
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=feature_names, name="permutation")


@register("treeshap")
def treeshap(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    vals = shap_values[:, :, 1]
    return pd.Series(np.mean(np.abs(vals), axis=0), index=feature_names, name="treeshap")


@register("loco")
def loco(X_df, y, feature_names, model, cfg, seed, rng):
    """Leave one covariate out explanations

    This implements the risk drop from Definition 2.7, using a configurable
    function class (e.g. random forest or linear model) evaluated on a held-out
    test set.
    """
    X = X_df.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    split = _risk_split(len(y), cfg["risk"].get("test_size", 0.3), rng)
    random_state = int(rng.integers(1, 2**31))

    everything = tuple(range(X.shape[1]))
    full_risk = _subset_risk(everything, X, y, cfg["risk"], split, random_state)
    scores = [
        _subset_risk(
            tuple(k for k in everything if k != j), X, y, cfg["risk"], split,
            random_state,
        ) - full_risk
        for j in everything
    ]
    return pd.Series(scores, index=feature_names, name="loco")


@register("lasso")
def lasso(X, y, feature_names, response_type, rng):
    """|beta[j]| from a cross-validated lasso.

    It's either an L1-regularized regression or logistic regression, depending
    on the response type.
    """
    random_state = int(rng.integers(1, 2**31))
    if response_type == "classification":
        estimator = LogisticRegressionCV(
            penalty="l1", solver="liblinear", max_iter=1000,
            random_state=random_state,
        )
    else:
        estimator = LassoCV(random_state=random_state)

    fitted = make_pipeline(StandardScaler(), estimator).fit(X, y)
    coef = np.ravel(fitted[-1].coef_)
    return pd.Series(np.abs(coef), index=feature_names, name="lasso")


@register("correlation")
def correlation(X, y, feature_names, response_type):
    corr_fn = pointbiserialr if response_type == "classification" else pearsonr
    scores = np.array([abs(corr_fn(y, X[:, j])[0]) for j in range(X.shape[1])])
    return pd.Series(scores, index=feature_names, name="correlation")


@register("knockoffs")
def knockoff_scores(X_df, y, feature_names, kcfg, rng):
    max_feat = kcfg["max_features"]

    X_work = X_df
    kept_features = feature_names
    if X_df.shape[1] > max_feat:
        variances = X_df.var()
        kept_features = list(variances.nlargest(max_feat).index)
        X_work = X_df[kept_features]

    kfilter = knockpy.KnockoffFilter(fstat="lasso", ksampler="gaussian")
    kfilter.forward(
        X=X_work.values, y=np.asarray(y, dtype=float), fdr=kcfg["fdr"]
    )
    w_stats = kfilter.W

    result = pd.Series(0.0, index=feature_names, name="knockoffs")
    for j, feat in enumerate(kept_features):
        result[feat] = w_stats[j]
    return result


def _fit_predictor(risk_cfg, random_state):
    """Estimator drawn from the declared function class F.

    We implement either Ridge Regression or XGBoost to mirror Risk Relevance's
    (Definition 2.7) dependence on simple vs. rich function classes.
    """
    model_class = risk_cfg.get("model_class", "rich")
    if model_class == "linear":
        return Ridge(**risk_cfg.get("linear_params", {}))
    if model_class == "rich":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1, # single threading actually faster, less setup/teardown cost
            **risk_cfg.get("model_params", {}),
        )
    raise ValueError(
        f"Unknown risk model_class: {model_class!r}. Use 'linear' or 'rich'."
    )


def _risk_split(n, test_size, rng):
    """Share train/test splits across coalitions."""
    indices = rng.permutation(n)
    n_test = min(n - 1, max(1, int(round(test_size * n))))
    return indices[n_test:], indices[:n_test]


def _subset_risk(subset, X, y, risk_cfg, split, random_state):
    """Held-out squared-error risk of a predictor fit on ``X[:, subset]``."""
    train, test = split
    if len(subset) == 0:
        return float(np.mean((y[test] - y[train].mean()) ** 2))

    columns = list(subset)
    model = _fit_predictor(risk_cfg, random_state)
    model.fit(X[np.ix_(train, columns)], y[train])
    prediction = model.predict(X[np.ix_(test, columns)])
    return float(np.mean((y[test] - prediction) ** 2))


def risk_importance_details(X, y, feature_names, risk_cfg, rng):
    """Share coalitions in SAGE and minSHAP."""
    details = _risk_contribution_details(X, y, risk_cfg, rng)
    contributions = details[0]
    minshap = pd.Series(
        contributions.min(axis=0), index=feature_names, name="minshap"
    )
    sage = pd.Series(contributions.mean(axis=0), index=feature_names, name="sage")
    return minshap, sage, risk_contribution_frame(details, feature_names)


@register("minshap")
def minshap_importance(X, y, feature_names, risk_cfg, rng):
    """Risk-based minSHAP (2604.15107, Thm 2)."""
    minshap, _, _ = risk_importance_details(X, y, feature_names, risk_cfg, rng)
    return minshap


@register("sage")
def sage_importance(X, y, feature_names, risk_cfg, rng):
    """Ordinary risk-based Shapley value ("SAGE")."""
    _, sage, _ = risk_importance_details(X, y, feature_names, risk_cfg, rng)
    return sage


def _risk_contribution_details(X, y, risk_cfg, rng):
    """Internal helper for risk_contribution_details.

    This returns sampled contributions C(j | S) and associated coalitions S. We
    avoid re-computing risks if we've seen the coalition before, using the
    risk_cache  object.
    """
    n, d = X.shape
    n_orderings = risk_cfg["n_orderings"]
    base_seed = int(rng.integers(1, 2**31))
    split = _risk_split(n, risk_cfg.get("test_size", 0.3), rng)
    risk_cache = {}

    # helper to compute risk and save to cache
    def risk(subset):
        key = tuple(sorted(int(j) for j in subset))
        if key not in risk_cache:
            risk_cache[key] = _subset_risk(
                key, X, y, risk_cfg, split, random_state=base_seed
            )
        return risk_cache[key]

    # randomly sample coalitions and evaluate contributions
    contributions = np.empty((n_orderings, d))
    predecessors = [[None] * d for _ in range(n_orderings)]

    for ordering in range(n_orderings):
        perm = rng.permutation(d)
        prev = risk(())
        for position, target in enumerate(perm):
            prefix = tuple(int(j) for j in perm[:position])
            cur = risk(prefix + (int(target),))
            contributions[ordering, target] = prev - cur
            predecessors[ordering][target] = tuple(sorted(prefix))
            prev = cur

    return contributions, predecessors


def risk_contribution_frame(details, feature_names):
    """Convert sampled contributions to a tidy table for visualization.

    ``details`` must be the tuple returned by
    :func:`_risk_contribution_details`. Each row records the exact predecessor
    coalition used for one feature in one sampled ordering.
    """
    contributions, predecessors = details
    feature_names = tuple(feature_names)
    rows = []
    for ordering in range(contributions.shape[0]):
        for target, name in enumerate(feature_names):
            coalition_indices = predecessors[ordering][target]
            membership = np.zeros(len(feature_names), dtype=int)
            membership[list(coalition_indices)] = 1
            row = {
                "ordering": ordering + 1,
                "target": name,
                "contribution": contributions[ordering, target],
            }
            row.update({
                f"in_{feature}": int(membership[j])
                for j, feature in enumerate(feature_names)
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _gcm_pvalue(x, y, z, seed_x, seed_y, n_estimators):
    """Shah & Peters (2018) GCM test for x _||_ y | z.

    This is adapted from dowhy.gcm.independence_test.generalised_cov_measure but
    simplified to a fixed RandomForestRegressor.
    """
    # train models and get two sets of residuals
    model_x = RandomForestRegressor(n_estimators=n_estimators, random_state=seed_x)
    model_y = RandomForestRegressor(n_estimators=n_estimators, random_state=seed_y)
    model_x.fit(z, x)
    model_y.fit(z, y)
    resid_x = x - model_x.predict(z)
    resid_y = y - model_y.predict(z)

    # compute test statistic from normalized product of residuals. See Eqn (3)
    # from the Shah & Peters paper.
    products = resid_x * resid_y
    denom = np.std(products)
    if denom == 0:
        return 1.0
    stat = (np.sum(products) / np.sqrt(len(x))) / denom
    return 2 * _norm.sf(abs(stat))


@register("gcm")
def gcm_importance(X, y, feature_names, gcm_cfg, rng):
    """GCM conditional-independence: X_j _||_ Y | X_{-j}.

    We compute -log10(p-value) from the associated test. Larger values are
    evidence of dependence.
    """
    n_estimators = gcm_cfg.get("n_estimators", 100)
    y = np.asarray(y, dtype=float)
    scores = np.zeros(X.shape[1])

    # feature-by-feature retraining
    for j in range(X.shape[1]):
        z = np.delete(X, j, axis=1)
        seed_x = int(rng.integers(1, 2**31))
        seed_y = int(rng.integers(1, 2**31))
        p_value = _gcm_pvalue(X[:, j], y, z, seed_x, seed_y, n_estimators)
        scores[j] = -np.log10(max(p_value, 1e-300))

    return pd.Series(scores, index=feature_names, name="gcm")


@register("pdp_variance")
def pdp_variance(model, X, feature_names, grid_resolution):
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        result = pd_func(model, X, features=[j], grid_resolution=grid_resolution, kind="average")
        scores[j] = np.var(result["average"][0])

    return pd.Series(scores, index=feature_names, name="pdp_variance")


@register("integrated_gradients")
def integrated_gradients_importance(model, X, feature_names, ig_cfg, seed, rng, response_type):
    n_samples = min(ig_cfg["n_samples"], X.shape[0])
    sample_idx = rng.choice(X.shape[0], n_samples, replace=False)

    # Use mean of data as baseline for tabular data
    baseline = np.mean(X, axis=0)
    explainer = presets.integrated_gradients(baseline, ig_cfg["n_steps"], eps=ig_cfg.get("eps", 1e-5))
    if response_type == "classification":
        f = lambda batch: model.predict_proba(batch)[:, 1]
    else:
        f = lambda batch: model.predict(batch)

    attr_matrix = np.zeros((n_samples, X.shape[1]))
    for i, idx in enumerate(sample_idx):
        res = explainer.explain(f, X[idx])
        attr_matrix[i] = res.as_array()

    return pd.Series(
        np.mean(np.abs(attr_matrix), axis=0), index=feature_names, name="integrated_gradients"
    )
