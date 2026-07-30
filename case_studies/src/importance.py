"""Consistent interface to feature importance methods."""

import logging
import numpy as np
import pandas as pd
import shap
import knockpy
from rf import fit_final
from axiom_interp import presets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import matthews_corrcoef as mcc_score
from sklearn.inspection import permutation_importance as pi
from scipy.stats import pointbiserialr, pearsonr, norm as _norm
from sklearn.inspection import partial_dependence as pd_func
import xgboost as xgb

log = logging.getLogger(__name__)

METHODS = {}


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
def loco(X_df, y, feature_names, model, cfg, seed):
    baseline_pred = model.classes_[model.oob_decision_function_.argmax(axis=1)]
    full_mcc = mcc_score(y, baseline_pred)

    mdi_order = np.argsort(model.feature_importances_)[::-1]
    top_k = cfg["loco"]["top_k"]
    top_features = [feature_names[i] for i in mdi_order[:top_k]]

    y_series = pd.Series(y, index=X_df.index)
    scores = {}
    for i, feat in enumerate(top_features):
        X_drop = X_df.drop(columns=[feat])
        fresh_rng = np.random.default_rng(seed)
        reduced_model, _ = fit_final(X_drop, y_series, cfg, fresh_rng)
        reduced_pred = reduced_model.classes_[
            reduced_model.oob_decision_function_.argmax(axis=1)
        ]
        scores[feat] = full_mcc - mcc_score(y, reduced_pred)
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  loco [{i + 1}/{top_k}] {feat}: {scores[feat]:.4f}")

    result = pd.Series(0.0, index=feature_names, name="loco")
    for feat, val in scores.items():
        result[feat] = val
    return result


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


def _subset_risk(subset, X, y, model_params, random_state):
    """In-sample MSE of an XGBoost regressor fit on X[:, subset].
    """
    if len(subset) == 0:
        return float(np.mean((y - y.mean()) ** 2))

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        **model_params,
    )
    X_subset = X[:, subset]
    model.fit(X_subset, y)
    y_pred = model.predict(X_subset)
    return float(np.mean((y - y_pred) ** 2))


@register("minshap")
def minshap_importance(X, y, feature_names, risk_cfg, rng):
    """Risk-based minSHAP: subset-refit value functions (2604.15107, Thm 2)."""
    contributions = _risk_contributions(X, y, risk_cfg["n_orderings"],
                                         risk_cfg["model_params"], rng)
    return pd.Series(contributions.min(axis=0), index=feature_names, name="minshap")


@register("sage")
def sage_importance(X, y, feature_names, risk_cfg, rng):
    """Ordinary risk-Shapley value: mean over sampled orderings of the same
    subset-risk contributions minSHAP takes the min over (2604.15107 §2 — the
    un-minned Shapley value of the risk game V(S) = -inf_f E[l(Y,f_S(X_S))],
    matching Covert/Lundberg/Lee 2020 "SAGE")."""
    contributions = _risk_contributions(X, y, risk_cfg["n_orderings"],
                                         risk_cfg["model_params"], rng)
    return pd.Series(contributions.mean(axis=0), index=feature_names, name="sage")


def _risk_contributions(X, y, n_orderings, model_params, rng):
    """Permutation-sampled subset-risk contributions (arxiv 2604.15107, Thm 2 setup).

    Parameters
    ----------
    X : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    n_orderings : int
        Number of sampled permutations.
    model_params : dict
        Forwarded to _subset_risk / xgb.XGBRegressor.
    rng : np.random.Generator
        Sole source of randomness.

    Returns
    -------
    ndarray of shape (n_orderings, d)
        contributions[k, j] = VI_j^{pi_k}: risk drop from adding feature j to
        its prefix under the k-th sampled ordering. minSHAP = min over k; the
        mean over k is the ordinary (SAGE-style) risk-Shapley value.
    """
    _, d = X.shape
    risk_empty = _subset_risk([], X, y, model_params, random_state=None)
    contributions = np.empty((n_orderings, d))

    for ordering in range(n_orderings):
        perm = rng.permutation(d)
        prev = risk_empty
        for i in range(d):
            cur = _subset_risk(
                perm[: i + 1], X, y, model_params,
                random_state=int(rng.integers(1, 2**31)),
            )
            contributions[ordering, perm[i]] = prev - cur
            prev = cur

    return contributions


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
        if (j + 1) % 100 == 0:
            log.info(f"  pdp_variance [{j + 1}/{X.shape[1]}]")

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
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  integrated_gradients [{i + 1}/{n_samples}]")

    return pd.Series(
        np.mean(np.abs(attr_matrix), axis=0), index=feature_names, name="integrated_gradients"
    )
