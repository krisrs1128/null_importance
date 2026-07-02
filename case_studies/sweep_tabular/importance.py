"""Consistent interface to feature importance methods."""

import logging
import numpy as np
import pandas as pd
import shap
import knockpy
from rf import fit_final
from axiom_interp import presets
from sklearn.metrics import matthews_corrcoef as mcc_score
from sklearn.inspection import permutation_importance as pi
from scipy.stats import pointbiserialr
from sklearn.inspection import partial_dependence as pd_func

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
def permutation(model, X, y, feature_names, n_repeats, rng):
    result = pi(
        model, X, y,
        scoring="matthews_corrcoef",
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


@register("minshap")
def minshap_importance(model, X, feature_names, mcfg, seed, rng):
    n_samples = min(mcfg["n_samples"], X.shape[0])
    sample_idx = rng.choice(X.shape[0], n_samples, replace=False)

    n_bg = min(mcfg["n_background"], X.shape[0])
    bg = X[rng.choice(X.shape[0], n_bg, replace=False)]

    explainer = presets.minshap(bg, mcfg["n_orderings"], seed)
    f = lambda batch: model.predict_proba(batch)[:, 1]

    attr_matrix = np.zeros((n_samples, X.shape[1]))
    for i, idx in enumerate(sample_idx):
        res = explainer.explain(f, X[idx])
        attr_matrix[i] = res.as_array()
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  minshap [{i + 1}/{n_samples}]")

    return pd.Series(
        np.mean(np.abs(attr_matrix), axis=0), index=feature_names, name="minshap"
    )


@register("kernelshap")
def kernelshap(model, X, feature_names, kshap_cfg, rng):
    n_samples = min(kshap_cfg["n_samples"], X.shape[0])
    sample_idx = rng.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[sample_idx]

    n_bg = min(kshap_cfg["n_background"], X.shape[0])
    bg = X[rng.choice(X.shape[0], n_bg, replace=False)]

    f = lambda batch: model.predict_proba(batch)[:, 1]
    explainer = shap.KernelExplainer(f, bg)
    sv = explainer.shap_values(X_sample, nsamples=kshap_cfg["n_coalitions"])

    return pd.Series(
        np.mean(np.abs(sv), axis=0), index=feature_names, name="kernelshap"
    )


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
def correlation(X, y, feature_names):
    scores = np.array([abs(pointbiserialr(y, X[:, j])[0]) for j in range(X.shape[1])])
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
def integrated_gradients_importance(model, X, feature_names, ig_cfg, seed, rng):
    n_samples = min(ig_cfg["n_samples"], X.shape[0])
    sample_idx = rng.choice(X.shape[0], n_samples, replace=False)

    # Use mean of data as baseline for tabular data
    baseline = np.mean(X, axis=0)
    explainer = presets.integrated_gradients(baseline, ig_cfg["n_steps"], eps=ig_cfg.get("eps", 1e-5))
    f = lambda batch: model.predict_proba(batch)[:, 1]

    attr_matrix = np.zeros((n_samples, X.shape[1]))
    for i, idx in enumerate(sample_idx):
        res = explainer.explain(f, X[idx])
        attr_matrix[i] = res.as_array()
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  integrated_gradients [{i + 1}/{n_samples}]")

    return pd.Series(
        np.mean(np.abs(attr_matrix), axis=0), index=feature_names, name="integrated_gradients"
    )