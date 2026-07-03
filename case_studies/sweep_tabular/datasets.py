"""Synthetic data-generating processes for the Sweep 1 case study.

Config parameters:
    n_features  -- total columns (signal + null pads)
    n_nonnull   -- signal columns. rest are iid N(0,1)
    gamma, beta, noise_scale -- DGP-specific parameters

Feature names: x1,...,x{n_nonnull} (signal), noise_1,...,noise_{n_features-n_nonnull} (null).

Caution: product_interaction and dependent_features require even n_nonnull.

DGPs (logit(Y) = ...):
    linear_additive    : beta * sum(x_j)
    xor                : -gamma * prod(x_j), x_j ∈ {-1,+1}
    product_interaction: gamma * sum_k(x_{2k-1} * x_{2k})
    dependent_features : gamma * sum_j(z_j), with x_{2j-1}=z_j, x_{2j}=z_j+eps
    confounding        : gamma * sum_j(z_j), with x_j=z_j+eps (z_j unobserved)
"""
import numpy as np
import pandas as pd

DATASETS = {}

def register(name):
    def decorator(fn):
        DATASETS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bernoulli(rng, logit):
    return rng.binomial(1, _sigmoid(logit))


def _make_feature_names(n_nonnull, n_features):
    """x1..x{n_nonnull}  followed by  noise_1..noise_{n_features - n_nonnull}."""
    signal = [f"x{j + 1}" for j in range(n_nonnull)]
    noise  = [f"noise_{j + 1}" for j in range(n_features - n_nonnull)]
    return signal + noise


def _null_cols(rng, n_noise, n):
    """Return a list of n_noise independent iid N(0,1) arrays of length n."""
    return [rng.standard_normal(n) for _ in range(n_noise)]


def _make_df(arrays, feature_names):
    return pd.DataFrame({name: arr for name, arr in zip(feature_names, arrays)})


def _dim(cfg):
    """Extract (n_nonnull, n_features, n_noise) from the merged cfg dict."""
    n_nonnull  = cfg.get("n_nonnull",  2)
    n_features = cfg.get("n_features", 3)
    return n_nonnull, n_features, n_features - n_nonnull


def _full_null():
    return ["functional", "marginal", "conditional", "causal"]


# ---------------------------------------------------------------------------
# DGP functions
# ---------------------------------------------------------------------------

@register("linear_additive")
def linear_additive(n, rng, cfg):
    """Logit(Y) = \sum_{nonnull}(beta * x_j), x_j ~ N(0,1)."""
    n_nonnull, n_features, n_noise = _dim(cfg)
    beta = cfg.get("beta", 4.0)

    # create signal and response
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]
    noise = _null_cols(rng, n_noise, n)
    y = _bernoulli(rng, beta * sum(signal))

    # create and merge in the nulls
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {name: [] for name in feature_names[:n_nonnull]}
    null_type.update({name: _full_null() for name in feature_names[n_nonnull:]})
    meta = {
        "null_type": null_type,
        "equation": f"logit(Y) = beta * sum(x_1, ..., x_{n_nonnull})",
        "beta": beta,
    }
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("xor")
def xor(n, rng, cfg):
    """Logit(Y) = -gamma·prod(x_j), x_j ~ Rademacher. Marginally but not
    functionally null."""
    n_nonnull, n_features, n_noise = _dim(cfg)
    gamma = cfg.get("gamma", 3.0)

    # define the signal
    signal = [rng.choice([-1.0, 1.0], size=n) for _ in range(n_nonnull)]
    product = np.prod(signal, axis=0)
    noise = _null_cols(rng, n_noise, n)
    y = _bernoulli(rng, -gamma * product)

    # create and merge in the null
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {name: ["marginal"] for name in feature_names[:n_nonnull]}
    null_type.update({name: _full_null() for name in feature_names[n_nonnull:]})
    meta = {
        "null_type": null_type,
        "equation": f"logit(Y) = -gamma * prod(x_1, ..., x_{n_nonnull}), x_j in {{-1,+1}}",
        "gamma": gamma,
    }
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("product_interaction")
def product_interaction(n, rng, cfg):
    """Logit(Y) = gamma·sum_k(x_{2k-1}·x_{2k}), x_i ~ N(0,1). Requires even n_nonnull.

    Each feature marginally null; pairs jointly non-null.
    """
    n_nonnull, n_features, n_noise = _dim(cfg)
    gamma = cfg.get("gamma", 3.0)

    # check for valid input
    if n_nonnull % 2 != 0:
        raise ValueError(
            f"product_interaction requires even n_nonnull (nonoverlapping pairs); "
            f"got n_nonnull={n_nonnull}."
        )
    n_pairs = n_nonnull // 2

    # defin the signal and nulls
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]
    logit = gamma * sum(signal[2*k] * signal[2*k + 1] for k in range(n_pairs))
    noise = _null_cols(rng, n_noise, n)
    y = _bernoulli(rng, logit)

    # wrap into a response df
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {name: ["marginal"] for name in feature_names[:n_nonnull]}
    null_type.update({name: _full_null() for name in feature_names[n_nonnull:]})
    meta = {
        "null_type": null_type,
        "equation": f"logit(Y) = gamma * sum_{{k=1}}^{{{n_pairs}}} x_{{2k-1}} * x_{{2k}}",
        "gamma": gamma,
        "n_pairs": n_pairs,
    }
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("dependent_features")
def dependent_features(n, rng, cfg):
    """Logit(Y) = gamma·sum(z_j). Requires even n_nonnull.

    Defined as x_{2j-1} = z_j (anchor), x_{2j} = z_j + ε (proxy).
    The anchors are non-null but proxies are conditionally null.
    """
    n_nonnull, n_features, n_noise = _dim(cfg)
    gamma = cfg.get("gamma", 3.0)
    noise_scale = cfg.get("noise_scale", 0.3)

    # check valid inputs
    if n_nonnull % 2 != 0:
        raise ValueError(
            f"dependent_features requires even n_nonnull (anchor+proxy groups); "
            f"got n_nonnull={n_nonnull}."
        )
    n_groups = n_nonnull // 2

    # define the true signals and correlated dependents
    signal, latents = [], []
    for _ in range(n_groups):
        z = rng.standard_normal(n)
        latents.append(z)
        signal.append(z) # anchor
        signal.append(z + noise_scale * rng.standard_normal(n))  # proxy

    # create the response
    logit = gamma * sum(latents)
    noise = _null_cols(rng, n_noise, n)
    y = _bernoulli(rng, logit)

    # save metadata about the feature types
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {}
    for j in range(n_nonnull):
        null_type[feature_names[j]] = [] if j % 2 == 0 else ["conditional"]
    null_type.update({name: _full_null() for name in feature_names[n_nonnull:]})
    meta = {
        "null_type": null_type,
        "equation": "logit(Y) = gamma * sum(z_j), x_{2j-1}=z_j, x_{2j}=z_j+eps",
        "gamma": gamma,
        "noise_scale": noise_scale,
        "n_groups": n_groups,
    }
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("confounding")
def confounding(n, rng, cfg):
    """Logit(Y) = gamma·sum(z_j), x_j = z_j + ε (z_j unobserved).

    The x_j's are causally null (do(x_j) does not affect Y), but marginally and
    conditionally non-null (consider the graph)
    """
    n_nonnull, n_features, n_noise = _dim(cfg)
    gamma = cfg.get("gamma", 3.0)
    noise_scale = cfg.get("noise_scale", 0.3)

    # define the z -> x -> y path
    latents = [rng.standard_normal(n) for _ in range(n_nonnull)]
    signal = [z + noise_scale * rng.standard_normal(n) for z in latents]
    logit = gamma * sum(latents)
    noise = _null_cols(rng, n_noise, n)
    y = _bernoulli(rng, logit)

    # track the null types of each variable
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {name: ["causal"] for name in feature_names[:n_nonnull]}
    null_type.update({name: _full_null() for name in feature_names[n_nonnull:]})
    meta = {
        "null_type": null_type,
        "equation": "logit(Y) = gamma * sum(z_j), x_j = z_j + eps (z_j unmeasured)",
        "gamma": gamma,
        "noise_scale": noise_scale,
    }
    return _make_df(signal + noise, feature_names), y, feature_names, meta
