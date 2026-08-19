"""Synthetic data-generating processes for the Sweep 1 case study.

Config parameters:
    n_features -- total columns (signal + null pads)
    n_nonnull -- signal columns. rest are iid N(0,1)
    gamma, beta, noise_scale -- DGP-specific parameters
    response_type -- 'classification' or 'regression'
    sigma_y -- noise std for regression targets

Feature names: x1,...,x{n_nonnull} (signal), noise_1,...,noise_{n_features-n_nonnull} (null).

Caution: product_interaction and dependent_features require even n_nonnull.

Mean functions are defined in means.py. These apply both to classification

    Y ~ Bernoulli(sigmoid(mean))

and regression

    Y ~ N(mean, sigma_y^2)

Mean formulas (s(x) denotes signal function):
    linear_additive: s(x) = beta * sum(x_j)
    parity: s(x) = -gamma * sum over groups of prod(sign(x_j)), x_j ∈ U[-1,1]
    product_interaction: s(x) = gamma * sum_k(x_{2k-1} * x_{2k})
    dependent_features: s(x) = gamma * sum_j(z_j), with x_{2j-1}=z_j, x_{2j}=z_j+eps
    mediated_chains: x_j=x_{j-1}+eps within chains; s(x) uses each terminal
    confounding: s(x) = gamma * sum_j(z_j), with x_j=z_j+eps (z_j unobserved)
    quadratic: s(x) = gamma * sum(x_j^2 - 1)
    redundant_pair: s(x) = gamma * sum_k(z_k), with x_{2k-1}=x_{2k}=z_k
    bayes_incomplete: s(x) = gamma * sum_k(u_k^2 - 1/3), with x_{2k}=u_k^2+eps
    heteroscedastic: s(x) = gamma * sum_k(m_k); x_{2k} rescales the noise
"""
import numpy as np
import pandas as pd
from means import MEAN_FNS, chain_terminal_indices

DATASETS = {}
NULL_NOTIONS = ("marginal", "conditional", "risk", "functional", "causal")

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


def _make_response(rng, mean, response_type, sigma_y, scale=None):
    """Generate response variable

    Args:
        rng: numpy.random.RandomState
        mean: np.ndarray of mean values
        response_type: 'classification' or 'regression'
        sigma_y: noise std for regression (ignored for classification)
        scale: rescale the regression errors by this scale, not used by
        classification.

    Returns:
        np.ndarray: response values (int for classification, float for regression)
    """
    if response_type == "classification":
        if scale is not None:
            raise ValueError("heteroscedastic noise is only defined for regression; a ")
        return rng.binomial(1, _sigmoid(mean))

    noise = sigma_y * rng.standard_normal(len(mean))
    return mean + (noise if scale is None else scale * noise)


def _make_feature_names(n_nonnull, n_features):
    """x1..x{n_nonnull}  followed by  noise_1..noise_{n_features - n_nonnull}."""
    signal = [f"x{j + 1}" for j in range(n_nonnull)]
    noise  = [f"noise_{j + 1}" for j in range(n_features - n_nonnull)]
    return signal + noise


def _null_cols(rng, n_noise, n):
    """Return a list of n_noise independent iid N(0,1) arrays of length n."""
    return [rng.standard_normal(n) for _ in range(n_noise)]


def _build_latent_pairs(n, n_groups, rng, pair_noise, feature_dist="normal"):
    """
    This draws pairs of nearly identical features, similar to what's needed in
    example 3.3 and 4.2.
    """
    draw_latent = (
        rng.standard_normal
        if feature_dist == "normal"
        else lambda size: rng.binomial(1, 0.5, size=size).astype(float)
    )

    signal, latents = [], []
    for _ in range(n_groups):
        z = draw_latent(n)
        latents.append(z)
        signal.extend([z, z + pair_noise * rng.standard_normal(n)])

    return signal, latents


def _pair_null_type(n_nonnull, anchor_null, proxy_null):
    """Assign labels to the anchor and (null) proxy in each pair."""
    return {
        f"x{j + 1}": list(anchor_null if j % 2 == 0 else proxy_null)
        for j in range(n_nonnull)
    }


def _make_df(arrays, feature_names):
    return pd.DataFrame({name: arr for name, arr in zip(feature_names, arrays)})


def _dim(cfg):
    """Extract (n_nonnull, n_features, n_noise) from a configuration dictionary."""
    n_nonnull  = cfg.get("n_nonnull",  2)
    n_features = cfg.get("n_features", 3)
    return n_nonnull, n_features, n_features - n_nonnull


def dataset_response_types(cfg_dict, name):
    """Small helper function to determine the response type from a configuration dictionary"""
    return cfg_dict["datasets"][name].get(
        "response_types", cfg_dict["response_types"]
    )


def _full_null():
    return list(NULL_NOTIONS)


def _simulate_response(n, rng, cfg, name, mean_cols, scale=None):
    """Generate the null and response columns a dataset.

    mean_cols -- arrays passed to MEAN_FNS[name] to compute the mean
    scale -- rescale the noise, see :func:`_make_response`

    Returns (noise, y, response_type, sigma_y).
    """
    _, _, n_noise = _dim(cfg)
    response_type = cfg.get("response_type", "classification")
    sigma_y = cfg.get("sigma_y", 1.0)

    noise = _null_cols(rng, n_noise, n)
    mean = MEAN_FNS[name](mean_cols, cfg)
    y = _make_response(rng, mean, response_type, sigma_y, scale=scale)
    return noise, y, response_type, sigma_y


def _build_meta(feature_names, n_nonnull, null_type, equation, response_type, sigma_y, extra_meta=None):
    """ Create a metadata dictionary summarizing a dataset."""
    full_null_type = dict(null_type)
    full_null_type.update({fname: _full_null() for fname in feature_names[n_nonnull:]})
    return {
        "null_type": full_null_type,
        "equation": equation,
        "response_type": response_type,
        "sigma_y": sigma_y,
        **(extra_meta or {}),
    }


# ---------------------------------------------------------------------------
# Mediated chains helpers
# ---------------------------------------------------------------------------

def _build_chain_signal(n, n_nonnull, chain_length, transition_noise, rng):
    """Build independent directed chains of correlated features.

    Each chain starts from an independent N(0,1) draw and propagates with
    additive Gaussian noise at each step.

    Returns
    -------
    list of np.ndarray
        Signal columns in feature order (all chain elements concatenated).
    """
    signal = []
    for start in range(0, n_nonnull, chain_length):
        current = rng.standard_normal(n)
        signal.append(current)
        stop = min(start + chain_length, n_nonnull)
        for _ in range(start + 1, stop):
            current = current + transition_noise * rng.standard_normal(n)
            signal.append(current)
    return signal


def _build_chain_null_type(n_nonnull, terminal_indices):
    """Build null_type for mediated_chains.

    Terminal nodes are non-null ([]); internal chain nodes are null under the
    conditional, risk, and functional notions (but not marginal or causal).
    """
    internal = ["conditional", "risk", "functional"]
    return {
        f"x{j + 1}": ([] if j in terminal_indices else list(internal))
        for j in range(n_nonnull)
    }


def _compute_chain_lengths(n_nonnull, chain_length):
    """Actual lengths of each chain (last chain may be shorter)."""
    return [
        min(chain_length, n_nonnull - start)
        for start in range(0, n_nonnull, chain_length)
    ]


# ---------------------------------------------------------------------------
# DGP functions
# ---------------------------------------------------------------------------

@register("linear_additive")
def linear_additive(n, rng, cfg):
    """E[Y|x] = beta * sum(x_j), x_j ~ N(0,1).
    """
    n_nonnull, n_features, _ = _dim(cfg)
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "linear_additive", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": [] for j in range(n_nonnull)}
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=f"E[Y|x] = beta * sum(x_1, ..., x_{n_nonnull})",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"beta": cfg.get("beta", 4.0)},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("parity")
def parity(n, rng, cfg):
    """E[Y|x] = -gamma * sum over groups of prod(sign(x_j)), x_j ~ U[-1,1].

    The signal features here are marginally but not conditionally known. When
    `group_size` is set to 2, then this becomes the XOR function (example 3.1).
    """
    # define data
    n_nonnull, n_features, _ = _dim(cfg)
    signal = [rng.uniform(-1.0, 1.0, size=n) for _ in range(n_nonnull)]
    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "parity", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)

    # annotate
    null_type = {f"x{j + 1}": ["marginal"] for j in range(n_nonnull)}
    group_size = cfg.get("group_size", 2)
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=(
            f"E[Y|x] = -gamma * sum of prod(sign(x_j)) over groups of "
            f"{group_size}, x_j in U[-1,1]"
        ),
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0), "group_size": group_size},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("product_interaction")
def product_interaction(n, rng, cfg):
    """E[Y|x] = gamma·sum_k(x_{2k-1}·x_{2k}), x_i ~ N(0,1).

    Even though the response is uncorrelated with each feature, we consider this
    marginally nonull because the V(Y | X_j) is related to X_j. So, I(X_j; Y)
    \neq 0.
    """
    # define data
    n_nonnull, n_features, _ = _dim(cfg)
    n_pairs = n_nonnull // 2
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]
    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "product_interaction", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)

    # annotate
    null_type = {f"x{j + 1}": [] for j in range(n_nonnull)}
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=f"E[Y|x] = gamma * sum_{{k=1}}^{{{n_pairs}}} x_{{2k-1}} * x_{{2k}}",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0), "n_pairs": n_pairs},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("dependent_features")
def dependent_features(n, rng, cfg):
    """E[Y|z] = gamma·sum(z_j).

    Defined by x_{2j-1} = z_j (anchor), x_{2j} = z_j + ε (proxy).
    The anchors are non-null; the proxies are null under every notion except
    the marginal one (they correlate with their anchor).
    """
    n_nonnull, n_features, _ = _dim(cfg)
    n_groups = n_nonnull // 2
    noise_scale = cfg.get("noise_scale", 0.3)

    # Define the true signals and correlated dependents.
    signal, latents = _build_latent_pairs(
        n, n_groups, rng, noise_scale
    )

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "dependent_features", latents)
    feature_names = _make_feature_names(n_nonnull, n_features)
    proxy_null = ["conditional", "risk", "functional", "causal"]
    null_type = _pair_null_type(n_nonnull, [], proxy_null)
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation="E[Y|z] = gamma * sum(z_j), x_{2j-1}=z_j, x_{2j}=z_j+eps",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0), "noise_scale": noise_scale, "n_groups": n_groups},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("mediated_chains")
def mediated_chains(n, rng, cfg):
    """Several independent directed chains whose terminal nodes determine Y.

    This follows Example 2.1 in arXiv:2604.15107v2
    """
    # extract relevant parameters
    n_nonnull, n_features, _ = _dim(cfg)
    chain_length = cfg.get("chain_length", 3)
    transition_noise = cfg.get("transition_noise", 1.0)
    terminal_indices = chain_terminal_indices(n_nonnull, chain_length)

    # create the features and response
    signal = _build_chain_signal(n, n_nonnull, chain_length, transition_noise, rng)
    terminals = [signal[j] for j in terminal_indices]
    noise, y, response_type, sigma_y = _simulate_response(
        n, rng, cfg, "mediated_chains", terminals
    )

    # metadata about real vs. null features in the chain example
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = _build_chain_null_type(n_nonnull, terminal_indices)
    chain_lengths = _compute_chain_lengths(n_nonnull, chain_length)
    terminal_names = [f"x{j + 1}" for j in terminal_indices]
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=(
            "Within each chain x_j=x_{j-1}+tau*eps; "
            "E[Y|x] = gamma/sqrt(n_chains) * sum(chain terminals)"
        ),
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={
            "gamma": cfg.get("gamma", 3.0),
            "chain_length": chain_length,
            "chain_lengths": chain_lengths,
            "transition_noise": transition_noise,
            "terminal_features": terminal_names,
            "n_chains": len(terminal_indices),
        },
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("quadratic")
def quadratic(n, rng, cfg):
    """E[Y|x] = gamma * sum(x_j^2 - 1), x_j ~ N(0,1).

    Signal features have zero marginal covariance with the response, Cov(x_j,
    y)=0, because (x_j) is symmetric about zero and the noise is independent.
    Still marginally non-null despite the lack of correlation. Also considered
    functional, conditional, risk, and causal non-marginal, see Zheng and
    Raskutti ("Comparing Model-agnostic Feature Selection Methods through
    Relative Efficiency", Example 2.1.1)
    """
    n_nonnull, n_features, _ = _dim(cfg)
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]
    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "quadratic", signal)

    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": [] for j in range(n_nonnull)}
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=f"E[Y|x] = gamma * sum(x_1^2 - 1, ..., x_{n_nonnull}^2 - 1)",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0)},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("confounding")
def confounding(n, rng, cfg):
    """E[Y|z] = gamma·sum(z_j), x_j = z_j + ε (z_j unobserved).

    The x_j's are causally null (do(x_j) does not affect Y), but marginally and
    conditionally non-null (consider the graph).
    """
    n_nonnull, n_features, _ = _dim(cfg)
    noise_scale = cfg.get("noise_scale", 0.3)

    # define the z -> x -> y path
    latents = [rng.standard_normal(n) for _ in range(n_nonnull)]
    signal = [z + noise_scale * rng.standard_normal(n) for z in latents]

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "confounding", latents)
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": ["causal"] for j in range(n_nonnull)}
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation="E[Y|z] = gamma * sum(z_j), x_j = z_j + eps (z_j unmeasured)",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0), "noise_scale": noise_scale},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("redundant_pair")
def redundant_pair(n, rng, cfg):
    """Conditional and risk null without functional null (Example 3.3).

    This is very similar to the dependent features generator, except the
    features have lower noise by default. We added a little bit of noise because
    otherwise the knockoff and GCM methods crash with numerical issues.
    """
    # Simulate the data
    n_nonnull, n_features, _ = _dim(cfg)
    n_groups = n_nonnull // 2
    duplicate_noise = cfg.get("duplicate_noise", 0.0)
    feature_dist = cfg.get("feature_dist", "normal")
    signal, latents = _build_latent_pairs(n, n_groups, rng, duplicate_noise, feature_dist)
    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "redundant_pair", latents)

    # Annotation
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = _pair_null_type(
        n_nonnull,
        ["conditional", "risk"] if duplicate_noise == 0.0 else [],
        ["conditional", "risk", "causal"],
    )
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=(
            "E[Y|z] = gamma * sum(z_k), x_{2k-1} = z_k, x_{2k} = z_k + tau*eps"
        ),
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={
            "gamma": cfg.get("gamma", 3.0),
            "duplicate_noise": duplicate_noise,
            "feature_dist": feature_dist,
            "n_groups": n_groups,
        },
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("bayes_incomplete")
def bayes_incomplete(n, rng, cfg):
    """Conditional and functional but not risk null (see Example 3.4). """
    n_nonnull, n_features, _ = _dim(cfg)
    n_groups = n_nonnull // 2
    proxy_noise = cfg.get("proxy_noise", 0.05)

    # Generate latent u_k and pairs (u_k, u_k^2 + eps)
    anchors = [rng.uniform(-1.0, 1.0, size=n) for _ in range(n_groups)]
    signal = [
        feature for u in anchors
        for feature in (u, u**2 + proxy_noise * rng.standard_normal(n))
    ]

    # Define the response E[Y | u]
    noise, y, response_type, sigma_y = _simulate_response(
        n, rng, cfg, "bayes_incomplete", anchors
    )

    # Annotations
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = _pair_null_type(
        n_nonnull,
        anchor_null=[],
        proxy_null=["conditional", "functional", "causal"],
    )
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=(
            "E[Y|u] = gamma * sum(u_k^2 - 1/3), x_{2k-1} = u_k ~ U[-1,1], "
            "x_{2k} = u_k^2 + eps"
        ),
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={
            "gamma": cfg.get("gamma", 3.0),
            "proxy_noise": proxy_noise,
            "n_groups": n_groups,
        },
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("heteroscedastic")
def heteroscedastic(n, rng, cfg):
    """Risk and functional but not conditional null.

    The even features are related to the variance. The odd features are related
    to the mean. Those which are related only to the variance are risk and
    functional nulls, but they're not conditionally null because there is shared
    information across those features and y.
    """
    n_nonnull, n_features, _ = _dim(cfg)
    n_groups = n_nonnull // 2
    variance_scale = cfg.get("variance_scale", 1.0)

    # Generate the mean and variance features.
    signal, means, variances = [], [], []
    for _ in range(n_groups):
        m = rng.standard_normal(n)
        v = rng.standard_normal(n)
        means.append(m)
        variances.append(v)
        signal.append(m)
        signal.append(v)

    # Keeps the noise scale comparable across different group sizes.
    scale = np.exp(0.5 * variance_scale * sum(variances) / np.sqrt(n_groups))
    noise, y, response_type, sigma_y = _simulate_response(
        n, rng, cfg, "heteroscedastic", means, scale=scale
    )

    # Record the annotation.
    feature_names = _make_feature_names(n_nonnull, n_features)
    variance_null = ["risk", "functional"]
    null_type = {
        f"x{j + 1}": ([] if j % 2 == 0 else list(variance_null))
        for j in range(n_nonnull)
    }

    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=(
            "E[Y|x] = gamma * sum(x_{2k-1}); "
            "sd(Y|x) = sigma_y * exp(kappa * sum(x_{2k}) / (2 sqrt(n_groups)))"
        ),
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={
            "gamma": cfg.get("gamma", 3.0),
            "variance_scale": variance_scale,
            "n_groups": n_groups,
        },
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta
