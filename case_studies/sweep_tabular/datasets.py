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
    parity: s(x) = -gamma * prod(sign(x_j)), x_j ∈ U[-1,1]
    product_interaction: s(x) = gamma * sum_k(x_{2k-1} * x_{2k})
    dependent_features: s(x) = gamma * sum_j(z_j), with x_{2j-1}=z_j, x_{2j}=z_j+eps
    mediated_chains: x_j=x_{j-1}+eps within chains; s(x) uses each terminal
    confounding: s(x) = gamma * sum_j(z_j), with x_j=z_j+eps (z_j unobserved)
    quadratic: s(x) = gamma * sum(x_j^2 - 1)
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


def _make_response(rng, mean, response_type, sigma_y):
    """Generate response variable

    Args:
        rng: numpy.random.RandomState
        mean: np.ndarray of mean values
        response_type: 'classification' or 'regression'
        sigma_y: noise std for regression (ignored for classification)

    Returns:
        np.ndarray: response values (int for classification, float for regression)
    """
    if response_type == "classification":
        return rng.binomial(1, _sigmoid(mean))
    elif response_type == "regression":
        return mean + sigma_y * rng.standard_normal(len(mean))
    else:
        raise ValueError(f"Unknown response_type: {response_type}. Use 'classification' or 'regression'.")


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
    return list(NULL_NOTIONS)


def _simulate_response(n, rng, cfg, name, mean_cols):
    """Generate the null and response columns a dataset.

    mean_cols -- arrays passed to MEAN_FNS[name] to compute the mean

    Returns (noise, y, response_type, sigma_y).
    """
    _, _, n_noise = _dim(cfg)
    response_type = cfg.get("response_type", "classification")
    sigma_y = cfg.get("sigma_y", 1.0)

    noise = _null_cols(rng, n_noise, n)
    mean = MEAN_FNS[name](mean_cols, cfg)
    y = _make_response(rng, mean, response_type, sigma_y)
    return noise, y, response_type, sigma_y


def _build_meta(feature_names, n_nonnull, null_type, equation, response_type, sigma_y, extra_meta=None):
    """Assemble a DGP's meta dict.

    null_type -- dict mapping each x_j's feature name to its null_type list.
    """
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
    """E[Y|x] = -gamma * prod(sign(x_j)), x_j ~ U[-1,1].

    Signal columns are marginally but not functionally null: E[Y|x_j] doesn't
    depend on sign(x_j), because the product of the *other* signs is itself
    symmetric ±1 regardless of x_j's sign (it's 0.5 for classification, 0 for
    regression).
    """
    n_nonnull, n_features, _ = _dim(cfg)
    signal = [rng.uniform(-1.0, 1.0, size=n) for _ in range(n_nonnull)]

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "parity", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": ["marginal"] for j in range(n_nonnull)}
    meta = _build_meta(
        feature_names, n_nonnull, null_type,
        equation=f"E[Y|x] = -gamma * prod(sign(x_1), ..., sign(x_{n_nonnull})), x_j in U[-1,1]",
        response_type=response_type, sigma_y=sigma_y,
        extra_meta={"gamma": cfg.get("gamma", 3.0)},
    )
    return _make_df(signal + noise, feature_names), y, feature_names, meta


@register("product_interaction")
def product_interaction(n, rng, cfg):
    """E[Y|x] = gamma·sum_k(x_{2k-1}·x_{2k}), x_i ~ N(0,1).

    n_nonnull must be even. Each feature is marginally null, but the pairs are
    jointly non-null.
    """
    n_nonnull, n_features, _ = _dim(cfg)
    if n_nonnull % 2 != 0:
        raise ValueError(
            f"product_interaction requires even n_nonnull (nonoverlapping pairs); "
            f"got n_nonnull={n_nonnull}."
        )
    n_pairs = n_nonnull // 2
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "product_interaction", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": ["marginal"] for j in range(n_nonnull)}
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
    if n_nonnull % 2 != 0:
        raise ValueError(
            f"dependent_features requires even n_nonnull (anchor+proxy groups); "
            f"got n_nonnull={n_nonnull}."
        )
    n_groups = n_nonnull // 2
    noise_scale = cfg.get("noise_scale", 0.3)

    # define the true signals and correlated dependents
    signal, latents = [], []
    for _ in range(n_groups):
        z = rng.standard_normal(n)
        latents.append(z)
        signal.append(z)  # anchor
        signal.append(z + noise_scale * rng.standard_normal(n))  # proxy

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "dependent_features", latents)
    feature_names = _make_feature_names(n_nonnull, n_features)
    proxy_null = ["conditional", "risk", "functional", "causal"]
    null_type = {}
    for j in range(n_nonnull):
        null_type[f"x{j + 1}"] = [] if j % 2 == 0 else list(proxy_null)
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
    However, the response depends on these features through x_j^2, so they are
    relevant from functional, conditional, and causal views. This example comes
    from Zheng and Raskutti ("Comparing Model-agnostic Feature Selection Methods
    through Relative Efficiency",Example 2.1.1). The "-1" recenters x_j^2 (whose
    mean is 1) so the logit is zero-mean like the other DGPs.
    """
    n_nonnull, n_features, _ = _dim(cfg)
    signal = [rng.standard_normal(n) for _ in range(n_nonnull)]

    noise, y, response_type, sigma_y = _simulate_response(n, rng, cfg, "quadratic", signal)
    feature_names = _make_feature_names(n_nonnull, n_features)
    null_type = {f"x{j + 1}": ["marginal"] for j in range(n_nonnull)}
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
