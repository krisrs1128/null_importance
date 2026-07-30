"""mean/logit functions for datasets.py and model.py.

The mean functions are the same for both classification and regression
synthetic datasets.
"""
from functools import partial

import numpy as np

MEAN_FNS = {}


def register_mean(name):
    """Decorator to register a mean function."""
    def decorator(fn):
        MEAN_FNS[name] = fn
        return fn
    return decorator


def _scaled_sum(key, default, cols, cfg):
    """mean = cfg[key] * sum(cols)."""
    return cfg.get(key, default) * sum(cols)


# linear_additive, dependent_features, and confounding all share this
# "coefficient * sum(cols)" mean; only the config key/default differ.
MEAN_FNS["linear_additive"] = partial(_scaled_sum, "beta", 4.0)
MEAN_FNS["dependent_features"] = partial(_scaled_sum, "gamma", 3.0)
MEAN_FNS["confounding"] = partial(_scaled_sum, "gamma", 3.0)


def chain_terminal_indices(n_nonnull, chain_length):
    """Zero-based terminal indices for contiguous chains of at most chain_length."""
    if chain_length < 1:
        raise ValueError(f"chain_length must be at least 1; got {chain_length}.")
    return [
        min(start + chain_length, n_nonnull) - 1
        for start in range(0, n_nonnull, chain_length)
    ]


@register_mean("mediated_chains")
def mean_mediated_chains(cols, cfg):
    """mean = gamma / sqrt(n_chains) * sum(terminal node of each chain)."""
    gamma = cfg.get("gamma", 3.0)
    return gamma / np.sqrt(len(cols)) * sum(cols)


@register_mean("parity")
def mean_parity(cols, cfg):
    """mean = -gamma * prod(sign(c) for c in cols)."""
    gamma = cfg.get("gamma", 3.0)
    signs = [np.sign(c) for c in cols]
    product_of_signs = np.prod(signs, axis=0)
    return -gamma * product_of_signs


@register_mean("product_interaction")
def mean_product_interaction(cols, cfg):
    """mean = gamma * sum_k(cols[2k] * cols[2k+1])."""
    gamma = cfg.get("gamma", 3.0)
    n_pairs = len(cols) // 2
    return gamma * sum(cols[2*k] * cols[2*k + 1] for k in range(n_pairs))


@register_mean("quadratic")
def mean_quadratic(cols, cfg):
    """mean = gamma * sum(c**2 - 1 for c in cols).

    Recentered by E[c**2]=1 (c ~ N(0,1)) so the logit is zero-mean.
    """
    gamma = cfg.get("gamma", 3.0)
    return gamma * sum(c**2 - 1 for c in cols)
