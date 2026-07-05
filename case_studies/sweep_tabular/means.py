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


@register_mean("xor")
def mean_xor(cols, cfg):
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
