"""Factory functions for axiom_interp explainers used in simulation benchmarks.

The benchmark refers to methods by stable config names. This module translates
those names into concrete `axiom_interp.presets` explainers with a shared
background sample and baseline for each dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from axiom_interp import aggregate, presets


DEFAULT_METHODS = (
    "baseline_shap",
    "baseline_minshap",
    "marginal_shap",
    "marginal_minshap",
    "integrated_gradients",
)

ALIASES = {
    "shap": "marginal_shap",
    "minshap": "marginal_minshap",
    "ig": "integrated_gradients",
}


@dataclass(frozen=True)
class ExplainerSpec:
    """A built explainer plus the context needed for benchmark metrics."""

    name: str
    explainer: object
    baseline: np.ndarray
    background: np.ndarray


def build_explainers(
    *,
    method_names: list[str] | tuple[str, ...] | None,
    X: np.ndarray,
    metadata: dict[str, Any],
    background_size: int,
    n_orderings: int,
    n_steps: int,
    seed: int,
    baseline_strategy: str = "zeros",
    include_unsupported: bool = False,
) -> list[ExplainerSpec]:
    """Build configured explainers for one generated dataset."""

    X_arr = np.asarray(X, dtype=float)
    background = select_background(X_arr, background_size=background_size, seed=seed)
    baseline = make_baseline(X_arr, strategy=baseline_strategy)
    requested = method_names or list(DEFAULT_METHODS)

    specs: list[ExplainerSpec] = []
    for raw_name in requested:
        name = canonical_method_name(raw_name)
        if not include_unsupported and not supports_dataset(name, metadata):
            continue
        explainer = _build_one(
            name=name,
            background=background,
            baseline=baseline,
            n_orderings=n_orderings,
            n_steps=n_steps,
            seed=seed,
        )
        specs.append(
            ExplainerSpec(
                name=name,
                explainer=explainer,
                baseline=baseline,
                background=background,
            )
        )
    return specs


def canonical_method_name(name: str) -> str:
    """Normalize user-facing aliases from configs into canonical method names."""

    canonical = ALIASES.get(name, name)
    valid = {
        "baseline_shap",
        "baseline_minshap",
        "marginal_shap",
        "marginal_minshap",
        "integrated_gradients",
    }
    if canonical not in valid:
        raise ValueError(f"Unknown attribution method {name!r}. Valid methods: {sorted(valid)}")
    return canonical


def supports_dataset(method_name: str, metadata: dict[str, Any]) -> bool:
    """Return whether a method is enabled by default for this dataset."""

    if method_name == "integrated_gradients" and metadata.get("task_type") == "classification":
        return False
    return True


def select_background(X: np.ndarray, *, background_size: int, seed: int) -> np.ndarray:
    """Select a deterministic background subset for marginal methods."""

    X_arr = np.asarray(X, dtype=float)
    n = len(X_arr)
    size = min(int(background_size), n)
    if size < 1:
        raise ValueError("background_size must select at least one row.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=size, replace=False)
    return X_arr[np.sort(idx)]


def make_baseline(X: np.ndarray, *, strategy: str) -> np.ndarray:
    """Construct the fixed baseline used by baseline SHAP and IG."""

    X_arr = np.asarray(X, dtype=float)
    if strategy == "zeros":
        return np.zeros(X_arr.shape[1], dtype=float)
    if strategy == "mean":
        return np.mean(X_arr, axis=0)
    raise ValueError("baseline_strategy must be 'zeros' or 'mean'.")


def _build_one(
    *,
    name: str,
    background: np.ndarray,
    baseline: np.ndarray,
    n_orderings: int,
    n_steps: int,
    seed: int,
):
    if name == "baseline_shap":
        return presets.baseline_shap(baseline, n_orderings=n_orderings, seed=seed)
    if name == "baseline_minshap":
        return presets.baseline_shap(baseline, n_orderings=n_orderings, seed=seed).replace(
            aggregator=aggregate.Min()
        )
    if name == "marginal_shap":
        return presets.shap(background, n_orderings=n_orderings, seed=seed)
    if name == "marginal_minshap":
        return presets.minshap(background, n_orderings=n_orderings, seed=seed)
    if name == "integrated_gradients":
        return presets.integrated_gradients(baseline, n_steps=n_steps)
    raise ValueError(f"Unknown attribution method {name!r}.")
