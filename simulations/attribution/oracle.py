"""Oracle functions for generated simulation datasets.

Generated CSV files store arrays while JSON sidecars store metadata, not Python
callables. These helpers rebuild the data-generating function from metadata so
attribution runs can test the known ground-truth mechanism directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ArrayFunction = Callable[[np.ndarray], np.ndarray]


def function_from_metadata(metadata: dict[str, Any]) -> ArrayFunction:
    """Return the oracle f(X) encoded by a simulation metadata dictionary.

    The generated CSV stores sampled rows, while the sidecar stores enough
    parameters to reconstruct the noiseless target `y_mean`.
    """

    dataset_type = metadata["dataset_type"]
    if dataset_type == "linear_additive":
        beta = np.asarray(metadata["beta"], dtype=float)

        def f(X: np.ndarray) -> np.ndarray:
            X_arr = _as_2d(X)
            return X_arr[:, : len(beta)] @ beta

        return f

    if dataset_type == "product_interaction":
        gamma = float(metadata["gamma"])
        include_main_effects = bool(metadata.get("include_main_effects", False))
        beta_main = metadata.get("beta_main")
        beta_main_arr = (
            np.asarray([4.0, 4.0], dtype=float)
            if include_main_effects and beta_main is None
            else np.asarray(beta_main or [0.0, 0.0], dtype=float)
        )

        def f(X: np.ndarray) -> np.ndarray:
            X_arr = _as_2d(X)
            y_mean = gamma * X_arr[:, 0] * X_arr[:, 1]
            if include_main_effects:
                y_mean = y_mean + beta_main_arr[0] * X_arr[:, 0] + beta_main_arr[1] * X_arr[:, 1]
            return y_mean

        return f

    if dataset_type == "xor":

        def f(X: np.ndarray) -> np.ndarray:
            X_arr = _as_2d(X)
            return (X_arr[:, 0] != X_arr[:, 1]).astype(float)

        return f

    if dataset_type == "reference_linear_interaction":
        beta = np.asarray(metadata["beta"], dtype=float)

        def f(X: np.ndarray) -> np.ndarray:
            X_arr = _as_2d(X)
            return (
                beta[0] * X_arr[:, 0]
                + beta[1] * X_arr[:, 1]
                + beta[2] * X_arr[:, 2] * X_arr[:, 3]
                + beta[3] * X_arr[:, 4]
                + beta[4] * X_arr[:, 5]
                + beta[5] * X_arr[:, 4] * X_arr[:, 5]
                + beta[6] * X_arr[:, 6]
                + beta[7] * X_arr[:, 7]
            )

        return f

    raise ValueError(f"Unsupported simulation dataset_type {dataset_type!r}.")


def max_oracle_error(f: ArrayFunction, X: np.ndarray, y_mean: np.ndarray) -> float:
    """Return max absolute error between rebuilt f(X) and stored y_mean."""

    return float(np.max(np.abs(np.asarray(f(X), dtype=float) - np.asarray(y_mean, dtype=float))))


def _as_2d(X: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(X, dtype=float))
