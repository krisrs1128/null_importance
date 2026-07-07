"""Synthetic simulation data generators for Section 8 experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def make_covariance(
    p: int,
    corr: float = 0.0,
    structure: str = "independent",
    block_size: int = 2,
) -> np.ndarray:
    """Return a p x p covariance matrix with unit diagonal."""

    p = int(p)
    corr = float(corr)
    if p < 1:
        raise ValueError("p must be positive.")

    if structure == "independent":
        if corr != 0.0:
            raise ValueError("corr must be 0.0 for independent covariance.")
        return np.eye(p, dtype=np.float64)

    if structure == "equicorrelated":
        _check_corr_bounds(corr, p)
        sigma = np.full((p, p), corr, dtype=np.float64)
        np.fill_diagonal(sigma, 1.0)
        return sigma

    if structure == "block":
        block_size = int(block_size)
        sigma = np.eye(p, dtype=np.float64)
        for start in range(0, p, block_size):
            stop = min(start + block_size, p)
            block_dim = stop - start
            if block_dim > 1:
                _check_corr_bounds(corr, block_dim)
                sigma[start:stop, start:stop] = corr
                np.fill_diagonal(sigma[start:stop, start:stop], 1.0)
        return sigma


def _check_corr_bounds(corr: float, dim: int) -> None:
    if dim <= 1:
        return
    lower = -1.0 / (dim - 1)
    if not (lower < corr < 1.0):
        raise ValueError(
            f"corr={corr} is not positive-definite for dimension {dim}; "
            f"requires {lower} < corr < 1."
        )


def generate_linear_additive_data(
    n: int,
    p: int,
    beta: list[float] | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
    corr: float = 0.0,
    structure: str = "independent",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate data with additive linear effects on the first len(beta) features."""

    beta_arr = np.asarray([4.0, 4.0, 3.0, 2.0] if beta is None else beta, dtype=float)
    n, p = int(n), int(p)
    if p < len(beta_arr):
        raise ValueError("p must be at least len(beta).")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    covariance = make_covariance(p, corr=corr, structure=structure)
    X = np.random.multivariate_normal(np.zeros(p), covariance, size=n)
    y_mean = X[:, : len(beta_arr)] @ beta_arr
    epsilon = np.random.normal(0.0, float(sigma), size=n)
    y = y_mean + epsilon

    metadata = _base_metadata(
        dataset_type="linear_additive",
        n=n,
        p=p,
        seed=seed,
        sigma=sigma,
        relevant_features=list(range(len(beta_arr))),
        additive_features=list(range(len(beta_arr))),
        interaction_features=[],
        noise_features=list(range(len(beta_arr), p)),
        data_generating_equation=(
            "y = sum_j beta[j] * X_j + epsilon, "
            "j in additive_features; epsilon ~ N(0, sigma^2)"
        ),
        covariance_structure=structure,
        corr=corr,
        block_size=None,
        task_type="regression",
    )
    metadata["beta"] = beta_arr.tolist()
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(y_mean, dtype=np.float64),
        metadata,
    )


def generate_xor_data(
    n: int,
    p: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate XOR classification data."""

    n, p = int(n), int(p)
    if p < 2:
        raise ValueError("p must be at least 2 for XOR data.")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    signal_features = np.random.binomial(1, 0.5, size=(n, 2))
    noise_features = np.random.binomial(1, 0.5, size=(n, p - 2))
    X = np.hstack([signal_features, noise_features]).astype(np.float64)

    y_mean = (X[:, 0] != X[:, 1]).astype(np.float64)
    y = y_mean.astype(np.int64)

    metadata = _base_metadata(
        dataset_type="xor",
        n=n,
        p=p,
        seed=seed,
        sigma=0.0,
        relevant_features=[0, 1],
        additive_features=[],
        interaction_features=[[0, 1]],
        noise_features=list(range(2, p)),
        data_generating_equation="y = 1{X0 != X1}",
        covariance_structure="independent",
        corr=0.0,
        block_size=None,
        task_type="classification",
    )
    metadata["noise_feature_type"] = "bernoulli"
    return X, y, y_mean, metadata


def generate_product_interaction_data(
    n: int,
    p: int,
    gamma: float = 3.0,
    beta_main: list[float] | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
    corr: float = 0.0,
    structure: str = "independent",
    include_main_effects: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate data with a product interaction between the first two features."""

    n, p = int(n), int(p)
    gamma = float(gamma)
    if p < 2:
        raise ValueError("p must be at least 2 for product interaction data.")
    if include_main_effects and beta_main is None:
        beta_main = [4.0, 4.0]
    if not include_main_effects and beta_main is not None:
        raise ValueError(
            "beta_main was provided but include_main_effects=False; "
            "set include_main_effects=True to use it, or omit beta_main."
        )
    if beta_main is not None and len(beta_main) != 2:
        raise ValueError("beta_main must contain exactly two values.")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    covariance = make_covariance(p, corr=corr, structure=structure)
    X = np.random.multivariate_normal(np.zeros(p), covariance, size=n)
    y_mean = gamma * X[:, 0] * X[:, 1]
    if include_main_effects:
        beta_main_arr = np.asarray(beta_main, dtype=float)
        y_mean = y_mean + beta_main_arr[0] * X[:, 0] + beta_main_arr[1] * X[:, 1]
    epsilon = np.random.normal(0.0, float(sigma), size=n)
    y = y_mean + epsilon

    metadata = _base_metadata(
        dataset_type="product_interaction",
        n=n,
        p=p,
        seed=seed,
        sigma=sigma,
        relevant_features=[0, 1],
        additive_features=[0, 1] if include_main_effects else [],
        interaction_features=[[0, 1]],
        noise_features=list(range(2, p)),
        data_generating_equation=(
            "y = gamma * X0 * X1 + epsilon"
            if not include_main_effects
            else "y = beta_main[0] * X0 + beta_main[1] * X1 + gamma * X0 * X1 + epsilon"
        ),
        covariance_structure=structure,
        corr=corr,
        block_size=None,
        task_type="regression",
    )
    metadata["gamma"] = gamma
    metadata["include_main_effects"] = bool(include_main_effects)
    metadata["beta_main"] = None if beta_main is None else [float(v) for v in beta_main]
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(y_mean, dtype=np.float64),
        metadata,
    )


def generate_reference_data(
    n: int,
    p: int,
    beta: list[float] | None = None,
    sigma: float = 1.0,
    seed: int | None = None,
    corr: float = 0.5,
    structure: str = "block",
    block_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the optional eight-feature additive-plus-interaction reference data."""

    beta_arr = np.asarray(
        [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0] if beta is None else beta,
        dtype=float,
    )
    n, p = int(n), int(p)
    if p < 8:
        raise ValueError("p must be at least 8 for reference data.")
    if len(beta_arr) != 8:
        raise ValueError("beta must contain exactly eight values.")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    covariance = make_covariance(
        p,
        corr=corr,
        structure=structure,
        block_size=block_size,
    )
    X = np.random.multivariate_normal(np.zeros(p), covariance, size=n)
    y_mean = (
        beta_arr[0] * X[:, 0]
        + beta_arr[1] * X[:, 1]
        + beta_arr[2] * X[:, 2] * X[:, 3]
        + beta_arr[3] * X[:, 4]
        + beta_arr[4] * X[:, 5]
        + beta_arr[5] * X[:, 4] * X[:, 5]
        + beta_arr[6] * X[:, 6]
        + beta_arr[7] * X[:, 7]
    )
    epsilon = np.random.normal(0.0, float(sigma), size=n)
    y = y_mean + epsilon

    additive_features = [0, 1, 4, 5, 6, 7]
    metadata = _base_metadata(
        dataset_type="reference_linear_interaction",
        n=n,
        p=p,
        seed=seed,
        sigma=sigma,
        relevant_features=sorted(set(additive_features) | {2, 3, 4, 5}),
        additive_features=additive_features,
        interaction_features=[[2, 3], [4, 5]],
        noise_features=list(range(8, p)),
        data_generating_equation=(
            "y = beta0*X0 + beta1*X1 + beta2*X2*X3 + beta3*X4 + "
            "beta4*X5 + beta5*X4*X5 + beta6*X6 + beta7*X7 + epsilon"
        ),
        covariance_structure=structure,
        corr=corr,
        block_size=block_size if structure == "block" else None,
        task_type="regression",
    )
    metadata["beta"] = beta_arr.tolist()
    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(y_mean, dtype=np.float64),
        metadata,
    )


def save_csv_dataset(
    X: np.ndarray,
    y: np.ndarray,
    y_mean: np.ndarray,
    metadata: dict[str, Any],
    path: str | Path,
) -> None:
    """Save one generated dataset as CSV plus a JSON metadata sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    y_mean_arr = np.asarray(y_mean, dtype=np.float64)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y_arr.shape != (X_arr.shape[0],):
        raise ValueError("y must be a 1D array with one value per row of X.")
    if y_mean_arr.shape != (X_arr.shape[0],):
        raise ValueError("y_mean must be a 1D array with one value per row of X.")

    feature_names = [f"x{j}" for j in range(X_arr.shape[1])]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([*feature_names, "y", "y_mean"])
        for row, y_value, y_mean_value in zip(X_arr, y_arr, y_mean_arr):
            writer.writerow([*row.tolist(), y_value, y_mean_value])

    with metadata_path_for(path).open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")


def load_csv_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a dataset saved by save_csv_dataset."""

    path = Path(path)
    with metadata_path_for(path).open() as file:
        metadata = json.load(file)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        feature_names = _feature_columns(reader.fieldnames)
        if "y" not in reader.fieldnames or "y_mean" not in reader.fieldnames:
            raise ValueError(f"{path} must contain 'y' and 'y_mean' columns.")
        rows = list(reader)

    X = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    if metadata.get("task_type") == "classification":
        y = np.asarray([int(float(row["y"])) for row in rows], dtype=np.int64)
    else:
        y = np.asarray([float(row["y"]) for row in rows], dtype=np.float64)
    y_mean = np.asarray([float(row["y_mean"]) for row in rows], dtype=np.float64)
    return X, y, y_mean, metadata


def metadata_path_for(path: str | Path) -> Path:
    """Return the JSON sidecar path for a simulation data CSV."""

    return Path(path).with_suffix(".metadata.json")


def build_filename(
    dataset_type: str,
    n: int,
    p: int,
    seed: int,
    **nondefault_kwargs: Any,
) -> str:
    """Build a collision-resistant filename for a simulation config."""

    stem = f"{dataset_type}_n{int(n)}_p{int(p)}_seed{int(seed)}"
    extras: list[str] = []
    display_names = {
        "beta_main": "betamain",
        "include_main_effects": "includemain",
    }
    preferred_order = [
        "corr",
        "sigma",
        "gamma",
        "beta",
        "beta_main",
        "include_main_effects",
    ]
    keys = [key for key in preferred_order if key in nondefault_kwargs]
    keys.extend(sorted(key for key in nondefault_kwargs if key not in preferred_order))
    for key in keys:
        value = nondefault_kwargs[key]
        if value is None:
            continue
        name = display_names.get(key, key)
        extras.append(f"{name}{_format_filename_value(value)}")
    if extras:
        stem = f"{stem}__{'_'.join(extras)}"
    return f"{stem}.csv"


def _feature_columns(fieldnames: list[str]) -> list[str]:
    feature_names = [name for name in fieldnames if name.startswith("x") and name[1:].isdigit()]
    if not feature_names:
        raise ValueError("CSV must contain feature columns named x0, x1, ...")
    return sorted(feature_names, key=lambda name: int(name[1:]))


def _format_filename_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_format_filename_value(v) for v in value)
    return str(value).replace(" ", "")


def _base_metadata(
    *,
    dataset_type: str,
    n: int,
    p: int,
    seed: int | None,
    sigma: float,
    relevant_features: list[int],
    additive_features: list[int],
    interaction_features: list[list[int]],
    noise_features: list[int],
    data_generating_equation: str,
    covariance_structure: str,
    corr: float,
    block_size: int | None,
    task_type: str,
) -> dict[str, Any]:
    return {
        "dataset_type": dataset_type,
        "n": int(n),
        "p": int(p),
        "seed": None if seed is None else int(seed),
        "sigma": float(sigma),
        "feature_indexing": "zero_based",
        "relevant_features": [int(v) for v in relevant_features],
        "additive_features": [int(v) for v in additive_features],
        "interaction_features": [[int(a), int(b)] for a, b in interaction_features],
        "noise_features": [int(v) for v in noise_features],
        "data_generating_equation": data_generating_equation,
        "covariance_structure": covariance_structure,
        "corr": float(corr),
        "block_size": None if block_size is None else int(block_size),
        "task_type": task_type,
    }
