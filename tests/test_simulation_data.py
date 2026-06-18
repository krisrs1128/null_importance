from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from simulation_data import (
    RNG_SEED_SEMANTICS,
    build_filename,
    derive_seed,
    generate_linear_additive_data,
    generate_product_interaction_data,
    generate_reference_linear_interaction_data,
    generate_xor_data,
    load_npz_dataset,
    make_covariance,
    save_npz_dataset,
)


MASTER_SEED = 20260618
ROOT = Path(__file__).resolve().parents[1]


def _stamp(metadata: dict, master_seed: int, rep_index: int) -> dict:
    metadata["master_seed"] = master_seed
    metadata["rep_index"] = rep_index
    return metadata


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.mark.parametrize(
    ("dataset_type", "generator", "kwargs"),
    [
        ("linear_additive", generate_linear_additive_data, {}),
        ("xor", generate_xor_data, {}),
        ("product_interaction", generate_product_interaction_data, {}),
    ],
)
def test_universal_contract_and_reproducibility(dataset_type, generator, kwargs):
    n, p, rep_index = 64, 10, 0
    seed = derive_seed(MASTER_SEED, dataset_type, n, p, rep_index)
    X, y, y_mean, metadata = generator(n=n, p=p, seed=seed, **kwargs)
    metadata = _stamp(metadata, MASTER_SEED, rep_index)

    assert X.shape == (n, p)
    assert y.shape == (n,)
    assert y_mean.shape == (n,)
    assert X.dtype == np.float64
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.isfinite(y_mean).all()
    assert metadata["n"] == n
    assert metadata["p"] == p
    assert metadata["master_seed"] == MASTER_SEED
    assert metadata["rep_index"] == rep_index
    assert metadata["seed"] == seed
    assert metadata["rng_seed_semantics"] == RNG_SEED_SEMANTICS

    X2, y2, y_mean2, _ = generator(n=n, p=p, seed=seed, **kwargs)
    assert np.array_equal(X, X2)
    assert np.array_equal(y, y2)
    assert np.array_equal(y_mean, y_mean2)

    rep_seed = derive_seed(MASTER_SEED, dataset_type, n, p, rep_index + 1)
    X_rep, _, _, _ = generator(n=n, p=p, seed=rep_seed, **kwargs)
    assert not np.array_equal(X, X_rep)

    master_seed = derive_seed(MASTER_SEED + 1, dataset_type, n, p, rep_index)
    X_master, _, _, _ = generator(n=n, p=p, seed=master_seed, **kwargs)
    assert not np.array_equal(X, X_master)


def test_cross_process_seed_derivation_is_stable():
    dataset_type, n, p, rep_index = "linear_additive", 64, 10, 0
    seed = derive_seed(MASTER_SEED, dataset_type, n, p, rep_index)
    X, y, y_mean, _ = generate_linear_additive_data(n=n, p=p, seed=seed)
    expected = {
        "seed": seed,
        "X": _digest(X),
        "y": _digest(y),
        "y_mean": _digest(y_mean),
    }

    code = f"""
import hashlib
import json
from simulation_data import derive_seed, generate_linear_additive_data

def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()

seed = derive_seed({MASTER_SEED}, {dataset_type!r}, {n}, {p}, {rep_index})
X, y, y_mean, _ = generate_linear_additive_data(n={n}, p={p}, seed=seed)
print(json.dumps({{"seed": seed, "X": digest(X), "y": digest(y), "y_mean": digest(y_mean)}}))
"""
    observed = json.loads(
        subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True)
    )
    assert observed == expected


def test_derived_seeds_are_pairwise_distinct_for_sampled_grid():
    dataset_types = ["linear_additive", "xor", "product_interaction"]
    ns = [20, 50, 200, 500, 1000]
    ps = [2, 4, 10, 20, 50]
    reps = [0, 1, 2, 3, 4]
    configs = list(product(dataset_types, ns, ps, reps))[:100]
    seeds = [derive_seed(MASTER_SEED, d, n, p, r) for d, n, p, r in configs]
    assert len(seeds) == 100
    assert len(set(seeds)) == len(seeds)


def test_save_load_round_trip(tmp_path):
    dataset_type, n, p, rep_index = "product_interaction", 32, 10, 0
    seed = derive_seed(MASTER_SEED, dataset_type, n, p, rep_index)
    X, y, y_mean, metadata = generate_product_interaction_data(n=n, p=p, seed=seed)
    metadata = _stamp(metadata, MASTER_SEED, rep_index)
    path = tmp_path / build_filename(dataset_type, n, p, rep_index)

    save_npz_dataset(X, y, y_mean, metadata, path)
    X2, y2, y_mean2, metadata2 = load_npz_dataset(path)

    assert np.array_equal(X, X2)
    assert np.array_equal(y, y2)
    assert np.array_equal(y_mean, y_mean2)
    assert metadata2 == metadata


def test_make_covariance_contracts():
    assert np.array_equal(make_covariance(3), np.eye(3))
    with pytest.raises(ValueError, match="corr must be 0.0"):
        make_covariance(3, corr=0.1, structure="independent")
    equicorr = make_covariance(3, corr=0.25, structure="equicorrelated")
    assert np.allclose(np.diag(equicorr), 1.0)
    assert np.allclose(equicorr[0, 1], 0.25)
    block = make_covariance(5, corr=0.5, structure="block", block_size=2)
    assert block[0, 1] == 0.5
    assert block[2, 3] == 0.5
    assert block[1, 2] == 0.0
    with pytest.raises(ValueError, match="positive-definite"):
        make_covariance(3, corr=-0.5, structure="equicorrelated")


def test_linear_additive_formula_and_metadata():
    beta = [1.0, -2.0, 0.5]
    n, p = 50, 6
    seed = derive_seed(MASTER_SEED, "linear_additive", n, p, 0)
    X, _, y_mean, metadata = generate_linear_additive_data(
        n=n,
        p=p,
        beta=beta,
        seed=seed,
    )
    assert metadata["relevant_features"] == [0, 1, 2]
    assert metadata["additive_features"] == [0, 1, 2]
    assert metadata["interaction_features"] == []
    assert metadata["noise_features"] == [3, 4, 5]
    assert np.allclose(y_mean, X[:, : len(beta)] @ np.asarray(beta))
    with pytest.raises(ValueError, match="p must be at least len"):
        generate_linear_additive_data(n=10, p=2, beta=beta, seed=seed)


def test_xor_classification_and_regression_modes():
    n, p = 1000, 5
    seed = derive_seed(MASTER_SEED, "xor", n, p, 0)
    X, y, y_mean, metadata = generate_xor_data(n=n, p=p, seed=seed)
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})
    assert np.array_equal(y_mean, (X[:, 0] != X[:, 1]).astype(float))
    assert metadata["relevant_features"] == [0, 1]
    assert metadata["additive_features"] == []
    assert metadata["interaction_features"] == [[0, 1]]
    assert metadata["noise_feature_type"] == "bernoulli"

    sigma = 1.25
    reg_seed = derive_seed(MASTER_SEED, "xor", n, p, 1)
    X_reg, y_reg, y_mean_reg, metadata_reg = generate_xor_data(
        n=n,
        p=p,
        seed=reg_seed,
        task_type="regression",
        sigma=sigma,
    )
    assert y_reg.dtype == np.float64
    assert np.isclose(np.std(y_reg - y_mean_reg), sigma, rtol=0.2)
    assert np.array_equal(y_mean_reg, (X_reg[:, 0] != X_reg[:, 1]).astype(float))
    assert metadata_reg["noise_feature_type"] == "gaussian"
    assert not np.isin(X_reg[:, 2:], [0.0, 1.0]).all()

    with pytest.raises(ValueError, match="at least 2"):
        generate_xor_data(n=10, p=1)
    with pytest.raises(ValueError, match="sigma must be 0.0"):
        generate_xor_data(n=10, p=2, sigma=0.1, task_type="classification")


def test_product_interaction_formula_modes():
    n, p = 80, 6
    seed = derive_seed(MASTER_SEED, "product_interaction", n, p, 0)
    X, _, y_mean, metadata = generate_product_interaction_data(
        n=n,
        p=p,
        seed=seed,
        gamma=2.5,
    )
    assert metadata["additive_features"] == []
    assert metadata["interaction_features"] == [[0, 1]]
    assert np.allclose(y_mean, 2.5 * X[:, 0] * X[:, 1])

    X_default, _, y_mean_default, metadata_default = generate_product_interaction_data(
        n=n,
        p=p,
        seed=seed,
        include_main_effects=True,
    )
    assert metadata_default["additive_features"] == [0, 1]
    assert metadata_default["beta_main"] == [4.0, 4.0]
    assert np.allclose(
        y_mean_default,
        4.0 * X_default[:, 0]
        + 4.0 * X_default[:, 1]
        + 3.0 * X_default[:, 0] * X_default[:, 1],
    )

    X_explicit, _, y_mean_explicit, _ = generate_product_interaction_data(
        n=n,
        p=p,
        seed=seed,
        gamma=1.5,
        include_main_effects=True,
        beta_main=[1.0, -2.0],
    )
    assert np.allclose(
        y_mean_explicit,
        1.0 * X_explicit[:, 0]
        - 2.0 * X_explicit[:, 1]
        + 1.5 * X_explicit[:, 0] * X_explicit[:, 1],
    )
    with pytest.raises(ValueError, match="at least 2"):
        generate_product_interaction_data(n=10, p=1)
    with pytest.raises(ValueError, match="include_main_effects=False"):
        generate_product_interaction_data(n=10, p=3, beta_main=[1.0, 2.0])


def test_reference_linear_interaction_formula():
    n, p = 40, 9
    seed = derive_seed(MASTER_SEED, "reference_linear_interaction", n, p, 0)
    X, _, y_mean, metadata = generate_reference_linear_interaction_data(
        n=n,
        p=p,
        seed=seed,
    )
    beta = np.asarray(metadata["beta"])
    expected = (
        beta[0] * X[:, 0]
        + beta[1] * X[:, 1]
        + beta[2] * X[:, 2] * X[:, 3]
        + beta[3] * X[:, 4]
        + beta[4] * X[:, 5]
        + beta[5] * X[:, 4] * X[:, 5]
        + beta[6] * X[:, 6]
        + beta[7] * X[:, 7]
    )
    assert metadata["dataset_type"] == "reference_linear_interaction"
    assert metadata["additive_features"] == [0, 1, 4, 5, 6, 7]
    assert metadata["interaction_features"] == [[2, 3], [4, 5]]
    assert metadata["relevant_features"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert metadata["noise_features"] == [8]
    assert np.allclose(y_mean, expected)
    with pytest.raises(ValueError, match="at least 8"):
        generate_reference_linear_interaction_data(n=10, p=7)


def test_build_filename_encodes_nondefault_parameters():
    assert (
        build_filename("xor", 200, 10, 0)
        == "xor_n200_p10_seed0.npz"
    )
    assert (
        build_filename(
            "product_interaction",
            1000,
            20,
            0,
            corr=0.3,
            include_main_effects=True,
        )
        == "product_interaction_n1000_p20_seed0__corr0.3_includemainTrue.npz"
    )
