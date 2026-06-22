from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simulation_data import (
    build_filename,
    generate_linear_additive_data,
    generate_product_interaction_data,
    generate_reference_data,
    generate_xor_data,
    load_npz_dataset,
    make_covariance,
    save_npz_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_SEED = 123


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_minimal_sweep(output_dir: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/generate_simulation_data.py",
            "--minimal",
            "--seed",
            str(TEST_SEED),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
    )


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (generate_linear_additive_data, {}),
        (generate_xor_data, {}),
        (generate_product_interaction_data, {}),
    ],
)
def test_generator_contract_reproducibility_and_seed_sensitivity(generator, kwargs):
    n, p = 64, 10
    X, y, y_mean, metadata = generator(n=n, p=p, seed=TEST_SEED, **kwargs)

    assert X.shape == (n, p)
    assert y.shape == (n,)
    assert y_mean.shape == (n,)
    assert X.dtype == np.float64
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.isfinite(y_mean).all()
    assert metadata["n"] == n
    assert metadata["p"] == p
    assert metadata["seed"] == TEST_SEED

    X2, y2, y_mean2, _ = generator(n=n, p=p, seed=TEST_SEED, **kwargs)
    assert np.array_equal(X, X2)
    assert np.array_equal(y, y2)
    assert np.array_equal(y_mean, y_mean2)

    X_next, _, _, _ = generator(n=n, p=p, seed=TEST_SEED + 1, **kwargs)
    assert not np.array_equal(X, X_next)


def test_full_sweep_reproducibility_and_sweep_config(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run_minimal_sweep(out_a)
    _run_minimal_sweep(out_b)

    files_a = sorted(path.relative_to(out_a) for path in out_a.rglob("*") if path.is_file())
    files_b = sorted(path.relative_to(out_b) for path in out_b.rglob("*") if path.is_file())
    assert files_a == files_b
    assert {path: _digest_file(out_a / path) for path in files_a} == {
        path: _digest_file(out_b / path) for path in files_b
    }

    with (out_a / "sweep_config.json").open() as file:
        config = json.load(file)
    assert config == {
        "seed": TEST_SEED,
        "dataset_types": ["linear_additive", "xor", "product_interaction"],
        "sample_sizes": [20],
        "dimensions": [10],
        "loop_order": "dataset_type -> n -> p",
    }

    for dataset_type in config["dataset_types"]:
        path = out_a / dataset_type / build_filename(dataset_type, 20, 10, TEST_SEED)
        _, _, _, metadata = load_npz_dataset(path)
        assert metadata["seed"] == TEST_SEED


def test_save_load_round_trip(tmp_path):
    X, y, y_mean, metadata = generate_product_interaction_data(
        n=32,
        p=10,
        seed=TEST_SEED,
    )
    path = tmp_path / build_filename("product_interaction", 32, 10, TEST_SEED)

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
    X, _, y_mean, metadata = generate_linear_additive_data(
        n=50,
        p=6,
        beta=beta,
        seed=TEST_SEED,
    )
    assert metadata["relevant_features"] == [0, 1, 2]
    assert metadata["additive_features"] == [0, 1, 2]
    assert metadata["interaction_features"] == []
    assert metadata["noise_features"] == [3, 4, 5]
    assert np.allclose(y_mean, X[:, : len(beta)] @ np.asarray(beta))
    with pytest.raises(ValueError, match="p must be at least len"):
        generate_linear_additive_data(n=10, p=2, beta=beta, seed=TEST_SEED)


def test_xor_classification_data():
    n, p = 1000, 5
    X, y, y_mean, metadata = generate_xor_data(n=n, p=p, seed=TEST_SEED)
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})
    assert np.array_equal(y_mean, (X[:, 0] != X[:, 1]).astype(float))
    assert metadata["relevant_features"] == [0, 1]
    assert metadata["additive_features"] == []
    assert metadata["interaction_features"] == [[0, 1]]
    assert metadata["noise_feature_type"] == "bernoulli"
    assert metadata["task_type"] == "classification"
    assert metadata["sigma"] == 0.0

    with pytest.raises(ValueError, match="at least 2"):
        generate_xor_data(n=10, p=1)


def test_product_interaction_formula_modes():
    n, p = 80, 6
    X, _, y_mean, metadata = generate_product_interaction_data(
        n=n,
        p=p,
        seed=TEST_SEED,
        gamma=2.5,
    )
    assert metadata["additive_features"] == []
    assert metadata["interaction_features"] == [[0, 1]]
    assert np.allclose(y_mean, 2.5 * X[:, 0] * X[:, 1])

    X_default, _, y_mean_default, metadata_default = generate_product_interaction_data(
        n=n,
        p=p,
        seed=TEST_SEED,
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
        seed=TEST_SEED,
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
    X, _, y_mean, metadata = generate_reference_data(
        n=40,
        p=9,
        seed=TEST_SEED,
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
        generate_reference_data(n=10, p=7)


def test_build_filename_encodes_nondefault_parameters():
    assert build_filename("xor", 200, 10, 0) == "xor_n200_p10_seed0.npz"
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
