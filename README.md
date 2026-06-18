# axiomatic_interpretability

## Simulation data

This repository includes synthetic generators for the Section 8 simulation
experiments:

- `linear_additive`: Gaussian features with
  `y = 4*X0 + 4*X1 + 3*X2 + 2*X3 + epsilon` by default. Features `0..3`
  are relevant additive features; all later columns are pure noise.
- `xor`: binary XOR signal with `y_mean = 1{X0 != X1}`. Features `0` and
  `1` are the only relevant interaction features; all later columns are
  noise. Classification is the default, and regression is available with
  Gaussian response noise.
- `product_interaction`: Gaussian features with
  `y = gamma * X0 * X1 + epsilon` by default. Features `0` and `1` are the
  relevant interaction features; optional main effects can be enabled.

The default main sweep varies:

- `n in [200, 500, 1000, 5000]`
- `p in [10, 20, 50, 100]`
- `rep_index in [0, 1, 2, 3, 4]`

One `MASTER_SEED` controls the entire sweep. `rep_index` is a repetition
index, not a raw RNG seed. Each `(dataset_type, n, p, rep_index)` derives a
stable per-configuration seed with `hashlib.sha256` plus
`numpy.random.SeedSequence`, and every saved metadata dictionary records both
the `MASTER_SEED` and the derived seed.

Generate a small smoke-test sweep:

```bash
python scripts/generate_simulation_data.py --minimal
```

Generate the full sweep:

```bash
python scripts/generate_simulation_data.py
```

Outputs are written under
`data/simulations/{dataset_type}/master{MASTER_SEED}/`. Existing output files
are skipped by default; pass `--overwrite` to regenerate them in place.

The optional `generate_reference_linear_interaction_data` generator implements
the richer eight-feature additive-plus-interaction reference model, but it is
excluded from the default sweep.

### NPZ file format

Each generated `.npz` file is a compressed NumPy archive with four entries:

- `X`: feature matrix, shape `(n, p)`, dtype `float64`. For XOR data, binary
  features are still stored as floats for downstream compatibility.
- `y`: observed target, shape `(n,)`. Regression datasets use `float64`;
  XOR classification uses `int64` labels in `{0, 1}`.
- `y_mean`: noiseless structural signal, shape `(n,)`, dtype `float64`.
  For regression this is the target before adding noise; for XOR
  classification this is the clean XOR signal before integer casting.
- `metadata`: JSON string containing generation provenance and feature
  semantics.

Use `load_npz_dataset` to load the archive and parse `metadata`:

```python
from simulation_data import load_npz_dataset

X, y, y_mean, metadata = load_npz_dataset(
    "data/simulations/linear_additive/master20260618/"
    "linear_additive_n20_p10_seed0.npz"
)
```

Important metadata fields include:

- `dataset_type`: one of `linear_additive`, `xor`, `product_interaction`, or
  `reference_linear_interaction`.
- `n`, `p`: sample size and ambient feature dimension.
- `master_seed`, `rep_index`, `seed`: sweep-level seed, repetition index, and
  derived per-configuration RNG seed.
- `relevant_features`, `additive_features`, `interaction_features`,
  `noise_features`: zero-based feature-role annotations.
- `data_generating_equation`: human-readable equation used for the dataset.
- `task_type`: `regression` or `classification`.
- `covariance_structure`, `corr`, `block_size`: feature covariance settings.
