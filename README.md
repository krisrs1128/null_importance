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
