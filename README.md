# axiomatic_interpretability

## Simulation data

This repository includes synthetic generators for the Section 8 simulation
experiments:

- `linear_additive`: Gaussian features with
  `y = 4*X0 + 4*X1 + 3*X2 + 2*X3 + epsilon` by default. Features `0..3`
  are relevant additive features; all later columns are pure noise.
- `xor`: binary XOR signal with `y_mean = 1{X0 != X1}`. Features `0` and
  `1` are the only relevant interaction features; all later columns are
  Bernoulli noise. XOR is generated as classification data only.
- `product_interaction`: Gaussian features with
  `y = gamma * X0 * X1 + epsilon` by default. Features `0` and `1` are the
  relevant interaction features; optional main effects can be enabled.

The default main sweep varies:

- `n in [200, 500, 1000, 5000]`
- `p in [10, 20, 50, 100]`

Generate a small smoke-test sweep:

```bash
python scripts/generate_simulation_data.py --minimal
```

Generate the full sweep:

```bash
python scripts/generate_simulation_data.py
```

Outputs are written under
`data/simulations/{dataset_type}/`, with `data/simulations/sweep_config.json`
recording the seed, grid, and loop order. Existing output files are skipped by
default; pass `--overwrite` to regenerate them in place.
